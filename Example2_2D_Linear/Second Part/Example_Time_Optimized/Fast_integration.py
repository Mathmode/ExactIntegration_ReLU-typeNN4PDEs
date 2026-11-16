#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast_integration.py  —  compiled version of the explicit neuron-wise training in
Deep_Ritz_2D_CMC/My_integration.py and Deep_Ritz_2D_CMC/train_u_1D.py (CMC Deep Ritz).

Same loss at every update and same order of updates.  No random selection: every
neuron interaction is taken explicitly.

  FB  (my_integration_FB) : one update with
                            J = 1/2 sum_ij r_i^T K_ij r_j - sum_i r_i^T F_i
  NW  (my_integration_NW) : for k = 1..N, one update with
                            J_k = 1/2 [ r_k^T K_kk r_k + 2 sum_{j>k} r_k^T K_kj r_j ] - r_k^T F_k
  DNW (my_integration_DNW): for k = 1..N and k1 = 1..N, one update with
                            J_kk1 = 1/2 r_k^T K_{k,k1} r_k1 - (1/N) r_k^T F_k     (K_kk if k1 = k)
                            (the 1/N makes one epoch sum to the FB energy)

Every update adds interior_regularization of train_u_1D.py.

Notation: r_i = c_i (w0_i, w1_i, b_i), so u_i = r_i . (x, y, 1) M with M = (x^2-x)(y^2-y), and
K_ij = r_i^T A_ij r_j with A_ij the integral of grad(phi).grad(phi) + alpha phi phi over
D_i ∩ {z_j >= 0}, D_i = Omega ∩ {z_i >= 0}.  Gradients flow through r_i and, with
geometry_gradient=True (default, paper Corollary 4.2), also through the clipped polygons: these are
the boundary terms of the energy.  geometry_gradient=False treats the polygons as constants, which is
what My_integration.py does (used for the equivalence tests).

Speed
  * one epoch is one tf.function: the loops over neurons run inside the graph;
  * the stiffness uses a moment-compressed rule.  The CMC points (n_MC per triangle, the same
    for every triangle) are replaced, once per epoch, by K = dim P_d nodes with weights that
    reproduce every polynomial moment of the n_MC points.  The stiffness integrands are
    polynomials of degree d = 8 (10 if alpha != 0) in each triangle, so the estimate equals the
    one with the n_MC points (up to round-off), at a cost independent of n_MC.  The load uses
    the n_MC points themselves, since f may be any function;
  * all polygons of an update are clipped and integrated as one batch.
"""

import numpy as np
import tensorflow as tf
from scipy.linalg import lu_factor, lu_solve, qr

DT = tf.float64


# ════════════════════════════════════════════════════════════════════════════
#  Batched geometry (same vertex order as My_integration._clip_np)
# ════════════════════════════════════════════════════════════════════════════
def clip_batch(V, valid, lc):
    """
    Sutherland-Hodgman clip of P polygons by P half-planes a x + b y + c >= 0.
    V     : [P, n, 2]  vertices (padding repeats the last real vertex)
    valid : [P, n]     bool, real vertices
    lc    : [P, 3]     (a, b, c)
    returns V_out [P, 2n, 2], valid_out [P, 2n].  An edge emits at most 2 points,
    so no vertex is ever dropped.
    """
    n = V.shape[1]
    Vp = tf.roll(V, shift=1, axis=1)
    a = lc[:, None, 0]; b = lc[:, None, 1]; c = lc[:, None, 2]
    dc = V[..., 0] * a + V[..., 1] * b + c
    dp = Vp[..., 0] * a + Vp[..., 1] * b + c
    den = dc - dp
    den = tf.where(tf.abs(den) < 1e-9, 1e-9 * tf.sign(den + 1e-10), den)
    t = -dp / den
    inter = Vp + t[..., None] * (V - Vp)
    inside = dc >= 0
    keep_curr = tf.logical_and(inside, valid)
    keep_inter = tf.logical_and(tf.not_equal(inside, dp >= 0), valid)
    pts = tf.reshape(tf.stack([inter, V], axis=2), [-1, 2 * n, 2])
    mask = tf.reshape(tf.stack([keep_inter, keep_curr], axis=2), [-1, 2 * n])
    cnt = tf.reduce_sum(tf.cast(mask, tf.int32), axis=1)
    rng = tf.range(2 * n)[None, :]
    key = tf.where(mask, rng, rng + 2 * n)
    order = tf.argsort(key, axis=1, stable=True)                       # kept points first, in order
    out = tf.gather(pts, order, batch_dims=1)
    last = tf.gather(out, tf.maximum(cnt - 1, 0), batch_dims=1)        # [P, 2]
    v_out = rng < cnt[:, None]
    out = tf.where(v_out[..., None], out, last[:, None, :])            # pad with the last point
    out = tf.where((cnt > 0)[:, None, None], out, tf.zeros_like(out))
    return out, v_out


def _basis(x, y):
    """phi_a = (x, y, 1) M, M = (x^2-x)(y^2-y):  gradients px, py and values m, each [..., 3]."""
    xm = x * x - x; ym = y * y - y
    B = xm * ym; Bx = (2 * x - 1) * ym; By = xm * (2 * y - 1)
    px = tf.stack([B + x * Bx, y * Bx, Bx], axis=-1)
    py = tf.stack([x * By, B + y * By, By], axis=-1)
    m = tf.stack([x * B, y * B, B], axis=-1)
    return px, py, m


# ════════════════════════════════════════════════════════════════════════════
#  Moment-compressed reference rule (exact for polynomials of degree <= deg)
# ════════════════════════════════════════════════════════════════════════════
def _collapsed_gauss(n):
    """Collapsed Gauss points on the reference triangle (candidates for the compressed rule)."""
    x, _ = np.polynomial.legendre.leggauss(n)
    r = 0.5 * x + 0.5
    R, S = np.meshgrid(r, r, indexing="ij")
    return np.stack([R.ravel(), (S * (1 - R)).ravel()], axis=1)


def _cheb_basis(pts, deg):
    """T_a(2u-1) T_b(2v-1), a + b <= deg  ->  [Q, nb]  (a basis of the polynomials of degree <= deg)."""
    u = 2 * pts[:, 0] - 1; v = 2 * pts[:, 1] - 1
    Tu = np.polynomial.chebyshev.chebvander(u, deg)
    Tv = np.polynomial.chebyshev.chebvander(v, deg)
    return np.stack([Tu[:, a] * Tv[:, b] for a in range(deg + 1) for b in range(deg + 1 - a)], axis=1)


class _MomentRule:
    """
    Nodes U (chosen once, by pivoted QR on the candidates) and the LU factors of the basis at U.
    For any points xi_q and weights w_q, the weights W solving B(U)^T W = B(xi)^T w satisfy
        sum_k W_k p(U_k) = sum_q w_q p(xi_q)    for every polynomial p of degree <= deg.
    """
    def __init__(self, deg):
        self.deg = deg
        cand = _collapsed_gauss(deg + 1)
        V = _cheb_basis(cand, deg).T                                   # [nb, candidates]
        nb = V.shape[0]
        _, _, piv = qr(V, pivoting=True, mode="economic")
        self.U = cand[np.sort(piv[:nb])]
        self.lu = lu_factor(_cheb_basis(self.U, deg).T)

    def weights(self, pts, wts):
        mu = _cheb_basis(np.asarray(pts, np.float64), self.deg).T @ np.asarray(wts, np.float64)
        return lu_solve(self.lu, mu)


def _check_polynomial(f_fun, deg):
    """Raise unless f_fun is a polynomial of total degree <= deg (least-squares fit on random points)."""
    rng = np.random.default_rng(0)
    P = rng.uniform(0.0, 1.0, size=(4 * (deg + 1) ** 2, 2))
    fx = np.asarray(f_fun(tf.constant(P[:, 0], DT), tf.constant(P[:, 1], DT)), np.float64)
    B = _cheb_basis(P, deg)
    coef, *_ = np.linalg.lstsq(B, fx, rcond=None)
    err = np.max(np.abs(B @ coef - fx)) / max(1.0, np.max(np.abs(fx)))
    if err > 1e-10:
        raise ValueError(f"f_fun is not a polynomial of degree <= {deg} (fit error {err:.1e}); "
                         "use load_degree=None")


# ════════════════════════════════════════════════════════════════════════════
#  The integrator
# ════════════════════════════════════════════════════════════════════════════
class FastIntegrator:
    """
    num_neurons    : N
    domain_vertices: Omega (convex polygon)
    alpha, f_fun   : reaction coefficient and right-hand side (TF-compatible)
    pts, wts       : reference-triangle points and weights (Integrate_Points.get_MC_quad).
                     The same points are used in every triangle, as in My_integration.py.
                     Replace them every epoch with set_points (no re-tracing).
    """

    def __init__(self, num_neurons, domain_vertices, alpha, f_fun, pts, wts, load_degree=None,
                 geometry_gradient=True):
        """geometry_gradient=True : gradients flow through the clipped polygons (boundary terms of the
                              energy, paper Corollary 4.2).  False : the polygons are constants, as in
                              My_integration.py (its gradient misses the boundary terms).
           load_degree=None : the load uses the n_MC points (any f).
           load_degree=L    : the load uses a moment-compressed rule of degree L, exact when f is a
                              polynomial of degree <= L - 5 (checked here; f_rhs of Data.py has degree 4,
                              so L = 10).  Same estimate, much cheaper."""
        self.N = int(num_neurons)
        self.alpha = float(alpha)
        self.f_fun = f_fun
        self.geometry_gradient = bool(geometry_gradient)
        dv = np.asarray(domain_vertices, np.float64)
        self.box = tf.constant(dv[None], DT)                              # [1, n0, 2]
        self.box_valid = tf.ones([1, dv.shape[0]], tf.bool)
        pI, pJ = np.triu_indices(self.N, 1)                               # pairs i < j (FB)
        self.pI = tf.constant(pI, tf.int32)
        self.pJ = tf.constant(pJ, tf.int32)
        # stiffness: compressed rule (its integrands are polynomials of degree d)
        self.rule = _MomentRule(8 if self.alpha == 0.0 else 10)
        self.sp = tf.Variable(self.rule.U, dtype=DT, trainable=False)     # [K, 2]
        self.sw = tf.Variable(np.zeros(len(self.rule.U)), dtype=DT, trainable=False)
        # load: the n_MC reference points, or a compressed rule of degree load_degree
        self.lrule = None
        if load_degree is None:
            self.kp = tf.Variable(np.zeros((len(wts), 2)), dtype=DT, trainable=False)
            self.kw = tf.Variable(np.zeros(len(wts)), dtype=DT, trainable=False)
        else:
            _check_polynomial(f_fun, int(load_degree) - 5)
            self.lrule = _MomentRule(int(load_degree))
            self.kp = tf.Variable(self.lrule.U, dtype=DT, trainable=False)
            self.kw = tf.Variable(np.zeros(len(self.lrule.U)), dtype=DT, trainable=False)
        self.set_points(pts, wts)

    def set_points(self, pts, wts):
        """New reference points (Main_2D.py draws new ones every epoch).  The stiffness and, if
        compressed, the load weights are recomputed from the moments of the new points."""
        pts = np.asarray(pts, np.float64); wts = np.asarray(wts, np.float64)
        if self.lrule is None:
            self.kp.assign(pts); self.kw.assign(wts)
        else:
            self.kw.assign(self.lrule.weights(pts, wts))
        self.sw.assign(self.rule.weights(pts, wts))

    # ---- polygon mapping (centroid fan, as My_integration._map_np) -----------
    def _fan(self, V, valid):
        """Centre cen [P,2], edge vectors e1, e2 [P,n,2] and |det| dJ [P,n] of the fan."""
        vf = tf.cast(valid, DT)
        cen = tf.reduce_sum(V * vf[..., None], axis=1) / tf.maximum(tf.reduce_sum(vf, axis=1), 1.0)[:, None]
        e1 = V - cen[:, None, :]                                          # v_t - c
        e2 = tf.roll(V, shift=-1, axis=1) - cen[:, None, :]               # v_t+1 - c
        dJ = tf.abs(e1[..., 0] * e2[..., 1] - e2[..., 0] * e1[..., 1])    # [P, n]
        return cen, e1, e2, dJ

    @staticmethod
    def _map(cen, e1, e2, q):
        """Reference points q [Q,2] mapped into every triangle: x, y [P, n, Q]."""
        qu = q[:, 0]; qv = q[:, 1]
        x = cen[:, 0, None, None] + qu * e1[..., 0, None] + qv * e2[..., 0, None]
        y = cen[:, 1, None, None] + qu * e1[..., 1, None] + qv * e2[..., 1, None]
        return x, y

    # ---- integrals over a batch of polygons ----------------------------------
    def _stiffness(self, V, valid, sp=None, sw=None):
        """A [P,3,3] = int_P grad(phi).grad(phi) + alpha phi phi, with the compressed rule
        (sp, sw: the nodes and weights as tensors, default the variables self.sp, self.sw)."""
        sp = self.sp if sp is None else sp
        sw = self.sw if sw is None else sw
        cen, e1, e2, dJ = self._fan(V, valid)
        P = tf.shape(V)[0]
        x, y = self._map(cen, e1, e2, sp)
        W = tf.reshape(dJ[..., None] * sw, [P, -1])                       # [P, n*K]
        px, py, m = [tf.reshape(g, [P, -1, 3]) for g in _basis(x, y)]     # [P, n*K, 3]

        def gram(g):                                                      # sum_k w_k g_k g_k^T
            return tf.matmul(g, g * W[..., None], transpose_a=True)

        K = gram(px) + gram(py)
        if self.alpha != 0.0:
            K = K + self.alpha * gram(m)
        return K

    def _load(self, V, valid):
        """F [P,3] = int_P f phi, with the n_MC reference points."""
        cen, e1, e2, dJ = self._fan(V, valid)
        P = tf.shape(V)[0]
        x, y = self._map(cen, e1, e2, self.kp)
        W = tf.reshape(dJ[..., None] * self.kw, [P, -1])
        m = tf.reshape(_basis(x, y)[2], [P, -1, 3])
        return tf.einsum('pk,pka->pa', W * tf.reshape(self.f_fun(x, y), [P, -1]), m)

    # ---- geometry of one update ----------------------------------------------
    def _geometry(self, w0, w1, b, c):
        """Half-plane coefficients lc (differentiable if geometry_gradient) and r_i = c_i (w0_i, w1_i, b_i)."""
        lc = tf.stack([w0, w1, b], axis=1)                                # [N, 3]
        if not self.geometry_gradient:
            lc = tf.stop_gradient(lc)
        R = tf.stack([c * w0, c * w1, c * b], axis=1)                     # [N, 3]
        return lc, R

    def _domain(self, lc_i):
        """D_i = Omega ∩ {z_i >= 0} for one neuron (lc_i [1, 3]) -> [1, 8, 2], [1, 8]."""
        return clip_batch(self.box, self.box_valid, lc_i)

    # ---- the three update types ------------------------------------------------
    def _pair_energy(self, Vd, vd, lc, R, I, J, D=None, chunk=4096):
        """Sum over the pairs p of r_{I_p}^T K(D_{I_p} ∩ {z_{J_p} >= 0}) r_{J_p}.
        The pairs are processed in chunks by a loop of the graph, so the memory does not grow with
        the number of pairs and the graph does not grow with N.
        D[p] is the row of Vd that holds the domain D_{I_p} (default D = I: Vd holds all N domains)."""
        M = tf.shape(I)[0]
        D = I if D is None else D

        # recompute_grad takes tensor arguments only: floats for the mask, the nodes and weights as tensors
        vdf = tf.cast(vd, DT)
        sp, sw = tf.convert_to_tensor(self.sp), tf.convert_to_tensor(self.sw)

        @tf.recompute_grad      # the backward pass recomputes the chunk: memory stays one chunk
        def chunk_energy(Vd_, vdf_, lc_, R_, sp_, sw_, d, i, j):
            Vp, vp = clip_batch(tf.gather(Vd_, d), tf.gather(tf.cast(vdf_, tf.bool), d), tf.gather(lc_, j))
            Kp = self._stiffness(Vp, vp, sp_, sw_)
            return tf.reduce_sum(tf.gather(R_, i) * tf.linalg.matvec(Kp, tf.gather(R_, j)))

        def body(s, acc):
            e = tf.minimum(s + chunk, M)
            return e, acc + chunk_energy(Vd, vdf, lc, R, sp, sw, D[s:e], I[s:e], J[s:e])

        _, total = tf.while_loop(lambda s, acc: s < M, body, (tf.constant(0), tf.constant(0.0, DT)))
        return total

    def fb_terms(self, w0, w1, b, c):
        """my_integration_FB:  du = sum_ij r_i^T K_ij r_j,  uf = sum_i r_i^T F_i.
        The only pair work is the N(N-1)/2 pairs i < j: cost ~ N^2 / 2."""
        N = self.N
        lc, R = self._geometry(w0, w1, b, c)
        Vd, vd = clip_batch(tf.tile(self.box, [N, 1, 1]), tf.tile(self.box_valid, [N, 1]), lc)
        Kd = self._stiffness(Vd, vd)                                       # D_i
        Fd = self._load(Vd, vd)
        pairs = self._pair_energy(Vd, vd, lc, R, self.pI, self.pJ)         # i < j
        du = tf.reduce_sum(R * tf.linalg.matvec(Kd, R)) + 2.0 * pairs
        uf = tf.reduce_sum(R * Fd)
        return du, uf

    def nw_terms(self, k, w0, w1, b, c):
        """my_integration_NW for neuron k:
           du = r_k^T K_kk r_k + 2 sum_{j>k} r_k^T K_kj r_j,  uf = r_k^T F_k,
           has = D_k is not empty (otherwise the reference returns constants: no update)."""
        N = self.N
        lc, R = self._geometry(w0, w1, b, c)
        Vk, vk = self._domain(tf.gather(lc, k)[None])                      # D_k
        Kk = self._stiffness(Vk, vk)
        Fk = self._load(Vk, vk)
        Vj, vj = clip_batch(tf.tile(Vk, [N, 1, 1]), tf.tile(vk, [N, 1]), lc)   # D_k ∩ {z_j >= 0}
        Kj = self._stiffness(Vj, vj)
        rk = tf.gather(R, k)
        kj = tf.einsum('a,jab,jb->j', rk, Kj, R)                           # r_k^T K_kj r_j
        upper = tf.cast(tf.range(N) > k, DT)                               # j > k only
        du = tf.reduce_sum(rk * tf.linalg.matvec(Kk[0], rk)) + 2.0 * tf.reduce_sum(upper * kj)
        uf = tf.reduce_sum(rk * Fk[0])
        return du, uf, tf.reduce_any(vk)

    def dnw_terms(self, k, k1, w0, w1, b, c):
        """my_integration_DNW for the pair (k, k1):
           du = r_k^T A r_k1  with A = K_kk if k1 == k else K_{k,k1},
           uf = r_k^T F_k / N  (the load of neuron k is spread over its N steps),
           has = D_k is not empty (see nw_terms)."""
        lc, R = self._geometry(w0, w1, b, c)
        Vk, vk = self._domain(tf.gather(lc, k)[None])                      # D_k
        Kk = self._stiffness(Vk, vk)
        Fk = self._load(Vk, vk)
        Vp, vp = clip_batch(Vk, vk, tf.gather(lc, k1)[None])               # D_k ∩ {z_k1 >= 0}
        Kp = self._stiffness(Vp, vp)
        A = tf.where(tf.equal(k, k1), Kk[0], Kp[0])
        rk = tf.gather(R, k)
        du = tf.reduce_sum(rk * tf.linalg.matvec(A, tf.gather(R, k1)))
        uf = tf.reduce_sum(rk * Fk[0]) / tf.cast(self.N, DT)
        return du, uf, tf.reduce_any(vk)

    # ---- random batches: all selected terms in one graph, no Python loop over pairs ----
    def batched_nw_terms(self, K1, n1, w0, w1, b, c):
        """Sum over the n1 neurons k in K1 of the NW terms du_k/2 - uf_k, with
        du_k/2 = 1/2 r_k^T K_kk r_k + sum_{j>k} r_k^T K_kj r_j.  Only the domains D_k, k in K1, and the
        pairs j > k are constructed: n1 domains and about n1 * (N-1)/2 pairs.
        Returns (sum, any D_k non-empty)."""
        N = self.N
        lc, R = self._geometry(w0, w1, b, c)
        Vk, vk = clip_batch(tf.tile(self.box, [n1, 1, 1]), tf.tile(self.box_valid, [n1, 1]), tf.gather(lc, K1))
        Kk = self._stiffness(Vk, vk)                                       # [n1, 3, 3]
        Fk = self._load(Vk, vk)                                            # [n1, 3]
        rk = tf.gather(R, K1)                                              # [n1, 3]
        kk = tf.repeat(K1, N)                                              # pairs (k, j): k in K1, all j
        jj = tf.tile(tf.range(N), [n1])
        kl = tf.repeat(tf.range(n1), N)                                    # row of D_k in Vk
        keep = jj > kk
        pairs = self._pair_energy(Vk, vk, lc, R, tf.boolean_mask(kk, keep), tf.boolean_mask(jj, keep),
                                  D=tf.boolean_mask(kl, keep))
        diag = 0.5 * tf.reduce_sum(tf.einsum('ka,kab,kb->k', rk, Kk, rk))
        load = tf.reduce_sum(rk * Fk)
        return diag + pairs - load, tf.reduce_any(vk)

    def batched_dnw_terms(self, K1, K2, n1, n2, w0, w1, b, c):
        """Stochastic DNW energy of one epoch with fixed sizes n1 = r and n2 = rbar:
           J~_i = (n/rbar) sum_{j in K2} 1/2 r_i^T K_ij r_j - r_i^T F_i,   J~ = sum_{i in K1} J~_i,
        whose expectation over the random sets is (r/n) grad J.  Only the domains D_k, k in K1, and the pairs
        k != k1 are constructed; the diagonal terms (k = k1 in K1 and K2) use K_kk.
        Returns (sum, any D_k non-empty)."""
        N = self.N
        lc, R = self._geometry(w0, w1, b, c)
        Vk, vk = clip_batch(tf.tile(self.box, [n1, 1, 1]), tf.tile(self.box_valid, [n1, 1]), tf.gather(lc, K1))
        Kk = self._stiffness(Vk, vk)                                       # [n1, 3, 3]
        Fk = self._load(Vk, vk)                                            # [n1, 3]
        rk = tf.gather(R, K1)                                              # [n1, 3]
        kk = tf.repeat(K1, n2)                                             # pairs (k, k1), k in K1, k1 in K2
        k1 = tf.tile(K2, [n1])
        kl = tf.repeat(tf.range(n1), n2)                                   # row of D_k in Vk
        keep = tf.not_equal(kk, k1)
        off = self._pair_energy(Vk, vk, lc, R, tf.boolean_mask(kk, keep), tf.boolean_mask(k1, keep),
                                D=tf.boolean_mask(kl, keep))
        in_k2 = tf.cast(tf.reduce_any(tf.equal(K1[:, None], K2[None, :]), axis=1), DT)   # k also in K2
        diag = tf.reduce_sum(in_k2 * tf.einsum('ka,kab,kb->k', rk, Kk, rk))
        scale = tf.cast(N, DT) / tf.cast(n2, DT)                           # n / rbar
        load = tf.reduce_sum(rk * Fk)                                      # r_k^T F_k, once per k in K1
        return scale * (0.5 * off + 0.5 * diag) - load, tf.reduce_any(vk)


# ════════════════════════════════════════════════════════════════════════════
#  Training updates of train_u_1D.py, one compiled epoch
# ════════════════════════════════════════════════════════════════════════════
def _neuron_penalty(w0, w1, b):
    """Per-neuron interior penalty R_j of train_u_1D.interior_regularization (one value per neuron)."""
    norm_w = tf.sqrt(w0 * w0 + w1 * w1 + 1e-12)
    corners = tf.stack([b, w0 + b, w1 + b, w0 + w1 + b], axis=0) / norm_w
    margin = 0.05
    return tf.square(tf.nn.relu(-tf.reduce_max(corners, axis=0) + margin)) + \
           tf.square(tf.nn.relu(tf.reduce_min(corners, axis=0) + margin))


def interior_regularization(w0, w1, b, lambda_reg):
    """Same as train_u_1D.interior_regularization: lambda/N sum over all neurons of R_j."""
    if lambda_reg == 0.0:
        return tf.constant(0.0, DT)
    return tf.cast(lambda_reg, DT) * tf.reduce_mean(_neuron_penalty(w0, w1, b))


def interior_regularization_selected(w0, w1, b, idx, N, lambda_reg):
    """Stochastic version: lambda/N sum over the selected neurons idx only. Its expectation over the random
    selection is (r/n) times the full regulariser, the same weight as the (r/n) energy."""
    if lambda_reg == 0.0:
        return tf.constant(0.0, DT)
    pen = _neuron_penalty(tf.gather(w0, idx), tf.gather(w1, idx), tf.gather(b, idx))
    return tf.cast(lambda_reg, DT) / tf.cast(N, DT) * tf.reduce_sum(pen)


def _weights(u_net):
    W1 = tf.cast(u_net.layers[0].weights[0], DT)
    b1 = tf.cast(u_net.layers[0].weights[1], DT)
    c = tf.cast(u_net.layers[-1].weights[0][:, 0], DT)
    return W1[0], W1[1], b1, c


def make_train_steps(u_net, optimizer, integrator, variant, lambda_reg=0.1,
                     n_batched_1=None, n_batched_2=None):
    """
    Returns train_epoch() -> loss, compiled as one graph.

    Without n_batched_1 it performs the updates of train_u_1D.py in the same order:
      FB : one update                                  (train_step_FB)
      NW : k = 0..N-1, one update per neuron           (train_step_NW)
      DNW: k, k1 = 0..N-1, one update per pair         (train_step_DNW)
    The returned loss is the sum of the step losses, as in train_u_1D.py.

    With n_batched_1 (and, for DNW, n_batched_2): every epoch draws a random set K1 of n_batched_1
    neurons (and K2 of n_batched_2 neurons for DNW), the loss is the sum over the selected terms
      NW : sum_{k in K1} J_k
      DNW: sum_{k in K1, k1 in K2} J_{k,k1}
    plus the regularisation once, and there is ONE gradient and ONE update per epoch.
    """
    assert variant in ("FB", "NW", "DNW")
    lam = float(lambda_reg)
    N = integrator.N
    trainable = u_net.trainable_variables

    def _apply(g, has):
        """Apply the Adam update only if has.  When the neuron's domain is empty, My_integration
        returns constants, the gradient of the output weights is None, and train_u_1D skips the
        update (the loss, regularisation only, is still counted)."""
        def update():
            optimizer.apply_gradients(zip(g, trainable))
            return tf.constant(0)
        tf.cond(has, update, lambda: tf.constant(0))

    def step_FB():
        with tf.GradientTape() as tape:
            w0, w1, b, c = _weights(u_net)
            du, uf = integrator.fb_terms(w0, w1, b, c)
            loss = du / 2.0 - uf + interior_regularization(w0, w1, b, lam)
        optimizer.apply_gradients(zip(tape.gradient(loss, trainable), trainable))
        return loss

    def step_NW(k):
        with tf.GradientTape() as tape:
            w0, w1, b, c = _weights(u_net)
            du, uf, has = integrator.nw_terms(k, w0, w1, b, c)
            loss = du / 2.0 - uf + interior_regularization(w0, w1, b, lam)
        _apply(tape.gradient(loss, trainable), has)
        return loss

    def step_DNW(k, k1):
        with tf.GradientTape() as tape:
            w0, w1, b, c = _weights(u_net)
            du, uf, has = integrator.dnw_terms(k, k1, w0, w1, b, c)
            loss = du / 2.0 - uf + interior_regularization(w0, w1, b, lam)
        _apply(tape.gradient(loss, trainable), has)
        return loss

    if variant == "FB":
        return tf.function(step_FB)

    if n_batched_1 is not None:
        # random batches, ONE gradient and ONE update per epoch on the summed loss (vectorised)
        n1 = min(int(n_batched_1), N)
        n2 = min(int(n_batched_2), N) if variant == "DNW" else 0

        @tf.function
        def train_epoch():
            K1 = tf.random.shuffle(tf.range(N))[:n1]
            K2 = tf.random.shuffle(tf.range(N))[:n2]
            with tf.GradientTape() as tape:
                w0, w1, b, c = _weights(u_net)
                if variant == "NW":
                    part, has_any = integrator.batched_nw_terms(K1, n1, w0, w1, b, c)
                else:
                    part, has_any = integrator.batched_dnw_terms(K1, K2, n1, n2, w0, w1, b, c)
                total = part + interior_regularization_selected(w0, w1, b, K1, N, lam)
            _apply(tape.gradient(total, trainable), has_any)
            return total
        return train_epoch

    if variant == "NW":
        @tf.function
        def train_epoch():
            total = tf.constant(0.0, DT)
            for k in tf.range(N):
                total += step_NW(k)
            return total
        return train_epoch

    @tf.function
    def train_epoch():
        total = tf.constant(0.0, DT)
        for k in tf.range(N):
            for k1 in tf.range(N):
                total += step_DNW(k, k1)
        return total
    return train_epoch
