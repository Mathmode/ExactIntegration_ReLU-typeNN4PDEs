import tensorflow as tf
import numpy as np
from Data import f_rhs


# ─── Compiled clipping kernel ─────────────────────────────────────────────────
@tf.function(reduce_retracing=True)
def clip_polygon_with_line_diff(vertices, line_coeffs):
    vertices    = tf.cast(vertices,    dtype=tf.float64)
    line_coeffs = tf.cast(line_coeffs, dtype=tf.float64)
    v_curr = vertices
    v_prev = tf.roll(vertices, shift=1, axis=0)
    d_prev = tf.reduce_sum(v_prev * line_coeffs[:2], axis=1) + line_coeffs[2]
    d_curr = tf.reduce_sum(v_curr * line_coeffs[:2], axis=1) + line_coeffs[2]
    denominator = d_curr - d_prev
    denominator = tf.where(tf.abs(denominator) < 1e-9,
                           1e-9 * tf.sign(denominator + 1e-10), denominator)
    t = -d_prev / denominator
    intersections = v_prev + tf.expand_dims(t, axis=-1) * (v_curr - v_prev)
    curr_inside = d_curr >= 0
    keep_inter  = tf.not_equal(curr_inside, d_prev >= 0)
    keep_curr   = curr_inside
    all_pts   = tf.concat([tf.expand_dims(intersections, 1),
                            tf.expand_dims(v_curr, 1)], axis=1)
    all_masks = tf.stack([keep_inter, keep_curr], axis=1)
    return tf.boolean_mask(tf.reshape(all_pts, [-1, 2]),
                           tf.reshape(all_masks, [-1]))


# ─── Shared geometry: map quadrature points onto polygon triangles ─────────────
@tf.function(reduce_retracing=True)
def _map_quad_pts(v, q_pts, q_wts):
    v     = tf.cast(v,     tf.float64)
    q_pts = tf.cast(q_pts, tf.float64)
    q_wts = tf.cast(q_wts, tf.float64)
    center = tf.reduce_mean(v, axis=0)
    v1 = v; v2 = tf.roll(v, shift=-1, axis=0)
    det_J = tf.abs(
        (v1[:,0]-center[0])*(v2[:,1]-center[1]) -
        (v2[:,0]-center[0])*(v1[:,1]-center[1])
    )
    c_  = center[tf.newaxis, tf.newaxis, :]
    v1_ = v1[:, tf.newaxis, :]; v2_ = v2[:, tf.newaxis, :]
    qp_ = q_pts[tf.newaxis, :, :]
    xt = c_[:,:,0] + qp_[:,:,0]*(v1_[:,:,0]-c_[:,:,0]) + qp_[:,:,1]*(v2_[:,:,0]-c_[:,:,0])
    yt = c_[:,:,1] + qp_[:,:,0]*(v1_[:,:,1]-c_[:,:,1]) + qp_[:,:,1]*(v2_[:,:,1]-c_[:,:,1])
    return xt, yt, det_J, q_wts


@tf.function(reduce_retracing=True)
def _eval_M(xt, yt):
    xt2_m_xt = xt * xt - xt
    yt2_m_yt = yt * yt - yt
    M    = xt2_m_xt * yt2_m_yt
    dx_M = (2.0 * xt - 1.0) * yt2_m_yt
    dy_M = xt2_m_xt * (2.0 * yt - 1.0)
    return M, dx_M, dy_M


@tf.function(reduce_retracing=True)
def _eval_ui_preM(M, dx_M, dy_M, xt, yt, wi0, wi1, bi, ci):
    wi0 = tf.cast(wi0, tf.float64); wi1 = tf.cast(wi1, tf.float64)
    bi  = tf.cast(bi,  tf.float64); ci  = tf.cast(ci,  tf.float64)
    Ni    = wi0 * xt + wi1 * yt + bi
    ui    = ci * Ni * M
    dx_ui = ci * (wi0 * M + Ni * dx_M)
    dy_ui = ci * (wi1 * M + Ni * dy_M)
    return ui, dx_ui, dy_ui


# ─── Compiled integration kernels ────────────────────────────────────────────
@tf.function(reduce_retracing=True)
def compute_MC_integrals_K_ij(v, q_pts, q_wts, wi0_i, wi1_i, bi_i, ci_i,
                               wi0_j, wi1_j, bi_j, ci_j, alpha):
    xt, yt, det_J, q_wts = _map_quad_pts(v, q_pts, q_wts)
    M, dx_M, dy_M = _eval_M(xt, yt)
    ui, dx_ui, dy_ui = _eval_ui_preM(M, dx_M, dy_M, xt, yt, wi0_i, wi1_i, bi_i, ci_i)
    uj, dx_uj, dy_uj = _eval_ui_preM(M, dx_M, dy_M, xt, yt, wi0_j, wi1_j, bi_j, ci_j)
    alpha_cast = tf.cast(alpha, tf.float64)
    integrand  = dx_ui * dx_uj + dy_ui * dy_uj + alpha_cast * ui * uj
    tri_ints   = tf.reduce_sum(integrand * q_wts[tf.newaxis, :], axis=1)
    return tf.reduce_sum(tri_ints * det_J)


@tf.function(reduce_retracing=True)
def compute_MC_integrals_F_i(v, q_pts, q_wts, wi0_i, wi1_i, bi_i, ci_i):
    xt, yt, det_J, q_wts = _map_quad_pts(v, q_pts, q_wts)
    M, dx_M, dy_M = _eval_M(xt, yt)
    ui, _, _ = _eval_ui_preM(M, dx_M, dy_M, xt, yt, wi0_i, wi1_i, bi_i, ci_i)
    f_val     = tf.cast(f_rhs(xt, yt), tf.float64)
    integrand = f_val * ui
    tri_ints  = tf.reduce_sum(integrand * q_wts[tf.newaxis, :], axis=1)
    return tf.reduce_sum(tri_ints * det_J)


@tf.function(reduce_retracing=True)
def compute_MC_integrals_K_ii_F_i(v, q_pts, q_wts, wi0_i, wi1_i, bi_i, ci_i, alpha):
    """Compute K_ii and F_i together — shares polygon mapping for diagonal terms.
    Saves one full _map_quad_pts + _eval_M + _eval_ui_preM call per neuron."""
    xt, yt, det_J, q_wts = _map_quad_pts(v, q_pts, q_wts)
    M, dx_M, dy_M = _eval_M(xt, yt)
    ui, dx_ui, dy_ui = _eval_ui_preM(M, dx_M, dy_M, xt, yt, wi0_i, wi1_i, bi_i, ci_i)
    # K_ii = integral of |∇u_i|² + α·u_i²
    alpha_cast = tf.cast(alpha, tf.float64)
    integrand_k = dx_ui * dx_ui + dy_ui * dy_ui + alpha_cast * ui * ui
    tri_ints_k  = tf.reduce_sum(integrand_k * q_wts[tf.newaxis, :], axis=1)
    K_ii = tf.reduce_sum(tri_ints_k * det_J)
    # F_i = integral of f · u_i
    f_val       = tf.cast(f_rhs(xt, yt), tf.float64)
    tri_ints_f  = tf.reduce_sum(f_val * ui * q_wts[tf.newaxis, :], axis=1)
    F_i = tf.reduce_sum(tri_ints_f * det_J)
    return K_ii, F_i


# ─── NumPy fast kernels (used in FB to avoid TF per-call overhead) ───────────

def _clip_np(v, lc):
    """Sutherland–Hodgman half-plane clip — pure NumPy."""
    vc = v; vp = np.roll(v, 1, 0)
    dc = vc @ lc[:2] + lc[2]; dp = vp @ lc[:2] + lc[2]
    den = dc - dp
    den = np.where(np.abs(den) < 1e-9, 1e-9 * np.sign(den + 1e-10), den)
    t   = (-dp / den)[:, None]
    inter = vp + t * (vc - vp)
    ki = (dc >= 0) != (dp >= 0); kc = dc >= 0
    pts  = np.stack([inter, vc], axis=1).reshape(-1, 2)
    mask = np.stack([ki, kc],   axis=1).reshape(-1)
    return pts[mask]


def _map_np(v, q_pts, q_wts):
    """Fan-triangulate polygon, map quadrature points — NumPy.
    Returns xt [ntri,nq], yt [ntri,nq], det_J [ntri]."""
    c  = v.mean(0)
    v1 = v; v2 = np.roll(v, -1, 0)
    dJ = np.abs((v1[:,0]-c[0])*(v2[:,1]-c[1]) - (v2[:,0]-c[0])*(v1[:,1]-c[1]))
    xt = c[0] + q_pts[None,:,0]*(v1[:,None,0]-c[0]) + q_pts[None,:,1]*(v2[:,None,0]-c[0])
    yt = c[1] + q_pts[None,:,0]*(v1[:,None,1]-c[1]) + q_pts[None,:,1]*(v2[:,None,1]-c[1])
    return xt, yt, dJ


def _K_ij_np(v, q_pts, q_wts, w0i, w1i, bii, cii, w0j, w1j, bij, cij, alpha):
    """K_ij scalar — pure NumPy."""
    xt, yt, dJ = _map_np(v, q_pts, q_wts)
    xt2m = xt*xt - xt; yt2m = yt*yt - yt
    M   = xt2m * yt2m
    dxM = (2*xt - 1) * yt2m
    dyM = xt2m * (2*yt - 1)
    Ni   = w0i*xt + w1i*yt + bii
    dx_ui = cii * (w0i*M + Ni*dxM);  dy_ui = cii * (w1i*M + Ni*dyM)
    ui    = cii * Ni * M
    Nj   = w0j*xt + w1j*yt + bij
    dx_uj = cij * (w0j*M + Nj*dxM);  dy_uj = cij * (w1j*M + Nj*dyM)
    uj    = cij * Nj * M
    intgd = dx_ui*dx_uj + dy_ui*dy_uj + alpha*ui*uj
    return float(np.einsum('tq,q,t->', intgd, q_wts, dJ))


def _K_ii_F_i_np(v, q_pts, q_wts, w0, w1, bi, ci, alpha):
    """K_ii and F_i in one polygon mapping — pure NumPy.
    f_rhs is called once with the mapped points (TF-compatible call)."""
    xt, yt, dJ = _map_np(v, q_pts, q_wts)
    xt2m = xt*xt - xt; yt2m = yt*yt - yt
    M   = xt2m * yt2m
    dxM = (2*xt - 1) * yt2m
    dyM = xt2m * (2*yt - 1)
    Ni   = w0*xt + w1*yt + bi
    ui   = ci * Ni * M
    dx_ui = ci * (w0*M + Ni*dxM)
    dy_ui = ci * (w1*M + Ni*dyM)
    intgd_k = dx_ui**2 + dy_ui**2 + alpha*ui**2
    K_ii = float(np.einsum('tq,q,t->', intgd_k, q_wts, dJ))
    # f_rhs may use TF ops — call once and convert
    f_val = np.asarray(f_rhs(tf.constant(xt, tf.float64),
                              tf.constant(yt, tf.float64)), dtype=np.float64)
    F_i  = float(np.einsum('tq,tq,q,t->', f_val, ui, q_wts, dJ))
    return K_ii, F_i


# ── Matrix-returning variants (for gradient-compatible training) ──────────────
# K_ij = r_i^T A_ij r_j  where A_ij is a pure geometric 3×3 matrix.
# u_i = c_i*(w0_i*x + w1_i*y + b_i)*M  so r_i = [c_i*w0_i, c_i*w1_i, c_i*b_i]
# and  ∇u_i = r_i · p_x / p_y,  u_i = r_i · m  (basis vectors, geometry only).

def _A_ij_np(v, q_pts, q_wts, alpha):
    """Return 3×3 bilinear-form matrix A_ij (no weights applied)."""
    xt, yt, dJ = _map_np(v, q_pts, q_wts)
    xt2m = xt*xt - xt; yt2m = yt*yt - yt
    M   = xt2m * yt2m
    dxM = (2*xt - 1) * yt2m
    dyM = xt2m * (2*yt - 1)
    # basis column vectors evaluated at each quadrature point
    # p_x[:,a] = d/dx [phi_a * M],  phi = (x, y, 1)
    p_x = np.stack([M + xt*dxM, yt*dxM,       dxM], axis=-1)  # [T,Q,3]
    p_y = np.stack([xt*dyM,     M + yt*dyM,    dyM], axis=-1)  # [T,Q,3]
    m   = np.stack([xt*M,       yt*M,          M  ], axis=-1)  # [T,Q,3]
    intgd = (np.einsum('...a,...b->...ab', p_x, p_x)
           + np.einsum('...a,...b->...ab', p_y, p_y)
           + alpha * np.einsum('...a,...b->...ab', m, m))       # [T,Q,3,3]
    return np.einsum('tqab,q,t->ab', intgd, q_wts, dJ)         # [3,3]


def _A_ii_F_i_np(v, q_pts, q_wts, alpha):
    """Return (A_ii [3,3], f_i [3]) for diagonal cell — one polygon mapping."""
    xt, yt, dJ = _map_np(v, q_pts, q_wts)
    xt2m = xt*xt - xt; yt2m = yt*yt - yt
    M   = xt2m * yt2m
    dxM = (2*xt - 1) * yt2m
    dyM = xt2m * (2*yt - 1)
    p_x = np.stack([M + xt*dxM, yt*dxM,    dxM], axis=-1)
    p_y = np.stack([xt*dyM,     M+yt*dyM,  dyM], axis=-1)
    m   = np.stack([xt*M,       yt*M,      M  ], axis=-1)
    intgd = (np.einsum('...a,...b->...ab', p_x, p_x)
           + np.einsum('...a,...b->...ab', p_y, p_y)
           + alpha * np.einsum('...a,...b->...ab', m, m))
    A_ii = np.einsum('tqab,q,t->ab', intgd, q_wts, dJ)
    f_val = np.asarray(f_rhs(tf.constant(xt, tf.float64),
                              tf.constant(yt, tf.float64)), dtype=np.float64)
    # F_i[a] = ∫ f(x,y) * phi_a(x,y) * M dx  — 3-vector
    f_i = np.einsum('tq,tqa,q,t->a', f_val, m, q_wts, dJ)
    return A_ii, f_i


# ─── Outer loops: eager Python (no @tf.function) ──────────────────────────────

def my_integration_FB(u_net, num_neurons, domain_vertices, pts, wts, alpha):
    weights_1 = u_net.layers[0].weights[0]
    biases_1  = u_net.layers[0].weights[1]
    weights_2 = u_net.layers[-1].weights[0]
    wi0, wi1, bi, ci = weights_1[0,:], weights_1[1,:], biases_1[:], weights_2[:,0]

    # ── NumPy phase: build full 3N×3N stiffness matrix and 3N load vector ──────
    wi0_np = wi0.numpy(); wi1_np = wi1.numpy()
    bi_np  = bi.numpy();  ci_np  = ci.numpy()
    alpha_np  = float(alpha)
    q_pts_np  = np.asarray(pts,  dtype=np.float64)
    q_wts_np  = np.asarray(wts,  dtype=np.float64)
    dv_np     = np.asarray(domain_vertices, dtype=np.float64)
    all_lc_np = np.stack([wi0_np, wi1_np, bi_np], axis=1)  # [N, 3]
    N3        = 3 * num_neurons

    M_np = np.zeros((N3, N3))  # stiffness [3N, 3N]
    f_np = np.zeros(N3)         # load      [3N]

    # Pre-clip all domain_i polygons
    domains_i = []
    for i in range(num_neurons):
        di = _clip_np(dv_np, all_lc_np[i])
        domains_i.append(di if len(di) > 0 else None)

    for i in range(num_neurons):
        di = domains_i[i]
        if di is None:
            continue
        si = slice(3*i, 3*i+3)

        # Diagonal: A_ii and f_i share one polygon mapping
        A_ii, f_i = _A_ii_F_i_np(di, q_pts_np, q_wts_np, alpha_np)
        M_np[si, si] += A_ii
        f_np[si]     += f_i

        # Off-diagonal (j > i): exploit symmetry A_ij = A_ji
        for j in range(i + 1, num_neurons):
            dij = _clip_np(di, all_lc_np[j])
            if len(dij) == 0:
                continue
            sj = slice(3*j, 3*j+3)
            A_ij = _A_ij_np(dij, q_pts_np, q_wts_np, alpha_np)
            M_np[si, sj] += A_ij
            M_np[sj, si] += A_ij   # symmetry

    # ── TF phase: ONE matmul — gradients flow through r ──────────────────────
    M_tf = tf.constant(M_np, dtype=tf.float64)  # [3N, 3N] constant
    f_tf = tf.constant(f_np, dtype=tf.float64)  # [3N]  constant

    wi0_f = tf.cast(wi0, tf.float64); wi1_f = tf.cast(wi1, tf.float64)
    bi_f  = tf.cast(bi,  tf.float64); ci_f  = tf.cast(ci,  tf.float64)
    # r_flat[i*3+k] = [c_i*w0_i, c_i*w1_i, c_i*b_i]  for all i
    R_tf   = tf.stack([ci_f*wi0_f, ci_f*wi1_f, ci_f*bi_f], axis=0)  # [3, N]
    r_flat = tf.reshape(tf.transpose(R_tf), [N3])                     # [3N]

    Integral_du_du = tf.reduce_sum(r_flat * tf.linalg.matvec(M_tf, r_flat))
    Integral_u_f   = tf.reduce_sum(r_flat * f_tf)
    return Integral_du_du, Integral_u_f


def my_integration_NW(u_net, num_neurons, domain_vertices, sampled_i_list,
                       pts, wts, alpha):
    """Neuron-wise integration — symmetric upper-triangle, pre-allocated buffers.

    Key ideas
    ---------
    1. **Symmetry exploit** (like FB): for each sampled i, only iterate j > i.
       Off-diagonal entry stored as 2·A_ij.  This is exact because A_ij is a
       symmetric matrix, so r_i^T A_ij r_j = r_j^T A_ij r_i (scalar identity),
       giving r_i^T A_ij r_j + r_j^T A_ij r_i = 2·r_i^T A_ij r_j.
       The TF einsum propagates gradients to BOTH r_i and r_j via the two
       tf.gather calls — no information lost.

    2. **Pre-allocated contiguous buffer** (no Python list / no np.stack):
       A_buf[p] = A_ij is a cache-friendly C-level memcpy, same pattern as
       FB's M_np[si,sj] = A_ij — eliminates scattered heap allocations.
       A_buf[:p] is a zero-copy numpy view → tf.constant uploads one block.

    Clip count vs FB
    ----------------
      FB : N (domain pre-clips) + N*(N-1)/2 inner clips  ≈  N²/2
      NW : n_i domain clips    + n_i*(N-1-avg_i)/2       ≈  n_i*N/4
                                                           ≈  N²/8  (n_i=N/2)
    TF tensor size vs FB
    --------------------
      FB : [3N, 3N]          = 9N²   floats
      NW : [n_pairs, 3, 3]  ≈ n_i*N/4 * 9  ≈  9N²/8 floats  (n_i=N/2)
    → both clip work and TF tensor are ~4× smaller than FB for average n_i.

    Returns (Integral_du_du, Integral_u_f)."""
    weights_1 = u_net.layers[0].weights[0]
    biases_1  = u_net.layers[0].weights[1]
    weights_2 = u_net.layers[-1].weights[0]
    wi0, wi1, bi, ci = weights_1[0,:], weights_1[1,:], biases_1[:], weights_2[:,0]

    # ── Precompute ONCE per training step ──────────────────────────────────────
    wi0_np = wi0.numpy(); wi1_np = wi1.numpy()
    bi_np  = bi.numpy();  ci_np  = ci.numpy()
    alpha_np  = float(alpha)
    q_pts_np  = np.asarray(pts,  dtype=np.float64)
    q_wts_np  = np.asarray(wts,  dtype=np.float64)
    dv_np     = np.asarray(domain_vertices, dtype=np.float64)
    all_lc_np = np.stack([wi0_np, wi1_np, bi_np], axis=1)  # [N, 3]

    wi0_f = tf.cast(wi0, tf.float64); wi1_f = tf.cast(wi1, tf.float64)
    bi_f  = tf.cast(bi,  tf.float64); ci_f  = tf.cast(ci,  tf.float64)
    R_T   = tf.transpose(
        tf.stack([ci_f*wi0_f, ci_f*wi1_f, ci_f*bi_f], axis=0))   # [N, 3]

    # ── Geometry phase: upper-triangle only, pre-allocated contiguous buffers ─
    # Worst-case pairs per sampled i:  1 (diagonal) + (N-1-i) (upper triangle).
    # When i=0 is sampled alone, that's N pairs total — must bound by N per row.
    # Safe upper bound: n_i_est * num_neurons.
    n_i_est = len(sampled_i_list)
    n_max   = n_i_est * num_neurons
    A_buf  = np.empty((n_max, 3, 3), dtype=np.float64)
    i_buf  = np.empty(n_max,         dtype=np.int32)
    j_buf  = np.empty(n_max,         dtype=np.int32)
    f_buf  = np.empty((n_i_est, 3),  dtype=np.float64)
    vi_buf = np.empty(n_i_est,       dtype=np.int32)
    p  = 0   # running pair count
    nv = 0   # running valid-i count

    for i in sampled_i_list:
        i = int(i)
        domain_i = _clip_np(dv_np, all_lc_np[i])
        if len(domain_i) == 0:
            continue

        A_ii, f_i = _A_ii_F_i_np(domain_i, q_pts_np, q_wts_np, alpha_np)
        A_buf[p] = A_ii;  i_buf[p] = i;  j_buf[p] = i;  p += 1   # diagonal (×1)

        for j in range(i + 1, num_neurons):                        # UPPER TRIANGLE only
            dij = _clip_np(domain_i, all_lc_np[j])
            if len(dij) > 0:
                A_buf[p] = 2.0 * _A_ij_np(dij, q_pts_np, q_wts_np, alpha_np)
                i_buf[p] = i;  j_buf[p] = j;  p += 1              # off-diag (×2)

        f_buf[nv] = f_i;  vi_buf[nv] = i;  nv += 1

    if nv == 0:
        return tf.constant(0.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64)

    # ── TF phase: 4 ops, compact contiguous tensors ───────────────────────────
    A_tf      = tf.constant(A_buf[:p],    dtype=tf.float64)   # [n_pairs, 3, 3]
    r_i_tf    = tf.gather(R_T, i_buf[:p])                      # [n_pairs, 3]
    r_j_tf    = tf.gather(R_T, j_buf[:p])                      # [n_pairs, 3]
    total_du_du = tf.einsum('pa,pab,pb->', r_i_tf, A_tf, r_j_tf)

    f_tf      = tf.constant(f_buf[:nv],   dtype=tf.float64)   # [n_v, 3]
    r_sel_tf  = tf.gather(R_T, vi_buf[:nv])                    # [n_v, 3]
    total_u_f = tf.reduce_sum(r_sel_tf * f_tf)

    return total_du_du, total_u_f


def my_integration_DNW(u_net, num_neurons, domain_vertices, sampled_i_list,
                        pts, wts, alpha):
    """Double neuron-wise — for each sampled i, sample j from ALL {0,...,N-1}, scale by N/n_j.

    This directly mirrors the old (proven-correct) DNW algorithm using the new
    3×3 matrix formulation:
      - Sample n_j uniformly from {1,...,N}
      - For j == i: use A_ii (already computed); scaled by N/n_j
      - For j != i: compute domain_ij = domain_i ∩ {z_j>0}, then A_ij; scaled by N/n_j
      - NO 2× factor (no upper-triangle symmetry exploit — j ranges over all neurons)
      - F_i is exact (not scaled by N/n_j)

    The expected value of (N/n_j) * sum_{j in sampled_j} K_ij equals sum_j K_ij,
    giving an unbiased stochastic gradient estimate.

    Clip count vs NW
    ----------------
      NW  : n_i × (N-1-avg_i)    ≈  n_i × N/2   (ALL j ≠ i)
      DNW : n_i × n_j_samp       ≈  n_i × N/2   (random subset of ALL N neurons)
    → DNW has ~same clip count as NW on average, but random j gives different gradient noise.

    Returns (total_du_du, total_u_f, n_pairs)."""
    weights_1 = u_net.layers[0].weights[0]
    biases_1  = u_net.layers[0].weights[1]
    weights_2 = u_net.layers[-1].weights[0]
    wi0, wi1, bi, ci = weights_1[0,:], weights_1[1,:], biases_1[:], weights_2[:,0]

    wi0_np = wi0.numpy(); wi1_np = wi1.numpy()
    bi_np  = bi.numpy()
    alpha_np  = float(alpha)
    q_pts_np  = np.asarray(pts,  dtype=np.float64)
    q_wts_np  = np.asarray(wts,  dtype=np.float64)
    dv_np     = np.asarray(domain_vertices, dtype=np.float64)
    all_lc_np = np.stack([wi0_np, wi1_np, bi_np], axis=1)  # [N, 3]

    wi0_f = tf.cast(wi0, tf.float64); wi1_f = tf.cast(wi1, tf.float64)
    bi_f  = tf.cast(bi,  tf.float64); ci_f  = tf.cast(ci,  tf.float64)
    R_T   = tf.transpose(
        tf.stack([ci_f*wi0_f, ci_f*wi1_f, ci_f*bi_f], axis=0))   # [N, 3]

    # ── Geometry phase: pre-allocated contiguous buffers (same as NW) ─────────
    n_i_est = len(sampled_i_list)
    n_max   = n_i_est * num_neurons          # safe worst-case (i=0 alone → N entries)
    A_buf  = np.empty((n_max, 3, 3), dtype=np.float64)
    i_buf  = np.empty(n_max,         dtype=np.int32)
    j_buf  = np.empty(n_max,         dtype=np.int32)
    f_buf  = np.empty((n_i_est, 3),  dtype=np.float64)
    vi_buf = np.empty(n_i_est,       dtype=np.int32)
    p  = 0
    nv = 0

    for i in sampled_i_list:
        i = int(i)
        domain_i = _clip_np(dv_np, all_lc_np[i])
        if len(domain_i) == 0:
            continue

        A_ii, f_i = _A_ii_F_i_np(domain_i, q_pts_np, q_wts_np, alpha_np)

        # Sample n_j from ALL {0,...,N-1} — unbiased estimator of sum_j K_ij
        all_j = np.arange(num_neurons)
        n_j_samp = np.random.randint(1, num_neurons + 1)
        np.random.shuffle(all_j)
        sampled_j = all_j[:n_j_samp]
        scale = float(num_neurons) / float(n_j_samp)   # unbiasing factor N/n_j

        for j in sampled_j:
            j = int(j)
            if j == i:
                A_buf[p] = scale * A_ii              # diagonal, scaled (no ×2)
                i_buf[p] = i;  j_buf[p] = i;  p += 1
            else:
                dij = _clip_np(domain_i, all_lc_np[j])
                if len(dij) > 0:
                    A_buf[p] = scale * _A_ij_np(dij, q_pts_np, q_wts_np, alpha_np)
                    i_buf[p] = i;  j_buf[p] = j;  p += 1   # off-diag, scaled (no ×2)

        f_buf[nv] = f_i;  vi_buf[nv] = i;  nv += 1

    if nv == 0:
        return tf.constant(0.0, dtype=tf.float64), tf.constant(0.0, dtype=tf.float64), tf.constant(0, dtype=tf.int32)

    # ── TF phase: 4 ops total (same as NW) ───────────────────────────────────
    A_tf      = tf.constant(A_buf[:p],    dtype=tf.float64)   # [n_pairs, 3, 3]
    r_i_tf    = tf.gather(R_T, i_buf[:p])                      # [n_pairs, 3]
    r_j_tf    = tf.gather(R_T, j_buf[:p])                      # [n_pairs, 3]
    total_du_du = tf.einsum('pa,pab,pb->', r_i_tf, A_tf, r_j_tf)

    f_tf      = tf.constant(f_buf[:nv],   dtype=tf.float64)   # [n_v, 3]
    r_sel_tf  = tf.gather(R_T, vi_buf[:nv])                    # [n_v, 3]
    total_u_f = tf.reduce_sum(r_sel_tf * f_tf)

    return total_du_du, total_u_f, tf.constant(p, dtype=tf.int32)
