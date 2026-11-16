import tensorflow as tf
import numpy as np

# ─── Basis functions ──────────────────────────────────────────────────────────
phi = [
    [(1.0, 3, 2), (-1.0, 3, 1), (-1.0, 2, 2), (1.0, 2, 1)],
    [(1.0, 2, 3), (-1.0, 2, 2), (-1.0, 1, 3), (1.0, 1, 2)],
    [(1.0, 2, 2), (-1.0, 2, 1), (-1.0, 1, 2), (1.0, 1, 1)]
]
dx_phi = [
    [(3.0, 2, 2), (-3.0, 2, 1), (-2.0, 1, 2), (2.0, 1, 1)],
    [(2.0, 1, 3), (-2.0, 1, 2), (-1.0, 0, 3), (1.0, 0, 2)],
    [(2.0, 1, 2), (-2.0, 1, 1), (-1.0, 0, 2), (1.0, 0, 1)]
]
dy_phi = [
    [(2.0, 3, 1), (-1.0, 3, 0), (-2.0, 2, 1), (1.0, 2, 0)],
    [(3.0, 2, 2), (-2.0, 2, 1), (-3.0, 1, 2), (2.0, 1, 1)],
    [(2.0, 2, 1), (-1.0, 2, 0), (-2.0, 1, 1), (1.0, 1, 0)]
]


def poly_mul(p1, p2):
    res = []
    for c1, x1, y1 in p1:
        for c2, x2, y2 in p2:
            res.append((c1 * c2, x1 + x2, y1 + y2))
    return res


# ─── Module-level precomputation of coefficient tensors ───────────────────────
K_C  = np.zeros((3, 3, 32)); K_DX  = np.zeros((3, 3, 32)); K_DY  = np.zeros((3, 3, 32))
B_C  = np.zeros((3, 3, 16)); B_DX  = np.zeros((3, 3, 16)); B_DY  = np.zeros((3, 3, 16))

for _a in range(3):
    for _b in range(3):
        _p_dx  = poly_mul(dx_phi[_a], dx_phi[_b])
        _p_dy  = poly_mul(dy_phi[_a], dy_phi[_b])
        _k_poly = _p_dx + _p_dy
        for _k, (_c, _dx, _dy) in enumerate(_k_poly):
            K_C[_a,_b,_k]=_c; K_DX[_a,_b,_k]=_dx; K_DY[_a,_b,_k]=_dy
        _b_poly = poly_mul(phi[_a], phi[_b])
        for _k, (_c, _dx, _dy) in enumerate(_b_poly):
            B_C[_a,_b,_k]=_c; B_DX[_a,_b,_k]=_dx; B_DY[_a,_b,_k]=_dy

# Merge K and B into [2,3,3,32] so compute_matrices_KB is called ONCE per (i,j)
_n_K = K_C.shape[2]; _n_B = B_C.shape[2]; _pad = _n_K - _n_B
_B_C_padded  = np.pad(B_C,  [(0,0),(0,0),(0,_pad)])
_B_DX_padded = np.pad(B_DX, [(0,0),(0,0),(0,_pad)])
_B_DY_padded = np.pad(B_DY, [(0,0),(0,0),(0,_pad)])
KB_C_tf  = tf.constant(np.stack([K_C,  _B_C_padded],  axis=0), dtype=tf.float64)  # [2,3,3,32]
KB_DX_tf = tf.constant(np.stack([K_DX, _B_DX_padded], axis=0), dtype=tf.float64)
KB_DY_tf = tf.constant(np.stack([K_DY, _B_DY_padded], axis=0), dtype=tf.float64)

# Precomputed phi arrays for vectorised build_F_tensors
_phi_c_tf  = tf.constant([[c  for c,dx,dy in phi[a]] for a in range(3)], dtype=tf.float64)
_phi_dx_tf = tf.constant([[dx for c,dx,dy in phi[a]] for a in range(3)], dtype=tf.float64)
_phi_dy_tf = tf.constant([[dy for c,dx,dy in phi[a]] for a in range(3)], dtype=tf.float64)

# NumPy mirrors — used in the FB loop to avoid per-pair TF kernel-launch overhead
_KB_C_np  = KB_C_tf.numpy()   # [2, 3, 3, 32]
_KB_DX_np = KB_DX_tf.numpy()
_KB_DY_np = KB_DY_tf.numpy()


# ─── Vectorised F-tensor construction ────────────────────────────────────────
@tf.function
def build_F_tensors(f_coeffs, f_degree):
    deg_x = f_degree[0]; deg_y = f_degree[1]
    f_flat = tf.cast(tf.reshape(f_coeffs, [-1]), tf.float64)
    j1 = tf.cast(tf.range(deg_x + 1), tf.float64)
    k1 = tf.cast(tf.range(deg_y + 1), tf.float64)
    j1_flat = tf.reshape(tf.tile(tf.expand_dims(j1, 1), [1, deg_y+1]), [-1])
    k1_flat = tf.reshape(tf.tile(tf.expand_dims(k1, 0), [deg_x+1, 1]), [-1])
    C_all  = _phi_c_tf[:, :, tf.newaxis]  * f_flat[tf.newaxis, tf.newaxis, :]   # [3,4,n]
    DX_all = _phi_dx_tf[:, :, tf.newaxis] + j1_flat[tf.newaxis, tf.newaxis, :]
    DY_all = _phi_dy_tf[:, :, tf.newaxis] + k1_flat[tf.newaxis, tf.newaxis, :]
    n_total = 4 * (deg_x + 1) * (deg_y + 1)
    return (tf.reshape(C_all,  [3, n_total]),
            tf.reshape(DX_all, [3, n_total]),
            tf.reshape(DY_all, [3, n_total]))


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


# ─── Shared polygon mapping ───────────────────────────────────────────────────
@tf.function(reduce_retracing=True)
def _map_polygon(vertices, q_pts, q_wts):
    """Map quadrature points onto polygon triangles. Returns xt, yt, det_J."""
    v     = tf.cast(vertices, tf.float64)
    q_pts = tf.cast(q_pts,   tf.float64)
    q_wts = tf.cast(q_wts,   tf.float64)
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


# ─── Specialised integration kernels ─────────────────────────────────────────
@tf.function(reduce_retracing=True)
def compute_matrices_F(vertices, q_pts, q_wts, C_tensor, DX_tensor, DY_tensor):
    """F integrals.  C/DX/DY: [3, n_terms].  Returns [3]."""
    xt, yt, det_J, q_wts = _map_polygon(vertices, q_pts, q_wts)
    xt_e = xt[:, :, tf.newaxis, tf.newaxis]
    yt_e = yt[:, :, tf.newaxis, tf.newaxis]
    poly_val = tf.reduce_sum(
        C_tensor[tf.newaxis, tf.newaxis, :, :] * tf.pow(xt_e, DX_tensor[tf.newaxis, tf.newaxis, :, :])
        * tf.pow(yt_e, DY_tensor[tf.newaxis, tf.newaxis, :, :]), axis=-1)          # [ntri,nq,3]
    tri_int  = tf.reduce_sum(poly_val * q_wts[tf.newaxis, :, tf.newaxis], axis=1)  # [ntri,3]
    return tf.reduce_sum(tri_int * det_J[:, tf.newaxis], axis=0)                   # [3]


@tf.function(reduce_retracing=True)
def compute_matrices_KB(vertices, q_pts, q_wts):
    """KB integrals using global KB tensors.  Returns [2, 3, 3]."""
    xt, yt, det_J, q_wts = _map_polygon(vertices, q_pts, q_wts)
    xt_e = xt[:, :, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
    yt_e = yt[:, :, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
    poly_val = tf.reduce_sum(
        KB_C_tf [tf.newaxis, tf.newaxis, :, :, :, :] * tf.pow(xt_e, KB_DX_tf[tf.newaxis, tf.newaxis, :, :, :, :])
        * tf.pow(yt_e, KB_DY_tf[tf.newaxis, tf.newaxis, :, :, :, :]), axis=-1)    # [ntri,nq,2,3,3]
    tri_int  = tf.reduce_sum(
        poly_val * q_wts[tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis], axis=1)  # [ntri,2,3,3]
    return tf.reduce_sum(tri_int * det_J[:, tf.newaxis, tf.newaxis, tf.newaxis], axis=0)  # [2,3,3]


@tf.function(reduce_retracing=True)
def compute_matrices_KB_F(vertices, q_pts, q_wts, F_C, F_DX, F_DY):
    """Compute KB [2,3,3] AND F [3] in one pass — shares the polygon mapping.
    Use for diagonal (i==j) terms to avoid mapping the same polygon twice."""
    xt, yt, det_J, q_wts = _map_polygon(vertices, q_pts, q_wts)

    # ── F integrals [3] ───────────────────────────────────────────────────────
    xt_e4 = xt[:, :, tf.newaxis, tf.newaxis]
    yt_e4 = yt[:, :, tf.newaxis, tf.newaxis]
    poly_val_f = tf.reduce_sum(
        F_C [tf.newaxis, tf.newaxis, :, :] * tf.pow(xt_e4, F_DX[tf.newaxis, tf.newaxis, :, :])
        * tf.pow(yt_e4, F_DY[tf.newaxis, tf.newaxis, :, :]), axis=-1)              # [ntri,nq,3]
    tri_int_f  = tf.reduce_sum(poly_val_f * q_wts[tf.newaxis, :, tf.newaxis], axis=1)  # [ntri,3]
    F_int      = tf.reduce_sum(tri_int_f * det_J[:, tf.newaxis], axis=0)               # [3]

    # ── KB integrals [2,3,3] ─────────────────────────────────────────────────
    xt_e6 = xt[:, :, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
    yt_e6 = yt[:, :, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis]
    poly_val_kb = tf.reduce_sum(
        KB_C_tf [tf.newaxis, tf.newaxis, :, :, :, :] * tf.pow(xt_e6, KB_DX_tf[tf.newaxis, tf.newaxis, :, :, :, :])
        * tf.pow(yt_e6, KB_DY_tf[tf.newaxis, tf.newaxis, :, :, :, :]), axis=-1)    # [ntri,nq,2,3,3]
    tri_int_kb = tf.reduce_sum(
        poly_val_kb * q_wts[tf.newaxis, :, tf.newaxis, tf.newaxis, tf.newaxis], axis=1)  # [ntri,2,3,3]
    KB_int = tf.reduce_sum(tri_int_kb * det_J[:, tf.newaxis, tf.newaxis, tf.newaxis], axis=0)  # [2,3,3]

    return KB_int, F_int


# ─── NumPy fast kernels (used in FB to avoid TF per-call overhead) ───────────

def _clip_np(v, lc):
    """Sutherland–Hodgman half-plane clip — pure NumPy, no TF overhead."""
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
    """Fan-triangulate polygon and map quadrature points — NumPy.
    Returns xt [ntri,nq], yt [ntri,nq], det_J [ntri]."""
    c  = v.mean(0)
    v1 = v; v2 = np.roll(v, -1, 0)
    dJ = np.abs((v1[:,0]-c[0])*(v2[:,1]-c[1]) - (v2[:,0]-c[0])*(v1[:,1]-c[1]))
    xt = c[0] + q_pts[None,:,0]*(v1[:,None,0]-c[0]) + q_pts[None,:,1]*(v2[:,None,0]-c[0])
    yt = c[1] + q_pts[None,:,0]*(v1[:,None,1]-c[1]) + q_pts[None,:,1]*(v2[:,None,1]-c[1])
    return xt, yt, dJ


def _KB_np(v, q_pts, q_wts):
    """KB integral [2,3,3] — pure NumPy."""
    xt, yt, dJ = _map_np(v, q_pts, q_wts)
    xe = xt[:,:,None,None,None,None]; ye = yt[:,:,None,None,None,None]
    poly = np.sum(_KB_C_np * xe**_KB_DX_np * ye**_KB_DY_np, axis=-1)  # (ntri,nq,2,3,3)
    return np.einsum('tqabc,q,t->abc', poly, q_wts, dJ)


def _KB_F_np(v, q_pts, q_wts, Fc, Fdx, Fdy):
    """KB [2,3,3] and F [3] — one polygon mapping, pure NumPy."""
    xt, yt, dJ = _map_np(v, q_pts, q_wts)
    # F integrals
    xe4 = xt[:,:,None,None]; ye4 = yt[:,:,None,None]
    poly_f = np.sum(Fc * xe4**Fdx * ye4**Fdy, axis=-1)            # (ntri,nq,3)
    F_int  = np.einsum('tqk,q,t->k', poly_f, q_wts, dJ)           # (3,)
    # KB integrals
    xe6 = xt[:,:,None,None,None,None]; ye6 = yt[:,:,None,None,None,None]
    poly_kb = np.sum(_KB_C_np * xe6**_KB_DX_np * ye6**_KB_DY_np, axis=-1)  # (ntri,nq,2,3,3)
    KB_int  = np.einsum('tqabc,q,t->abc', poly_kb, q_wts, dJ)     # (2,3,3)
    return KB_int, F_int


# ─── Outer loops: eager Python (no @tf.function) ──────────────────────────────

def my_integration_FB(u_net, num_neurons, domain_vertices, pts, wts, f_coeffs, f_degree, alpha):
    weights_1 = u_net.layers[0].weights[0]
    biases_1  = u_net.layers[0].weights[1]
    weights_2 = u_net.layers[-1].weights[0]
    wi0, wi1, bi, ci = weights_1[0,:], weights_1[1,:], biases_1[:], weights_2[:,0]

    F_C, F_DX, F_DY = build_F_tensors(f_coeffs, f_degree)

    # ── NumPy phase: build full 3N×3N stiffness matrix and 3N load vector ──────
    wi0_np = wi0.numpy(); wi1_np = wi1.numpy()
    bi_np  = bi.numpy();  ci_np  = ci.numpy()
    alpha_np   = float(alpha)
    q_pts_np   = np.asarray(pts,   dtype=np.float64)
    q_wts_np   = np.asarray(wts,   dtype=np.float64)
    F_C_np     = F_C.numpy();  F_DX_np = F_DX.numpy();  F_DY_np = F_DY.numpy()
    dv_np      = np.asarray(domain_vertices, dtype=np.float64)
    all_lc_np  = np.stack([wi0_np, wi1_np, bi_np], axis=1)  # [N, 3]
    N3         = 3 * num_neurons

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

        # Diagonal: KB and F share one polygon mapping
        KB_ii, F_i = _KB_F_np(di, q_pts_np, q_wts_np, F_C_np, F_DX_np, F_DY_np)
        M_np[si, si] += KB_ii[0] + alpha_np * KB_ii[1]
        f_np[si]     += F_i

        # Off-diagonal (j > i): exploit symmetry K_ij = K_ji
        for j in range(i + 1, num_neurons):
            dij = _clip_np(di, all_lc_np[j])
            if len(dij) == 0:
                continue
            sj = slice(3*j, 3*j+3)
            KB_ij = _KB_np(dij, q_pts_np, q_wts_np)
            A_ij  = KB_ij[0] + alpha_np * KB_ij[1]
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
                       pts, wts, f_coeffs, f_degree, alpha):
    """Neuron-wise integration — partial-M matmul (same TF phase as FB).

    Key ideas:
    1. **Partial stiffness matrix**: build [3N×3N] M_np but only fill rows
       corresponding to sampled_i and their upper-triangle j partners.
    2. **Symmetry exploit**: M_np[si,sj] = M_np[sj,si] — fill both sides, no 2× factor.
    3. **Same TF phase as FB**: ONE BLAS matvec, simple gradient (another matvec).
       The old buffer+einsum approach (tf.gather + tf.einsum 'pa,pab,pb->') had
       a scatter-based backward pass that was costlier than a matvec gradient.

    Clip count vs FB
    ----------------
      FB : N domain clips  +  N*(N-1)/2 inner clips
      NW : n_i domain clips + Σ_{i∈S} (N-1-i) clips  ≈ N²/8  (n_i=N/2)
    → NW has ~2× fewer _KB_np calls than FB → ~2× speedup guaranteed.
    """
    weights_1 = u_net.layers[0].weights[0]
    biases_1  = u_net.layers[0].weights[1]
    weights_2 = u_net.layers[-1].weights[0]
    wi0, wi1, bi, ci = weights_1[0,:], weights_1[1,:], biases_1[:], weights_2[:,0]

    F_C, F_DX, F_DY = build_F_tensors(f_coeffs, f_degree)
    wi0_np = wi0.numpy(); wi1_np = wi1.numpy()
    bi_np  = bi.numpy()
    alpha_np  = float(alpha)
    q_pts_np  = np.asarray(pts,  dtype=np.float64)
    q_wts_np  = np.asarray(wts,  dtype=np.float64)
    F_C_np = F_C.numpy(); F_DX_np = F_DX.numpy(); F_DY_np = F_DY.numpy()
    dv_np     = np.asarray(domain_vertices, dtype=np.float64)
    all_lc_np = np.stack([wi0_np, wi1_np, bi_np], axis=1)  # [N, 3]
    N3        = 3 * num_neurons

    # ── Geometry phase: build partial [3N×3N] stiffness + [3N] load ──────────
    M_np = np.zeros((N3, N3))
    f_np = np.zeros(N3)

    for i in sampled_i_list:
        i = int(i)
        domain_i = _clip_np(dv_np, all_lc_np[i])
        if len(domain_i) == 0:
            continue
        si = slice(3*i, 3*i+3)

        KB_ii, f_i = _KB_F_np(domain_i, q_pts_np, q_wts_np, F_C_np, F_DX_np, F_DY_np)
        M_np[si, si] += KB_ii[0] + alpha_np * KB_ii[1]
        f_np[si]     += f_i

        for j in range(i + 1, num_neurons):            # UPPER TRIANGLE only
            dij = _clip_np(domain_i, all_lc_np[j])
            if len(dij) == 0:
                continue
            sj = slice(3*j, 3*j+3)
            KB_ij = _KB_np(dij, q_pts_np, q_wts_np)
            A_ij  = KB_ij[0] + alpha_np * KB_ij[1]
            M_np[si, sj] += A_ij
            M_np[sj, si] += A_ij    # symmetry — no 2× factor needed

    # ── TF phase: ONE matvec — same as FB (cheap BLAS gradient) ──────────────
    M_tf = tf.constant(M_np, dtype=tf.float64)
    f_tf = tf.constant(f_np, dtype=tf.float64)

    wi0_f = tf.cast(wi0, tf.float64); wi1_f = tf.cast(wi1, tf.float64)
    bi_f  = tf.cast(bi,  tf.float64); ci_f  = tf.cast(ci,  tf.float64)
    R_tf   = tf.stack([ci_f*wi0_f, ci_f*wi1_f, ci_f*bi_f], axis=0)
    r_flat = tf.reshape(tf.transpose(R_tf), [N3])

    Integral_du_du = tf.reduce_sum(r_flat * tf.linalg.matvec(M_tf, r_flat))
    Integral_u_f   = tf.reduce_sum(r_flat * f_tf)
    return Integral_du_du, Integral_u_f


def my_integration_DNW(u_net, num_neurons, domain_vertices, sampled_i_list,
                        pts, wts, f_coeffs, f_degree, alpha):
    """Double neuron-wise — partial-M matmul with random j subset.

    Same structure as NW but the inner j loop uses a random subset of {i+1,...,N-1}.
    TF phase: identical ONE matvec as NW and FB — no einsum, no scatter gradient.

    Clip count vs NW
    ----------------
      NW  : Σ_{i∈S} (N-1-i) clips            (ALL j > i)
      DNW : Σ_{i∈S} n_j_samp clips  ≈ NW/2   (random ~half of j > i)
    → DNW has ~2× fewer _KB_np calls than NW → ~2× speedup over NW.
    """
    weights_1 = u_net.layers[0].weights[0]
    biases_1  = u_net.layers[0].weights[1]
    weights_2 = u_net.layers[-1].weights[0]
    wi0, wi1, bi, ci = weights_1[0,:], weights_1[1,:], biases_1[:], weights_2[:,0]

    F_C, F_DX, F_DY = build_F_tensors(f_coeffs, f_degree)
    wi0_np = wi0.numpy(); wi1_np = wi1.numpy()
    bi_np  = bi.numpy()
    alpha_np  = float(alpha)
    q_pts_np  = np.asarray(pts,  dtype=np.float64)
    q_wts_np  = np.asarray(wts,  dtype=np.float64)
    F_C_np = F_C.numpy(); F_DX_np = F_DX.numpy(); F_DY_np = F_DY.numpy()
    dv_np     = np.asarray(domain_vertices, dtype=np.float64)
    all_lc_np = np.stack([wi0_np, wi1_np, bi_np], axis=1)  # [N, 3]
    N3        = 3 * num_neurons

    # ── Geometry phase: build partial [3N×3N] stiffness + [3N] load ──────────
    M_np = np.zeros((N3, N3))
    f_np = np.zeros(N3)
    n_pairs = 0  # off-diagonal pairs processed (returned as diagnostic)

    for i in sampled_i_list:
        i = int(i)
        domain_i = _clip_np(dv_np, all_lc_np[i])
        if len(domain_i) == 0:
            continue
        si = slice(3*i, 3*i+3)

        KB_ii, f_i = _KB_F_np(domain_i, q_pts_np, q_wts_np, F_C_np, F_DX_np, F_DY_np)
        M_np[si, si] += KB_ii[0] + alpha_np * KB_ii[1]
        f_np[si]     += f_i

        # Random subset of upper-triangle j indices
        upper_j = np.arange(i + 1, num_neurons)
        if len(upper_j) > 0:
            n_j_samp = np.random.randint(1, len(upper_j) + 1)
            np.random.shuffle(upper_j)
            for j in upper_j[:n_j_samp]:                # RANDOM SUBSET, upper tri
                dij = _clip_np(domain_i, all_lc_np[j])
                if len(dij) == 0:
                    continue
                sj = slice(3*j, 3*j+3)
                KB_ij = _KB_np(dij, q_pts_np, q_wts_np)
                A_ij  = KB_ij[0] + alpha_np * KB_ij[1]
                M_np[si, sj] += A_ij
                M_np[sj, si] += A_ij    # symmetry — no 2× factor needed
                n_pairs += 1

    # ── TF phase: ONE matvec — same as FB and NW ─────────────────────────────
    M_tf = tf.constant(M_np, dtype=tf.float64)
    f_tf = tf.constant(f_np, dtype=tf.float64)

    wi0_f = tf.cast(wi0, tf.float64); wi1_f = tf.cast(wi1, tf.float64)
    bi_f  = tf.cast(bi,  tf.float64); ci_f  = tf.cast(ci,  tf.float64)
    R_tf   = tf.stack([ci_f*wi0_f, ci_f*wi1_f, ci_f*bi_f], axis=0)
    r_flat = tf.reshape(tf.transpose(R_tf), [N3])

    Integral_du_du = tf.reduce_sum(r_flat * tf.linalg.matvec(M_tf, r_flat))
    Integral_u_f   = tf.reduce_sum(r_flat * f_tf)
    return Integral_du_du, Integral_u_f, tf.constant(n_pairs, dtype=tf.int32)
