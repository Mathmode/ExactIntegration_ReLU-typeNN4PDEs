#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:45:37 2025

@author: sbahalouei
"""
import tensorflow as tf
tf.config.experimental_run_functions_eagerly(True)
import numpy as np
from shapely.geometry import Polygon, LineString, Point
from shapely.ops import split
import matplotlib.pyplot as plt
import sympy as sp
from Symbolic import compute_numeric_coeffs


from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


import tensorflow as tf
import numpy as np
from Symbolic import compute_numeric_coeffs

@tf.function
def my_integration_FB(u_net, num_neurons, domain_vertices, epoch, pts, wts, f_coeffs, f_degree):
    
    # 1. Fetch live symbolic graph variables
    alpha_tensor = u_net.analytical_layer.alpha
    b_tensor     = u_net.analytical_layer.b
    c_tensor     = u_net.analytical_layer.c
    
    # 2. Synchronize calculation tracks explicitly to 64-bit float tensors
    target_dtype = tf.float64 
    wi0_64 = tf.cast(tf.cos(alpha_tensor), dtype=target_dtype)
    wi1_64 = tf.cast(tf.sin(alpha_tensor), dtype=target_dtype)
    bi_64  = tf.cast(b_tensor, dtype=target_dtype)
    ci_64  = tf.cast(c_tensor, dtype=target_dtype)
    
    Integral_du_du = tf.constant(0.0, dtype=target_dtype)
    Integral_u_f   = tf.constant(0.0, dtype=target_dtype)
    
    # --- Energy bilinear term (du_du) integration track ---
    for i in range(num_neurons):
        # FIX: Explicitly enforce target_dtype (float64) right at creation
        wi0 = tf.cast(tf.cos(alpha_tensor), dtype=target_dtype)
        wi1 = tf.cast(tf.sin(alpha_tensor), dtype=target_dtype)
        bi  = bi_64
        
        # Now all components are float64; tf.stack will succeed perfectly
        line_coef = tf.stack([wi0, wi1, bi])
        domain_i = clip_polygon_with_line_diff(domain_vertices, line_coef)
        
        if domain_i is None:
            continue
            
        for j in range(num_neurons):
            wj0 = wi0
            wj1 = wi1
            bj  = bi
            cj  = ci_64     
            
            cij = ci_64 * cj
            line_coef_j = tf.stack([wj0, wj1, bj])
            final_domain = clip_polygon_with_line_diff(domain_i, line_coef_j)
            
            # Using the cleanly typed tensors for coefficient extraction
            coeffs_1, deg_x_1, deg_y_1, coeffs_2, deg_x_2, deg_y_2 = compute_numeric_coeffs(wi0, wi1, bi, wj0, wj1, bj)
    
            if final_domain is not None and len(final_domain) > 0:
                coords_np = domain_i #np.array(final_domain, dtype=np.float64)
                
                term_1 = tf.constant(0.0, dtype=target_dtype)
                term_2 = tf.constant(0.0, dtype=target_dtype)

                for c, dx, dy in zip(coeffs_1, deg_x_1, deg_y_1):
                    integral_val = triangulation_integral(coords_np, dx, dy, pts, wts)
                    term_1 += tf.cast(c, target_dtype) * integral_val
                
                for c, dx, dy in zip(coeffs_2, deg_x_2, deg_y_2):
                    integral_val = triangulation_integral(coords_np, dx, dy, pts, wts)
                    term_2 += tf.cast(c, target_dtype) * integral_val
                    
                Integral_du_du += tf.cast(cij, target_dtype) * (term_1 + term_2)

    # --- Source functional term (u_f) integration track ---
    for i in range(num_neurons):
        wi0 = tf.cast(tf.cos(alpha_tensor), dtype=target_dtype)
        wi1 = tf.cast(tf.sin(alpha_tensor), dtype=target_dtype)
        bi  = bi_64
        
        line_coef = tf.stack([wi0, wi1, bi])
        domain_i = clip_polygon_with_line_diff(domain_vertices, line_coef)
        
        if domain_i is None:
            continue
        coords = domain_i #np.array(domain_i, dtype=np.float64)
        
        for j1 in range(f_degree[0] + 1):
            for k1 in range(f_degree[1] + 1):
                f_jk = f_coeffs[j1, k1]
                
                I1 = tf.cast(triangulation_integral(coords, j1+3, k1+2, pts, wts), target_dtype)
                I2 = tf.cast(triangulation_integral(coords, j1+2, k1+2, pts, wts), target_dtype)
                I3 = tf.cast(triangulation_integral(coords, j1+3, k1+1, pts, wts), target_dtype)
                I4 = tf.cast(triangulation_integral(coords, j1+2, k1+3, pts, wts), target_dtype)
                I5 = tf.cast(triangulation_integral(coords, j1+2, k1+1, pts, wts), target_dtype)
                I6 = tf.cast(triangulation_integral(coords, j1+1, k1+3, pts, wts), target_dtype)
                I7 = tf.cast(triangulation_integral(coords, j1+1, k1+2, pts, wts), target_dtype)
                I8 = tf.cast(triangulation_integral(coords, j1+1, k1+1, pts, wts), target_dtype)

                Integral_u_f += ci_64 * tf.cast(f_jk, target_dtype) * (
                    wi0 * I1 + (-wi0 - wi1 + bi) * I2 + (-wi0) * I3 +
                    wi1 * I4 + (wi0 - bi) * I5 + (-wi1) * I6 +
                    (wi1 - bi) * I7 + bi * I8
                )
                
    return Integral_du_du, Integral_u_f




#==============================================================================
#==============================================================================
#====================== integration by triangulation method ===================
#==============================================================================
#==============================================================================


@tf.function
def clip_polygon_with_line_diff(vertices, line_coeffs):
    """
    Differentiable and vectorized clipping of a polygon against a single line.
    
    Args:
        vertices: Tensor of shape (N, 2)
        line_coeffs: Tensor [w0, w1, b] representing w0*x + w1*y + b >= 0
        
    Returns:
        clipped_vertices: Tensor of shape (M, 2)
    """
    vertices = tf.cast(vertices, dtype=tf.float64)
    line_coeffs = tf.cast(line_coeffs, dtype=tf.float64)

    # 1. Define current and previous vertices for edges
    v_curr = vertices
    v_prev = tf.roll(vertices, shift=1, axis=0)

    # 2. Compute signed distances (Dot product + Bias)
    # d > 0 means point is inside the active ReLU region
    d_prev = tf.reduce_sum(v_prev * line_coeffs[:2], axis=1) + line_coeffs[2]
    d_curr = tf.reduce_sum(v_curr * line_coeffs[:2], axis=1) + line_coeffs[2]

    # 3. Calculate all possible intersection points for every edge
    # Intersection occurs at t where: d_prev + t*(d_curr - d_prev) = 0
    # Small epsilon to avoid division by zero
    denominator = d_curr - d_prev
    denominator = tf.where(tf.abs(denominator) < 1e-15, 1e-15 * tf.sign(denominator + 1e-16), denominator)
    t = -d_prev / denominator
    intersections = v_prev + tf.expand_dims(t, axis=-1) * (v_curr - v_prev)

    # 4. Determine which points to keep based on Sutherland-Hodgman logic
    curr_inside = d_curr >= 0
    prev_inside = d_prev >= 0

    # Logic table:
    # Case 1: Both inside -> Keep curr
    # Case 2: Prev inside, Curr outside -> Keep intersection
    # Case 3: Prev outside, Curr inside -> Keep intersection AND curr
    # Case 4: Both outside -> Keep nothing
    
    # We construct the output by interleaving intersections and vertices
    # and using boolean masks to select the correct ones.
    
    # Selection 1: Intersections (Keep if edge crosses the boundary)
    keep_inter = tf.not_equal(curr_inside, prev_inside)
    
    # Selection 2: Current vertices (Keep if inside)
    keep_curr = curr_inside

    # To maintain differentiability and a valid tensor shape, we stack them
    # We will use boolean_mask to extract valid points
    all_potential_pts = tf.concat([
        tf.expand_dims(intersections, axis=1), 
        tf.expand_dims(v_curr, axis=1)
    ], axis=1) # Shape (N, 2, 2)
    
    all_masks = tf.stack([keep_inter, keep_curr], axis=1) # Shape (N, 2)
    
    # Flatten the first two dimensions to use boolean_mask
    flat_pts = tf.reshape(all_potential_pts, [-1, 2])
    flat_masks = tf.reshape(all_masks, [-1])
    
    clipped_vertices = tf.boolean_mask(flat_pts, flat_masks)

    # If the polygon is completely clipped away, return an empty tensor
    # but ensure it still has shape (0, 2) for consistency
    return clipped_vertices






import tensorflow as tf


def integrate_monomial_on_triangle_tf(v0, v1, v2, px, py, q_pts, q_wts):
    """Integrates x^px * y^py over a triangle (v0, v1, v2) using TF."""
    
    # Jacobian determinant
    # Ensure we use tf.abs and tensor indexing
    det_J = tf.abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]))
    
    # Map points to the physical triangle
    # q_pts is expected to be a (N, 2) tensor
    xt = v0[0] + q_pts[:, 0] * (v1[0] - v0[0]) + q_pts[:, 1] * (v2[0] - v0[0])
    yt = v0[1] + q_pts[:, 0] * (v1[1] - v0[1]) + q_pts[:, 1] * (v2[1] - v0[1])
    
    # Compute weighted sum
    # Using tf.pow for safety with float exponents
    px_f = tf.cast(px, tf.float64)
    py_f = tf.cast(py, tf.float64)
    
    term = tf.pow(xt, px_f) * tf.pow(yt, py_f)
    total = tf.reduce_sum(term * q_wts)
    
    return total * det_J


import tensorflow as tf

@tf.function
def triangulation_integral(vertices, px, py, q_pts, q_wts):
    """
    Vectorized version: No loops. Processes all triangles in a single pass.
    """
    # 1. Setup Tensors
    v = tf.cast(vertices, dtype=tf.float64)
    q_pts = tf.cast(q_pts, dtype=tf.float64)  # Shape: (Nq, 2)
    q_wts = tf.cast(q_wts, dtype=tf.float64)  # Shape: (Nq,)
    px_f = tf.cast(px, tf.float64)
    py_f = tf.cast(py, tf.float64)

    n = tf.shape(v)[0]
    center = tf.reduce_mean(v, axis=0) # Shape: (2,)

    # 2. Gather all V1 and V2 for all triangles at once
    # V1 is all vertices, V2 is vertices shifted by 1 (wrap around)
    v1 = v
    v2 = tf.roll(v, shift=-1, axis=0)

    # 3. Vectorized Jacobian Determinant
    # Shape of det_J: (n,)
    det_J = tf.abs(
        (v1[:, 0] - center[0]) * (v2[:, 1] - center[1]) - 
        (v2[:, 0] - center[0]) * (v1[:, 1] - center[1])
    )

    # 4. Map quadrature points to all triangles simultaneously
    # q_pts: (Nq, 2) -> expand to (1, Nq, 2)
    # v1, v2, center: (n, 2) -> expand to (n, 1, 2)
    c_exp = center[tf.newaxis, tf.newaxis, :]
    v1_exp = v1[:, tf.newaxis, :]
    v2_exp = v2[:, tf.newaxis, :]
    qp_exp = q_pts[tf.newaxis, :, :]

    # Formula: Physical = V0 + u*(V1-V0) + v*(V2-V0)
    # Resulting shape: (n, Nq)
    xt = c_exp[:, :, 0] + qp_exp[:, :, 0] * (v1_exp[:, :, 0] - c_exp[:, :, 0]) + \
         qp_exp[:, :, 1] * (v2_exp[:, :, 0] - c_exp[:, :, 0])
         
    yt = c_exp[:, :, 1] + qp_exp[:, :, 0] * (v1_exp[:, :, 1] - c_exp[:, :, 1]) + \
         qp_exp[:, :, 1] * (v2_exp[:, :, 1] - c_exp[:, :, 1])

    # 5. Compute Monomials for all points in all triangles
    # Shape: (n, Nq)
    terms = tf.pow(xt, px_f) * tf.pow(yt, py_f)

    # 6. Weighted sum over quadrature points for each triangle
    # Sum over axis 1 (the Nq dimension)
    triangle_integrals = tf.reduce_sum(terms * q_wts, axis=1)

    # 7. Final sum over all triangles weighted by their Jacobians
    total_integral = tf.reduce_sum(triangle_integrals * det_J)

    return total_integral










