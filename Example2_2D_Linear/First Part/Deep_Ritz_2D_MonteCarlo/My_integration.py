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
import sympy as sp

from Data import exact_u, f_rhs, exact_dx, exact_dy

@tf.function
def Lasserre_integration_dReLu_dReLu(u_net, num_neurons, domain_vertices, xy):
    
    
    with tf.GradientTape() as tape:
        tape.watch(xy)
        u = u_net(xy)   # shape (n_samples, 1)
    
    grads = tape.gradient(u, xy)   # shape (n_samples, 2)
    
    Integral_du_du = tf.reduce_mean(tf.reduce_sum(grads**2, axis=1))  # shape (n_samples,)
    
    # weights_1 = u_net.layers[0].weights[0]
    # biases_1 = u_net.layers[0].weights[1]
    # weights_2 = u_net.layers[-1].weights[0]
    
    # Integral_du_du = 0
    
    # for i in range(num_neurons):
        
    #     ci = weights_2[i, 0]
    #     wi0 = weights_1[0, i]
    #     wi1 = weights_1[1, i]
    #     bi = biases_1[i]
        
        
    #     line_coef = tf.stack([wi0, wi1, bi])
    #     line_coef = tf.cast(line_coef, dtype=tf.float32)
        
    #     domain_i = clip_polygon_with_line_diff(domain_vertices, line_coef)
    #     # if domain_i is None:
    #     #     continue
        
        
    #     for j in range(num_neurons):
            
    #         cj = weights_2[j, 0]
    #         wj0 = weights_1[0, j]
    #         wj1 = weights_1[1, j]
    #         bj = biases_1[j]
    #         cij = ci*cj
            
    #         line_coef = tf.stack([wj0, wj1, bj])
    #         line_coef = tf.cast(line_coef, dtype=tf.float32)
            
    #         final_domain = clip_polygon_with_line_diff(domain_i, line_coef)

            
    #         # plot_clipping_result(
    #         #     domain_vertices,
    #         #     inter_i + inter_j, # Show all intersection points
    #         #     final_domain,      # Show the region where BOTH are active
    #         #     line_i,
    #         #     line_j,
    #         #     wi0,wi1, bi,
    #         #     wj0,wj1, bj,
    #         #     title=f"Intersection: Neuron {i} and {j}"
    #         # )
            
    #         coeffs_1, deg_x_1, deg_y_1, coeffs_2, deg_x_2, deg_y_2 = compute_numeric_coeffs(wi0, wi1, bi, wj0, wj1, bj)

    #             # Integral_du_du += 0
    #         if final_domain is not None and tf.shape(final_domain)[0] > 0:
                
    #             term_1 = 0
    #             term_2 = 0

    #             for c1, dx1, dy1 in zip(coeffs_1, deg_x_1, deg_y_1):
    #                 term_1 += c1 * triangulation_integral(final_domain, dx1, dy1)
                
    #             for c2, dx2, dy2 in zip(coeffs_2, deg_x_2, deg_y_2):
    #                 term_2 += c2 * triangulation_integral(final_domain, dx2, dy2)
                    
    #             Integral_du_du += cij*(term_1 + term_2)
 
    # # print(Integral_du_du)
    
    return Integral_du_du



@tf.function
def Lasserre_integration_ReLu_ReLu_new(u_net, f_coeffs, f_degree, num_neurons_u, domain_vertices, xy):
    """
    Optimized Source Term Integration: Determine domain once per neuron.
    """
    
    u = u_net(xy)                      # (N,1)
    f_val = tf.reshape( f_rhs(xy[:, 0], xy[:, 1]), (-1, 1)) # (N,1)

    Integral_u_f = tf.reduce_mean(f_val * u)
    
    
    # u_weights_1 = u_net.layers[0].weights[0]
    # u_biases_1 = u_net.layers[0].weights[1]
    # u_weights_2 = u_net.layers[-1].weights[0]
    
    # Integral_u_f = 0.0
    # for i in range(num_neurons_u):
    #     ci = u_weights_2[i, 0]
    #     wi0 = u_weights_1[0, i]
    #     wi1 = u_weights_1[1, i]
    #     bi = u_biases_1[i]
        
    #     # Determine the active domain for this neuron ONCE
    #     line_coef = tf.stack([wi0, wi1, bi])
    #     line_coef = tf.cast(line_coef, dtype=tf.float32)
        
    #     final_domain = clip_polygon_with_line_diff(domain_vertices, line_coef)
        
    #     # plot_clipping_result_2(
    #     #             domain_vertices,
    #     #             inter,
    #     #             new_domain,
    #     #             line_coords,
    #     #             -wi0/wi1,
    #     #             -bi/wi0,
    #     #             title="Intersection Case"
    #     #         )
        
        
    #     # Check if the tensor has any points (rows)
    #     # if new_domain is not None and tf.shape(new_domain)[0] > 0:
    #         # Convert to numpy to use in your triangulation/integration logic
    
    
        
    #     # Loop through polynomial coefficients of f
    #     for j in range(f_degree[1] + 1):
    #         for k in range(f_degree[0] + 1):
                
    #             f_jk = f_coeffs[j, k]
    #             if abs(f_jk) > 1e-6: 
    #             # Compute the 8 monomial components of (ReLU * Mask)
    #                 I1 = triangulation_integral(final_domain, j+3, k+2)
    #                 I2 = triangulation_integral(final_domain, j+2, k+2)
    #                 I3 = triangulation_integral(final_domain, j+3, k+1)
    #                 I4 = triangulation_integral(final_domain, j+2, k+3)
    #                 I5 = triangulation_integral(final_domain, j+2, k+1)
    #                 I6 = triangulation_integral(final_domain, j+1, k+3)
    #                 I7 = triangulation_integral(final_domain, j+1, k+2)
    #                 I8 = triangulation_integral(final_domain, j+1, k+1)
                
    #                 Integral_u_f += ci * f_jk * (
    #                     (wi0)*I1 + (-wi0-wi1+bi)*I2 + (-wi0)*I3 +
    #                     (wi1)*I4 + (wi0-bi)*I5 + (-wi1)*I6 +
    #                     (wi1-bi)*I7 + (bi)*I8)
                    
        
    #     # unit_square = [(0, 0), (1, 0), (1, 1), (0, 1)]
    #     # # Define symbols
    #     # x, y = sp.symbols('x y')
    #     # u = x * y**2 * (x - 1) * (y - 1)
    #     # f = -(6*y - 2)*(x**2 - x) - 2*(y**2 - y)*y
    #     # expr = sp.expand(f * u)
    #     # poly = sp.Poly(expr, x, y)

    #     # Integral_u_f = 0.0

    #     # # 2. Iterate through each term using .as_dict()
    #     # # This returns {(pow_x, pow_y): coefficient}
    #     # terms_dict = poly.as_dict()

    #     # for (pow_x, pow_y), coeff in terms_dict.items():
    #     #     # Use your existing triangulation function
    #     #     term_val = float(coeff) * triangulation_integral(unit_square, pow_x, pow_y)
    #     #     Integral_u_f += term_val
            
    # # print(Integral_u_f)
    return Integral_u_f







#==============================================================================
#==============================================================================
#====================== integration by triangulation method ===================
#==============================================================================
#==============================================================================


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
# @tf.function
# def clip_polygon_with_line_diff(vertices, line_coeffs):
#     """
#     Differentiable clipping of a polygon against a single line.
#     """
    
    
#     vertices = tf.cast(vertices, dtype=tf.float32)
    
#     # Ensure vertices are closed
#     v_curr = vertices
#     v_prev = tf.roll(vertices, shift=1, axis=0)
    

#     # Signed distances: positive = inside
#     d_prev = tf.reduce_sum(v_prev * line_coeffs[:2], axis=1) + line_coeffs[2]

#     d_curr = tf.reduce_sum(v_curr * line_coeffs[:2], axis=1) + line_coeffs[2]

#     # Intersection points
#     t = -d_prev / (d_curr - d_prev + 1e-15)
#     intersections = v_prev + tf.expand_dims(t, axis=-1) * (v_curr - v_prev)

#     curr_inside = d_curr >= 0
#     prev_inside = d_prev >= 0

#     output_points = []
#     num_vertices = vertices.shape[0]

#     for i in range(num_vertices):
#         if curr_inside[i] != prev_inside[i]:
#             output_points.append(intersections[i])
#         if curr_inside[i]:
#             output_points.append(v_curr[i])

#     if len(output_points) == 0:
#         return tf.zeros((0, 2), dtype=vertices.dtype)

#     return tf.stack(output_points)




import tensorflow as tf

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
    vertices = tf.cast(vertices, dtype=tf.float32)
    line_coeffs = tf.cast(line_coeffs, dtype=tf.float32)

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
    denominator = tf.where(tf.abs(denominator) < 1e-9, 1e-9 * tf.sign(denominator + 1e-10), denominator)
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

#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================
    
    

#==============================================================================
#==============================================================================
#====================== integration by triangulation method ===================
#==============================================================================
#==============================================================================
@tf.function
def get_exact_conical_quad(n_points):
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    
    r = 0.5 * nodes + 0.5
    wr = 0.5 * weights
    s = 0.5 * nodes + 0.5
    ws = 0.5 * weights
    
    pts = []
    wts = []
    for i in range(n_points):
        for j in range(n_points):
            # Map Square (r,s) to Triangle (u,v)
            u = r[i]
            v = s[j] * (1 - r[i])
            weight = wr[i] * ws[j] * (1 - r[i])
            pts.append([u, v])
            wts.append(weight)
            
    return np.array(pts), np.array(wts)
@tf.function
def integrate_monomial_on_triangle(v0, v1, v2, px, py, q_pts, q_wts):
    """Integrates x^px * y^py over a triangle (v0, v1, v2)."""
    # Jacobian determinant = |(x1-x0)(y2-y0) - (x2-x0)(y1-y0)|
    det_J = abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]))
    
    # Map points to the physical triangle
    xt = v0[0] + q_pts[:, 0] * (v1[0] - v0[0]) + q_pts[:, 1] * (v2[0] - v0[0])
    yt = v0[1] + q_pts[:, 0] * (v1[1] - v0[1]) + q_pts[:, 1] * (v2[1] - v0[1])
    
    # Compute weighted sum
    total = np.sum((xt**px * yt**py) * q_wts)
    return total * det_J
@tf.function
def triangulation_integral(vertices, px, py):
    """
    Decomposes polygon into triangles using the center point and edges.
    """
    v = np.array(vertices, dtype=float)
    n = len(v)
    
    center = np.mean(v, axis=0)
    
    q_pts, q_wts = get_exact_conical_quad(n_points=15)
    
    total_integral = 0.0
    
    # 3. Sum integrals over triangles (Center, V_i, V_{i+1})
    for i in range(n):
        v1 = v[i]
        v2 = v[(i + 1) % n] # Wraps around to the first vertex
        
        total_integral += integrate_monomial_on_triangle(center, v1, v2, px, py, q_pts, q_wts)
        
    return total_integral


# @tf.function
# def get_exact_conical_quad(n_points=10):
#     """
#     TensorFlow implementation of the Duffy Transform (mapping square to triangle).
#     Note: Legendre nodes/weights are pre-computed via NumPy because 
#     TensorFlow lacks a native Gauss-Legendre generator.
#     """
#     # Pre-calculate nodes and weights using numpy (since they are constant)
#     nodes_np, weights_np = np.polynomial.legendre.leggauss(n_points)
    
#     # Cast to TensorFlow constants
#     nodes = tf.constant(nodes_np, dtype=tf.float32)
#     weights = tf.constant(weights_np, dtype=tf.float32)
    
#     # Map from [-1, 1] to [0, 1]
#     r = 0.5 * nodes + 0.5
#     wr = 0.5 * weights
#     s = 0.5 * nodes + 0.5
#     ws = 0.5 * weights
    
#     # Create meshgrid for vectorized mapping (replaces the nested Python loop)
#     R, S = tf.meshgrid(r, s, indexing='ij')
#     WR, WS = tf.meshgrid(wr, ws, indexing='ij')
    
#     # Map Square (R, S) to Triangle (U, V) via Duffy Transform
#     U = R
#     V = S * (1.0 - R)
#     WTS = WR * WS * (1.0 - R)
    
#     # Reshape to (N*N, 2) and (N*N,)
#     pts = tf.stack([tf.reshape(U, [-1]), tf.reshape(V, [-1])], axis=1)
#     wts = tf.reshape(WTS, [-1])
    
#     return pts, wts

# @tf.function
# def integrate_monomial_on_triangle(v0, v1, v2, px, py, q_pts, q_wts):
#     """Integrates x^px * y^py over a triangle (v0, v1, v2) using TensorFlow."""
    
#     # Jacobian determinant: Area of the triangle scaled by 2
#     # det_J = |(x1-x0)(y2-y0) - (x2-x0)(y1-y0)|
#     det_J = tf.abs((v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1]))
    
#     # Map quadrature points from reference triangle to physical triangle
#     # xt = x0 + u*(x1-x0) + v*(x2-x0)
#     xt = v0[0] + q_pts[:, 0] * (v1[0] - v0[0]) + q_pts[:, 1] * (v2[0] - v0[0])
#     yt = v0[1] + q_pts[:, 0] * (v1[1] - v0[1]) + q_pts[:, 1] * (v2[1] - v0[1])
    
#     # Compute the monomial x^px * y^py
#     # We use tf.pow for differentiability
#     monomial_vals = tf.pow(xt, tf.cast(px, tf.float32)) * tf.pow(yt, tf.cast(py, tf.float32))
    
#     # Compute weighted sum: sum(f(x,y) * weight) * Jacobian
#     total = tf.reduce_sum(monomial_vals * q_wts)
#     return total * det_J

# @tf.function
# def triangulation_integral(vertices, px, py):
#     """
#     Decomposes a polygon into triangles and integrates x^px * y^py.
#     """
#     # Ensure vertices is a float32 tensor
#     v = tf.cast(vertices, dtype=tf.float32)
#     n = tf.shape(v)[0]
    
#     # Check if we have enough vertices to form a polygon
#     # tf.cond is used to handle empty polygons safely
#     return tf.cond(n < 3, 
#         lambda: tf.constant(0.0, dtype=tf.float32),
#         lambda: _triangulate_and_sum(v, px, py, n)
#     )
# @tf.function
# def _triangulate_and_sum(v, px, py, n):
#     # 1. Compute geometric center
#     center = tf.reduce_mean(v, axis=0)
    
#     # 2. Get quadrature rule (using 10 points for high precision)
#     q_pts, q_wts = get_exact_conical_quad(n_points=10)
    
#     # 3. Use tf.roll to get the "next" vertex for each edge (V_i, V_{i+1})
#     v_next = tf.roll(v, shift=-1, axis=0)
    
#     # 4. Use tf.vectorized_map or a simple loop over the edges
#     # For a small number of vertices, a loop is fine inside tf.function
#     total_integral = 0.0
#     for i in range(tf.shape(v)[0]):
#         v1 = v[i]
#         v2 = v_next[i]
#         total_integral += integrate_monomial_on_triangle(center, v1, v2, px, py, q_pts, q_wts)
        
#     return total_integral

#==============================================================================
#==============================================================================
#==============================================================================
#==============================================================================


