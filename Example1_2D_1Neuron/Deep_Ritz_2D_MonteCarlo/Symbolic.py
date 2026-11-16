# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Sun Feb  1 14:58:25 2026

# @author: sbahalouei
# """
# import tensorflow as tf
# import sympy as sp


# import numpy as np
# from collections import defaultdict


# # ============================================================
# # 1. Define monomials of u_x and u_y
# # ============================================================

# def ux_monomials(a):
#     """
#     u_x =
#     a1 x^2 y^2 + a2 x^2 y + a3 x y^2 + a4 x y
#     + a5 x y^3 + a6 y^3 + a7 y^2 + a8 y
#     """
#     return [
#         (a[0], 2, 2),
#         (a[1], 2, 1),
#         (a[2], 1, 2),
#         (a[3], 1, 1),
#         (a[4], 1, 3),
#         (a[5], 0, 3),
#         (a[6], 0, 2),
#         (a[7], 0, 1),
#     ]



# def uy_monomials(b):
#     """
#     u_y =
#     b1 x^2 y^2 + b2 x^2 y + b3 x y^2 + b4 x y
#     + b5 x^3 y + b6 x^3 + b7 x^2 + b8 x
#     """
#     return [
#         (b[0], 2, 2),
#         (b[1], 2, 1),
#         (b[2], 1, 2),
#         (b[3], 1, 1),
#         (b[4], 3, 1),
#         (b[5], 3, 0),
#         (b[6], 2, 0),
#         (b[7], 1, 0),
#     ]


# # ============================================================
# # 2. Square a polynomial numerically
# # ============================================================

# def square_polynomial(monomials_1, monomials_2):
#     """
#     Squares a polynomial given as (coeff, dx, dy) list.

#     Returns:
#         coeffs, deg_x, deg_y
#     """
#     acc = defaultdict(float)

#     for c1, x1, y1 in monomials_1:
#         for c2, x2, y2 in monomials_2:
#             acc[(x1 + x2, y1 + y2)] += c1 * c2

#     coeffs = []
#     deg_x = []
#     deg_y = []

#     for (dx, dy), c in acc.items():
#         if abs(c) > 1e-10:
#             coeffs.append(c)
#             deg_x.append(dx)
#             deg_y.append(dy)

#     return coeffs, deg_x, deg_y


# # ============================================================
# # 3. Compute numerical coefficients from NN parameters
# # ============================================================

# def compute_numeric_coeffs(wi0, wi1, bi, wj0, wj1, bj):
#     """
#     Fully numerical replacement for symbolic() + subs()

#     Input:
#         wi0, wi1, bi : neural network parameters

#     Output:
#         coeffs_1, deg_x_1, deg_y_1   -> (u_x)^2
#         coeffs_2, deg_x_2, deg_y_2   -> (u_y)^2
#     """

#     # -------- coefficients of u_x ----------
#     a = np.array([
#         3 * wi0,
#         -3 * wi0,
#         -2 * wi0 - 2 * wi1 + 2 * bi,
#         2 * wi0 - 2 * bi,
#         2 * wi1,
#         -wi1,
#         wi1 - bi,
#         bi
#     ], dtype=float)
    
#     a1 = np.array([
#         3 * wj0,
#         -3 * wj0,
#         -2 * wj0 - 2 * wj1 + 2 * bj,
#         2 * wj0 - 2 * bj,
#         2 * wj1,
#         -wj1,
#         wj1 - bj,
#         bj
#     ], dtype=float)

#     # -------- coefficients of u_y ----------
#     b = np.array([
#         3 * wi1,
#         -2 * wi1 - 2*wi0 + 2 * bi,
#         -3 * wi1,
#         2 * wi1 - 2 * bi,
#         2 * wi0,
#         -wi0,
#         wi0 - bi,
#         bi
#     ], dtype=float)
    
#     b1 = np.array([
#         3 * wj1,
#         -2 * wj1 - 2*wj0 + 2 * bj,
#         -3 * wj1,
#         2 * wj1 - 2 * bj,
#         2 * wj0,
#         -wj0,
#         wj0 - bj,
#         bj
#     ], dtype=float)
    

#     # Build monomials
#     ux_i = ux_monomials(a)
#     ux_j = ux_monomials(a1)
#     uy_i = uy_monomials(b)
#     uy_j = uy_monomials(b1)

#     # Square polynomials
#     coeffs_1, deg_x_1, deg_y_1 = square_polynomial(ux_i, ux_j)
#     coeffs_2, deg_x_2, deg_y_2 = square_polynomial(uy_i, uy_j)
#     return (
#         coeffs_1, deg_x_1, deg_y_1,
#         coeffs_2, deg_x_2, deg_y_2
#     )



import tensorflow as tf

# ============================================================
# 1. Define monomials of u_x and u_y (Differentiable)
# ============================================================

def ux_monomial_degrees():
    """Returns the static exponents for u_x."""
    # (deg_x, deg_y)
    return [
        (2, 2), (2, 1), (1, 2), (1, 1),
        (1, 3), (0, 3), (0, 2), (0, 1)
    ]

def uy_monomial_degrees():
    """Returns the static exponents for u_y."""
    # (deg_x, deg_y)
    return [
        (2, 2), (2, 1), (1, 2), (1, 1),
        (3, 1), (3, 0), (2, 0), (1, 0)
    ]

# ============================================================
# 2. Square a polynomial (Differentiable)
# ============================================================

def square_polynomial_tf(coeffs_i, degrees_i, coeffs_j, degrees_j):
    """
    Multiplies two polynomials (representing partial derivatives).
    Using tf.stack and nested loops that TF can trace.
    """
    res_coeffs = []
    res_deg_x = []
    res_deg_y = []

    # Since the number of monomials is small and fixed, 
    # nested loops are efficient enough for TF tracing.
    for i in range(len(degrees_i)):
        for j in range(len(degrees_j)):
            # Multiply coefficients (This preserves the gradient)
            c_prod = coeffs_i[i] * coeffs_j[j]
            
            # Add exponents
            dx = degrees_i[i][0] + degrees_j[j][0]
            dy = degrees_i[i][1] + degrees_j[j][1]
            
            res_coeffs.append(c_prod)
            res_deg_x.append(dx)
            res_deg_y.append(dy)

    # Return as Tensors
    return tf.stack(res_coeffs), res_deg_x, res_deg_y

# ============================================================
# 3. Compute numerical coefficients (Differentiable)
# ============================================

def compute_numeric_coeffs(wi0, wi1, bi, wj0, wj1, bj):
    """
    Fully differentiable replacement using TensorFlow ops.
    Input: wi0, wi1, bi, etc. must be tf.Tensors/tf.Variables
    """

    # Use tf.stack instead of np.array to keep the gradient tape connected
    a = tf.stack([
        3.0 * wi0,
        -3.0 * wi0,
        -2.0 * wi0 - 2.0 * wi1 + 2.0 * bi,
        2.0 * wi0 - 2.0 * bi,
        2.0 * wi1,
        -wi1,
        wi1 - bi,
        bi
    ])
    
    a1 = tf.stack([
        3.0 * wj0,
        -3.0 * wj0,
        -2.0 * wj0 - 2.0 * wj1 + 2.0 * bj,
        2.0 * wj0 - 2.0 * bj,
        2.0 * wj1,
        -wj1,
        wj1 - bj,
        bj
    ])

    b = tf.stack([
        3.0 * wi1,
        -2.0 * wi1 - 2.0 * wi0 + 2.0 * bi,
        -3.0 * wi1,
        2.0 * wi1 - 2.0 * bi,
        2.0 * wi0,
        -wi0,
        wi0 - bi,
        bi
    ])
    
    b1 = tf.stack([
        3.0 * wj1,
        -2.0 * wj1 - 2.0 * wj0 + 2.0 * bj,
        -3.0 * wj1,
        2.0 * wj1 - 2.0 * bj,
        2.0 * wj0,
        -wj0,
        wj0 - bj,
        bj
    ])

    # Get static degrees
    deg_ux = ux_monomial_degrees()
    deg_uy = uy_monomial_degrees()

    # Perform the multiplication (squaring/cross-product)
    coeffs_1, deg_x_1, deg_y_1 = square_polynomial_tf(a, deg_ux, a1, deg_ux)
    coeffs_2, deg_x_2, deg_y_2 = square_polynomial_tf(b, deg_uy, b1, deg_uy)

    return (
        coeffs_1, deg_x_1, deg_y_1,
        coeffs_2, deg_x_2, deg_y_2
    )