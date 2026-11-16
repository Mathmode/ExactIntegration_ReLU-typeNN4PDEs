#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:27:37 2026

@author: sbahalouei
"""
import numpy as np
# def get_exact_conical_quad(n_points):
#     nodes, weights = np.polynomial.legendre.leggauss(n_points)
    
#     r = 0.5 * nodes + 0.5
#     wr = 0.5 * weights
#     s = 0.5 * nodes + 0.5
#     ws = 0.5 * weights
    
#     pts = []
#     wts = []
#     for i in range(n_points):
#         for j in range(n_points):
#             # Map Square (r,s) to Triangle (u,v)
#             u = r[i]
#             v = s[j] * (1 - r[i])
#             weight = wr[i] * ws[j] * (1 - r[i])
#             pts.append([u, v])
#             wts.append(weight)
            
#     return np.array(pts), np.array(wts)


# import numpy as np

def get_exact_conical_quad(n_points):
    """
    Generates n_points random uniform points inside the reference triangle
    (0,0), (1,0), (0,1) using the folding method, with uniform weights.
    """
    # 1. Generate random pairs in the unit square [0, 1] x [0, 1]
    u = np.random.uniform(0.0, 1.0, n_points)
    v = np.random.uniform(0.0, 1.0, n_points)
    
    # 2. Fold points that fall outside the triangle (u + v > 1) back inside
    outside = (u + v) > 1.0
    u[outside] = 1.0 - u[outside]
    v[outside] = 1.0 - v[outside]
    
    # Stack into an (n_points, 2) array
    pts = np.column_stack((u, v))
    
    # 3. Assign uniform weights
    # Note: The area of the reference triangle is 0.5. 
    # Because your original Gauss weights summed to 0.5, these weights 
    # MUST be 0.5 / n_points to act as a mathematically correct drop-in replacement.
    wts = np.full(n_points, 0.5 / n_points)
    
    return pts, wts