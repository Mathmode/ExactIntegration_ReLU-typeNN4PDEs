#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 11:05:41 2026

@author: sbahalouei
"""

import numpy as np

def get_exact_conical_quad(n_points=3):
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

def triangulation_integral(vertices, px, py):
    """
    Decomposes polygon into triangles using the center point and edges.
    """
    v = np.array(vertices, dtype=float)
    n = len(v)
    
    center = np.mean(v, axis=0)
    
    q_pts, q_wts = get_exact_conical_quad(n_points=3)
    
    total_integral = 0.0
    
    # 3. Sum integrals over triangles (Center, V_i, V_{i+1})
    for i in range(n):
        v1 = v[i]
        v2 = v[(i + 1) % n] # Wraps around to the first vertex
        
        total_integral += integrate_monomial_on_triangle(center, v1, v2, px, py, q_pts, q_wts)
        
    return total_integral

# --- TEST ---
unit_square = [(0, 0), (1, 0), (1, 1), (0, 1)]
px, py = 2, 2 

result = triangulation_integral(unit_square, px, py)

print(f"Domain: {unit_square}")
print(f"Centroid-based Result for x^2: {result:.16f}")