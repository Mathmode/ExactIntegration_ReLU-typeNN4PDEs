#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 15:56:44 2026

@author: sbahalouei
"""

import numpy as np

def integrate_monomial_on_line(p1, p2, pow_x, pow_y):
    """Line integral of x^a * y^b over segment p1-p2."""
    p1, p2 = np.array(p1, dtype=float), np.array(p2, dtype=float)
    dist = np.linalg.norm(p2 - p1)
    if dist < 1e-12: return 0.0

    # Gauss-Legendre quadrature (exact for polynomials)
    degree = pow_x + pow_y
    n_points = max(1, (degree // 2) + 1)
    nodes, weights = np.polynomial.legendre.leggauss(n_points)
    
    # Map from [-1, 1] to [0, 1]
    ts = 0.5 * nodes + 0.5 
    ws = 0.5 * weights 
    
    val = 0
    for t, w in zip(ts, ws):
        xt = p1[0] + t * (p2[0] - p1[0])
        yt = p1[1] + t * (p2[1] - p1[1])
        val += (xt**pow_x * yt**pow_y) * w
    return val * dist

def lasserre_2d_monomial(vertices, f_xy):
    """
    Computes integral of x^px * y^py using Lasserre's formula:
    I = 1/(2+q) * sum( h_i * Int_{Fi} f dsigma )
    where h_i is the signed distance from origin to edge i.
    """
    pow_x, pow_y = f_xy
    v = np.array(vertices, dtype=float)
    # Ensure CCW orientation for outward normals
    area = 0.5 * np.sum(v[:,0] * np.roll(v[:,1], -1) - np.roll(v[:,0], -1) * v[:,1])
    if area < 0:
        v = v[::-1]
    
    q = pow_x + pow_y
    total_integral = 0.0
    n = len(v)
    
    for i in range(n):
        v1 = v[i]
        v2 = v[(i + 1) % n]
        
        # Edge vector
        dx = v2[0] - v1[0]
        dy = v2[1] - v1[1]
        
        # Outward normal vector a_i = (dy, -dx)
        # Note: <(x,y), (dy, -dx)> = x*dy - y*dx = b_i
        norm_a = np.sqrt(dx**2 + dy**2)
        if norm_a < 1e-12: continue
        
        # h_i = b_i / ||a_i||
        # Using v1 to calculate b_i: b_i = v1[0]*dy - v1[1]*dx
        h_i = (v1[0] * dy - v1[1] * dx) / norm_a
        
        line_int = integrate_monomial_on_line(v1, v2, pow_x, pow_y)
        total_integral += h_i * line_int
        
    return total_integral / (2 + q)

# --- TEST: Unit Square [0,1]x[0,1] ---
unit_square = [(0, 0), (1, 0), (1, 1), (0, 1)]
f_xy = (2, 2) # pow_x=2, pow_y=0

result = lasserre_2d_monomial(unit_square, f_xy)

print(f"Domain: {unit_square}")
print(f"Lasserre Result: {result:.10f}")