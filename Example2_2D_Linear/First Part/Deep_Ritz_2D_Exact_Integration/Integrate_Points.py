#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 10:27:37 2026

@author: sbahalouei
"""
import numpy as np
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