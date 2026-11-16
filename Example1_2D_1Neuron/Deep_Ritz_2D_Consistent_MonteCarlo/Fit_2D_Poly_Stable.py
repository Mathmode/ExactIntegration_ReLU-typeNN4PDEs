#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 07:15:18 2026

@author: sbahalouei
"""

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial
from Data import f_rhs



def fit_2d_poly_stable(deg):
    """
    Separable 1D stable fit to avoid 2D domain bugs and Runge's Phenomenon.
    """
    
    x_coords = np.linspace(0, 1, 200) # Use odd number for perfect center
    y_coords = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    Z = f_rhs(X, Y) 
    
    num_y = len(y_coords)
    x_deg_p1 = deg[0] + 1
    y_deg_p1 = deg[1] + 1
    
    # Step 1: Fit each row (fixed y) along the x-dimension
    intermediate_coeffs = np.zeros((num_y, x_deg_p1))
    for j in range(num_y):
        row_data = Z[j, :]
        # 1D Chebyshev fit is robust against domain errors
        c_fit = Chebyshev.fit(x_coords, row_data, deg[0], domain=[0, 1])
        # Convert to power basis (x^i)
        p_fit = c_fit.convert(kind=Polynomial)
        c_x = p_fit.coef
        intermediate_coeffs[j, :min(len(c_x), x_deg_p1)] = c_x[:x_deg_p1]
    
    # Step 2: Fit each x-coefficient across the y-dimension
    final_coeffs = np.zeros((x_deg_p1, y_deg_p1))
    for i in range(x_deg_p1):
        col_data = intermediate_coeffs[:, i]
        c_fit = Chebyshev.fit(y_coords, col_data, deg[1], domain=[0, 1])
        # Convert to power basis (y^j)
        p_fit = c_fit.convert(kind=Polynomial)
        c_y = p_fit.coef
        final_coeffs[i, :min(len(c_y), y_deg_p1)] = c_y[:y_deg_p1]
        
    return np.transpose(final_coeffs)