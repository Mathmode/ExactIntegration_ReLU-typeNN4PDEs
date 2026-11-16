#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:47:17 2025

@author: sbahalouei
"""

import tensorflow as tf
import numpy as np


# Exact solution
def exact_u(x, y):
    return (x**2 - x)*(y**2 - y)*(x)*(y)

def exact_dx(x, y):
    return (2*x - 1)*(y**2 - y)*(x)*(y) + (x**2 - x)*(y**2 - y)*(y)

def exact_dy(x, y):
    return (x**2 - x)*(2*y - 1)*(x)*(y) + (x**2 - x)*(y**2 - y)*(x)

# Right-hand side
def f_rhs(x, y):
   return -(2*(x**2 - x)*(x)*(y) + \
            (x**2 - x)*(2*y - 1)*(x) + \
            (x**2 - x)*(2*y - 1)*(x))+ \
          -(2*(y**2 - y)*(x)*(y) + 
            (2*x - 1)*(y**2 - y)*(y) + 
            (2*x - 1)*(y**2 - y)*(y))


# import tensorflow as tf

# # Exact solution
# def exact_u(x, y):
#     A = (x**2 - x)
#     B = (y**2 - y)
#     P = (x*y + x + y + x**2 + y**2)
#     return A * B * P

# def exact_dx(x, y):
#     A = (x**2 - x)
#     dA_dx = (2*x - 1)
#     B = (y**2 - y)
#     P = (x*y + x + y + x**2 + y**2)
#     dP_dx = (y + 1.0 + 2*x)
    
#     return (dA_dx * B * P) + (A * B * dP_dx)

# def exact_dy(x, y):
#     A = (x**2 - x)
#     B = (y**2 - y)
#     dB_dy = (2*y - 1)
#     P = (x*y + x + y + x**2 + y**2)
#     dP_dy = (x + 1.0 + 2*y)
    
#     return (A * dB_dy * P) + (A * B * dP_dy)

# # Right-hand side f = -(u_xx + u_yy)
# def f_rhs(x, y):
#     A = (x**2 - x)
#     dA_dx = (2*x - 1)
    
#     B = (y**2 - y)
#     dB_dy = (2*y - 1)
    
#     P = (x*y + x + y + x**2 + y**2)
#     dP_dx = (y + 1.0 + 2*x)
#     dP_dy = (x + 1.0 + 2*y)
    
#     # Using product rule for u_xx = A''*B*P + 2*A'*B*P_x + A*B*P_xx 
#     # Note: A'' = 2, B'' = 2, P_xx = 2, P_yy = 2
#     u_xx = (2.0 * P * B) + (2.0 * dA_dx * dP_dx * B) + (A * B * 2.0)
    
#     # Using product rule for u_yy = A*B''*P + 2*A*B'*P_y + A*B*P_yy
#     u_yy = (A * 2.0 * P) + (2.0 * A * dB_dy * dP_dy) + (A * B * 2.0)
    
#     return -(u_xx + u_yy)







# # Exact solution: u(x,y) = sin(πx) * sin(πy)
# def exact_u(x, y):
#     return tf.sin(np.pi * x) * tf.sin(np.pi * y)
# # Partial derivative with respect to x
# def exact_dx(x, y):
#     return np.pi * tf.cos(np.pi * x) * tf.sin(np.pi * y)
# # Partial derivative with respect to y
# def exact_dy(x, y):
#     return np.pi * tf.sin(np.pi * x) * tf.cos(np.pi * y)
# # Right-hand side: f(x,y) = - \Delta u = -(u_xx + u_yy)
# def f_rhs(x, y):
#     # u_xx = -π^2 * sin(πx) * sin(πy)
#     # u_yy = -π^2 * sin(πx) * sin(πy)
    
#     # f = - ( -2π^2 * sin(πx) * sin(πy) )
#     return 2.0 * (np.pi ** 2) * tf.sin(np.pi * x) * tf.sin(np.pi * y)








# import tensorflow as tf

# # Exact solution: u(x,y) = (x^2 - x)(y^2 - y)exp(x + y)
# def exact_u(x, y):
#     return (x**2 - x) * (y**2 - y) * tf.exp(x + y)

# # Partial derivative with respect to x
# # Derived from product rule: d/dx [(x^2 - x)exp(x)] = (x^2 + x - 1)exp(x)
# def exact_dx(x, y):
#     return (x**2 + x - 1) * (y**2 - y) * tf.exp(x + y)

# # Partial derivative with respect to y
# # Derived from product rule: d/dy [(y^2 - y)exp(y)] = (y^2 + y - 1)exp(y)
# def exact_dy(x, y):
#     return (x**2 - x) * (y**2 + y - 1) * tf.exp(x + y)

# # Right-hand side: f(x,y) = - \Delta u = -(u_xx + u_yy)
# def f_rhs(x, y):
#     # Second derivative with respect to x
#     # d^2/dx^2 [(x^2 - x)exp(x)] = (x^2 + 3x)exp(x)
#     u_xx = (x**2 + 3*x) * (y**2 - y) * tf.exp(x + y)
    
#     # Second derivative with respect to y
#     # d^2/dy^2 [(y^2 - y)exp(y)] = (y^2 + 3y)exp(y)
#     u_yy = (x**2 - x) * (y**2 + 3*y) * tf.exp(x + y)
    
#     return -(u_xx + u_yy)




# import tensorflow as tf
# import numpy as np

# # Exact solution: u(x,y) = sin(πx) * sin(πy)
# def exact_u(x, y):
#     return tf.sin(np.pi * x) * tf.sin(np.pi * y)
# # Partial derivative with respect to x
# def exact_dx(x, y):
#     return np.pi * tf.cos(np.pi * x) * tf.sin(np.pi * y)
# # Partial derivative with respect to y
# def exact_dy(x, y):
#     return np.pi * tf.sin(np.pi * x) * tf.cos(np.pi * y)
# # Right-hand side: f(x,y) = - \Delta u = -(u_xx + u_yy)
# def f_rhs(x, y):
#     # u_xx = -π^2 * sin(πx) * sin(πy)
#     # u_yy = -π^2 * sin(πx) * sin(πy)
    
#     # f = - ( -2π^2 * sin(πx) * sin(πy) )
#     return 2.0 * (np.pi ** 2) * tf.sin(np.pi * x) * tf.sin(np.pi * y)

