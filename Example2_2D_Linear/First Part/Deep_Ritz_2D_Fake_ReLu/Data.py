#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:47:17 2025

@author: sbahalouei
"""

import tensorflow as tf
import numpy as np

# # Exact solution
# def exact_u(x, y):
#     return np.sin(np.pi*x)*np.sin(np.pi*y)

# def exact_dx(x, y):
#     return np.pi*np.cos(np.pi*x)*np.sin(np.pi*y)

# def exact_dy(x, y):
#     return np.pi*np.sin(np.pi*x)*np.cos(np.pi*y)

# # Right-hand side
# def f_rhs(x, y):
#    return 2*np.pi**2*np.sin(np.pi*x)*np.sin(np.pi*y)


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

# # Exact solution
# def exact_u(x, y):
#     return (x**2 - x)*(y**2 - y)*np.sin(x*np.pi)

# def exact_dx(x, y):
#     return (2*x - 1)*(y**2 - y)*np.sin(x*np.pi) + np.cos(x*np.pi)*(x**2 - x)*(y**2 - y)*np.pi

# def exact_dy(x, y):
#     return (x**2 - x)*(2*y - 1)*np.sin(x*np.pi)

# # Right-hand side
# def f_rhs(x, y):
#    return -2*(x**2 - x)*np.sin(x*np.pi) - (2*(y**2 - y)*np.sin(x*np.pi) + 
#                                        np.cos(x*np.pi)*(2*x - 1)*(y**2 - y)*np.pi + 
#                                        -np.sin(x*np.pi)*(x**2 - x)*(y**2 - y)*np.pi**2 + 
#                                        np.cos(x*np.pi)*(2*x - 1)*(y**2 - y)*np.pi)




# # Exact solution
# def exact_u(x, y):
#     return (x**2 - x)*(y**2 - y)*y

# def exact_dx(x, y):
#     return (2*x - 1)*(y**2 - y)*y

# def exact_dy(x, y):
#     return (x**2 - x)*(3*y**2 - 2*y)

# # Right-hand side
# def f_rhs(x, y):
#    return -(6*y - 2)*(x**2 - x) - 2*(y**2 - y)*y



# # Exact solution
# def exact_u(x, y):
#     return (x**3 - x**2)*(y**3 - y**2)

# def exact_dx(x, y):
#     return (3*x**2 - 2*x)*(y**3 - y**2)

# def exact_dy(x, y):
#     return (x**3 - x**2)*(3*y**2 - 2*y)

# # Right-hand side
# def f_rhs(x, y):
#    return -(6*x - 2)*(y**3 - y**2) - (6*y - 2)*(x**3 - x**2)



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
