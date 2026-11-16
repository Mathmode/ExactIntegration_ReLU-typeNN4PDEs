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



# # Exact solution
# def exact_u(x, y):
#     return (x**2 - x)*(y**2 - y)*(x)*(y)

# def exact_dx(x, y):
#     return (2*x - 1)*(y**2 - y)*(x)*(y) + (x**2 - x)*(y**2 - y)*(y)

# def exact_dy(x, y):
#     return (x**2 - x)*(2*y - 1)*(x)*(y) + (x**2 - x)*(y**2 - y)*(x)

# # Right-hand side
# def f_rhs(x, y):
#    return -(2*(x**2 - x)*(x)*(y) + \
#             (x**2 - x)*(2*y - 1)*(x) + \
#             (x**2 - x)*(2*y - 1)*(x))+ \
#           -(2*(y**2 - y)*(x)*(y) + 
#             (2*x - 1)*(y**2 - y)*(y) + 
#             (2*x - 1)*(y**2 - y)*(y))


# # Exact solution
# def exact_u(x, y):
#     return (x**2 - x)*(y**2 - y)*(1 + y + x)

# def exact_dx(x, y):
#     return ((2*x - 1)*(y**2 - y)*(1 + y + x) +
#             (x**2 - x)*(y**2 - y))
            

# def exact_dy(x, y):
#     return ((x**2 - x)*(2*y - 1)*(1 + y + x) +
#             (x**2 - x)*(y**2 - y))

# # Right-hand side
# def f_rhs(x, y):
#    return -(2*(y**2 - y)*(1 + y + x) + (2*x - 1)*(y**2 - y) + (2*x - 1)*(y**2 - y)+
#             2*(x**2 - x)*(1 + y + x) + (x**2 - x)*(2*y - 1) + (x**2 - x)*(2*y - 1))



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




import tensorflow as tf

# Exact solution: u = A * B * P
def exact_u(x, y):
    A = (x**2 - x)
    B = (y**2 - y)
    P = (x + y + x*y + x**2 + y**2)
    
    return A * B * P

# Partial derivative with respect to x
# Product rule: u_x = A'*B*P + A*B*P_x
def exact_dx(x, y):
    A = (x**2 - x)
    dA_dx = (2*x - 1)
    
    B = (y**2 - y)
    
    P = (x + y + x*y + x**2 + y**2)
    dP_dx = (1 + y + 2*x)
    
    return (dA_dx * B * P) + (A * B * dP_dx)

# Partial derivative with respect to y
# Product rule: u_y = A*B'*P + A*B*P_y
def exact_dy(x, y):
    A = (x**2 - x)
    
    B = (y**2 - y)
    dB_dy = (2*y - 1)
    
    P = (x + y + x*y + x**2 + y**2)
    dP_dy = (1 + x + 2*y)
    
    return (A * dB_dy * P) + (A * B * dP_dy)

# Right-hand side: f(x,y) = - \Delta u = -(u_xx + u_yy)
def f_rhs(x, y):
    # x-components
    A = (x**2 - x)
    dA_dx = (2*x - 1)
    d2A_dx2 = 2.0
    
    # y-components
    B = (y**2 - y)
    dB_dy = (2*y - 1)
    d2B_dy2 = 2.0
    
    # Polynomial P components
    P = (x + y + x*y + x**2 + y**2)
    dP_dx = (1 + y + 2*x)
    d2P_dx2 = 2.0
    
    dP_dy = (1 + x + 2*y)
    d2P_dy2 = 2.0
    
    # Second derivative with respect to x
    # u_xx = (A''*P + 2*A'*P_x + A*P_xx) * B
    u_xx = (d2A_dx2 * P + 2 * dA_dx * dP_dx + A * d2P_dx2) * B
    
    # Second derivative with respect to y
    # u_yy = A * (B''*P + 2*B'*P_y + B*P_yy)
    u_yy = A * (d2B_dy2 * P + 2 * dB_dy * dP_dy + B * d2P_dy2)
    
    return -(u_xx + u_yy)


