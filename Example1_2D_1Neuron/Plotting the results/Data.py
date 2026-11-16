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
