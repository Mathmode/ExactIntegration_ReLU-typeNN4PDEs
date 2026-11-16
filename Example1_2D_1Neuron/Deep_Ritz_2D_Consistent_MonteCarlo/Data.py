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


# Exact solution
def exact_u(x, y):
    return -5/3*x**3 + 5/2*x**2 - 5/3*y**3 + 5/2*y**2

def exact_dx(x, y):
    return -5*x**2 + 5*x

def exact_dy(x, y):
    return - 5*y**2 + 5*y

# Right-hand side
def f_rhs(x, y):
   return (10 * (x + y) - 10)

