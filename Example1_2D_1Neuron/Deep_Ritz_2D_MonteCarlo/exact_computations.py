#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 07:15:14 2026

@author: sbahalouei
"""

import sympy as sp

# Define symbols
x, y = sp.symbols('x y')

# Define u and f
u = x * y**2 * (x - 1) * (y - 1)
f = -(6*y - 2)*(x**2 - x) - 2*(y**2 - y)*y

# Compute the gradient components
ux = sp.diff(u, x)
uy = sp.diff(u, y)

# Compute |Grad u|^2
grad_u_sq = ux**2 + uy**2

# Compute the integrals over the domain [0, 1] x [0, 1]
int_grad_sq = sp.integrate(grad_u_sq, (x, 0, 1), (y, 0, 1))
int_fu = sp.integrate(f * u, (x, 0, 1), (y, 0, 1))

# Final result
result = 0.5*int_grad_sq - int_fu

print(f"Integral of |Grad u|^2: {int_grad_sq}")
print(f"Integral of f*u:        {int_fu}")
print(f"Final Result (Difference): {result}")

# Compute the gradient of the integrand with respect to spatial coordinates
grad_L_x = sp.diff(result, x)
grad_L_y = sp.diff(result, y)

print("Gradient of the integrand (dL/dx):")
print(sp.simplify(grad_L_x))
print("\nGradient of the integrand (dL/dy):")
print(sp.simplify(grad_L_y))