#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 21:04:31 2025

@author: sbahalouei
"""

import numpy as np
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt

from numpy.polynomial.chebyshev import Chebyshev
import numpy as np
from numpy.polynomial.polynomial import Polynomial

# Generate data points
x = np.linspace(0, 1, 1000)
y = np.sin(np.pi * x)

# Fit a polynomial of degree N
degree = 4
coeffs = np.polynomial.polynomial.polyfit(x, y, degree)
print(coeffs)
poly = Polynomial(coeffs)
print(poly)

# Evaluate the polynomial
y_poly = poly(x)


# Plot original function
# plt.plot(x, y, label=r'$\sin(\pi x)$', color='blue', linewidth=2)
# Least squares approximation
plt.plot(x, y_poly - y, label='Least Squares Fit', linestyle='--', color='green')

# Add labels, title, and legend
plt.xlabel('$x$', fontsize=14)
plt.ylabel('$y$', fontsize=14)
plt.title('Approximation of $\sin(\pi x)$ with Polynomials', fontsize=16)
plt.legend(fontsize=12)
plt.grid(alpha=0.3)
plt.show()