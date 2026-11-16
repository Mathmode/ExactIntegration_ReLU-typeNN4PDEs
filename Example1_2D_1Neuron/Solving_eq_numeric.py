# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Mon Jun  1 11:04:17 2026

# @author: sbahalouei
# """

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.optimize import minimize

# # ==========================================
# # 1. Setup the Domain Grid (Numerical Integration)
# # ==========================================
# N = 200  # Grid resolution
# x_1d = np.linspace(0, 1, N)
# y_1d = np.linspace(0, 1, N)
# X, Y = np.meshgrid(x_1d, y_1d)
# dx = 1.0 / (N - 1)
# dy = 1.0 / (N - 1)
# dA = dx * dy

# # Define the base boundary condition function and its derivatives
# g = X * Y * (1 - X) * (1 - Y)
# g_x = Y * (1 - Y) * (1 - 2 * X)
# g_y = X * (1 - X) * (1 - 2 * Y)

# # Define the force function
# f = 10 * (X + Y) - 10

# # ==========================================
# # 2. Define the Parametrized Functional J(alpha, b)
# # ==========================================
# def compute_A_B(alpha, b):
#     # Linear argument for the ReLU
#     L = X * np.sin(alpha) + Y * np.cos(alpha) + b
    
#     # ReLU Mask (1 where L > 0, 0 otherwise)
#     mask = (L > 0).astype(float)
    
#     # Base shape function phi (without c)
#     phi = L * g * mask
    
#     # Derivatives of phi (using chain rule and product rule where L > 0)
#     # If L > 0: phi_x = sin(alpha)*g + L*g_x
#     phi_x = (np.sin(alpha) * g + L * g_x) * mask
#     phi_y = (np.cos(alpha) * g + L * g_y) * mask
    
#     # Integrate to find A and B (Riemann sum over the grid)
#     A = np.sum(0.5 * (phi_x**2 + phi_y**2)) * dA
#     B = np.sum(phi * f) * dA
    
#     return A, B

# def objective_J(params):
#     alpha, b = params
#     A, B = compute_A_B(alpha, b)
    
#     if A < 1e-12:
#         return 1e6 # Return large penalty if A is zero to avoid division by zero
    
#     # Supervisor's logic: Substitute optimal c = B/A back into J
#     # J = 0.5 * A * (B/A)^2 - B * (B/A) = - B^2 / (2A)
#     J_val = - (B**2) / (2 * A)
#     return J_val

# # ==========================================
# # 3. Optimize (alpha, b)
# # ==========================================
# print("Optimizing alpha and b numerically...")

# # Initial guess (alpha = pi/4 corresponds to the x+y diagonal in your previous code)
# initial_guess = [np.pi/4, -0.7]

# # Bounds: alpha between 0 and 2*pi, b between -2 and 2
# bounds = [(0, 2*np.pi), (-2.0, 2.0)]

# res = minimize(objective_J, initial_guess, bounds=bounds, method='L-BFGS-B')
# alpha_opt, b_opt = res.x

# # Calculate the resulting optimal c
# A_opt, B_opt = compute_A_B(alpha_opt, b_opt)
# c_opt = B_opt / A_opt
# J_min = objective_J((alpha_opt, b_opt))

# print(f"Optimization Success: {res.success}")
# print(f"Optimal alpha = {alpha_opt:.4f} radians ({np.degrees(alpha_opt):.1f} degrees)")
# print(f"Optimal b     = {b_opt:.4f}")
# print(f"Optimal c     = {c_opt:.4f}")
# print(f"Minimum J     = {J_min:.6f}")

# # ==========================================
# # 4. Plot the Optimized Solution
# # ==========================================
# L_opt = X * np.sin(alpha_opt) + Y * np.cos(alpha_opt) + b_opt
# mask_opt = (L_opt > 0).astype(float)

# # Final solution: U = c * ReLU(L) * g
# U = c_opt * L_opt * g * mask_opt

# plt.figure(figsize=(8, 6))
# contours = plt.contourf(X, Y, U, levels=50, cmap='magma')
# plt.colorbar(contours, label=r'$u(x,y)$')

# # Plot the optimal boundary line: x*sin(alpha) + y*cos(alpha) + b = 0
# # y = (-sin(alpha)/cos(alpha))*x - b/cos(alpha)
# if np.abs(np.cos(alpha_opt)) > 1e-5:
#     x_line = np.linspace(0, 1, 100)
#     y_line = (-np.sin(alpha_opt) * x_line - b_opt) / np.cos(alpha_opt)
#     # Only plot line segments that fall within the unit square
#     valid = (y_line >= 0) & (y_line <= 1)
#     plt.plot(x_line[valid], y_line[valid], color='cyan', linestyle='--', linewidth=2, 
#              label=rf'ReLU Boundary ($\alpha$={np.degrees(alpha_opt):.1f}$^\circ$, b={b_opt:.2f})')

# plt.xlabel("x", fontsize=12)
# plt.ylabel("y", fontsize=12)
# plt.title("2D Piecewise Solution (Optimized Angle and Offset)", fontsize=13)
# plt.legend(loc="upper right")
# plt.tight_layout()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ==========================================
# 1. Setup the Domain Grid (Numerical Integration)
# ==========================================
N = 1000  # High resolution grid to capture the sharp cut precisely
x_1d = np.linspace(0, 1, N)
y_1d = np.linspace(0, 1, N)
X, Y = np.meshgrid(x_1d, y_1d)
dx = 1.0 / (N - 1)
dy = 1.0 / (N - 1)
dA = dx * dy

# Define the base boundary condition function and its derivatives
g = X * Y * (1 - X) * (1 - Y)
g_x = Y * (1 - Y) * (1 - 2 * X)
g_y = X * (1 - X) * (1 - 2 * Y)

# Define the force function
f = 10 * (X + Y) - 10

# ==========================================
# 2. Define the True ReLU Functional J(alpha, b)
# ==========================================
def compute_A_B_true_relu(alpha, b):
    # Linear argument for the ReLU: L = x*sin(alpha) + y*cos(alpha) + b
    L = X * np.sin(alpha) + Y * np.cos(alpha) + b
    
    # EXACT TRUE RELU: Hard Boolean Mask (True where L > 0, False otherwise)
    # This acts as the exact indicator function \mathbb{I}_{L > 0}
    active_zone = (L > 0)
    
    # Base shape function phi (Exactly zero where L <= 0)
    phi = np.zeros_like(L)
    phi[active_zone] = L[active_zone] * g[active_zone]
    
    # EXACT RELU DERIVATIVES:
    # d/dL(ReLU(L)) is exactly 1 when L > 0, and exactly 0 when L <= 0.
    # Therefore, by the product rule in the active zone:
    # phi_x = dL/dx * g + L * dg/dx
    phi_x = np.zeros_like(L)
    phi_y = np.zeros_like(L)
    
    phi_x[active_zone] = np.sin(alpha) * g[active_zone] + L[active_zone] * g_x[active_zone]
    phi_y[active_zone] = np.cos(alpha) * g[active_zone] + L[active_zone] * g_y[active_zone]
    
    # Integrate over the domain via Riemann Sum
    A = np.sum((phi_x**2 + phi_y**2)) * dA
    B = np.sum(phi * f) * dA
    
    return A, B

def objective_J(params):
    alpha, b = params
    A, B = compute_A_B_true_relu(alpha, b)
    
    if A < 1e-12:
        return 1e6 
        
    # Ritz Energy functional reduction formula
    return - (B**2) / (2 * A)

# ==========================================
# 3. Optimize (alpha, b) Using Nelder-Mead
# ==========================================
print("Optimizing alpha and b using True ReLU...")

initial_guess = [np.pi/7, -0.5]

# CRITICAL CHANGE: Because a true ReLU creates a non-differentiable step on a pixel grid,
# gradient-based optimizers like L-BFGS-B will instantly freeze or fail.
# We must use Nelder-Mead (Simplex), which does not use gradients.
res = minimize(objective_J, initial_guess, method='Nelder-Mead', options={'xatol': 1e-8, 'fatol': 1e-8})
alpha_opt, b_opt = res.x

# Calculate resulting parameters
A_opt, B_opt = compute_A_B_true_relu(alpha_opt, b_opt)
c_opt = B_opt / (A_opt)
J_min = objective_J((alpha_opt, b_opt))

print(f"Optimization Success: {res.success}")
print(f"Optimal alpha = {alpha_opt:.4f} rad ({np.degrees(alpha_opt):.1f} degrees)")
print(f"Optimal b     = {b_opt:.4f}")
print(f"Optimal c     = {c_opt:.4f}")
print(f"Minimum J     = {J_min:.6f}")