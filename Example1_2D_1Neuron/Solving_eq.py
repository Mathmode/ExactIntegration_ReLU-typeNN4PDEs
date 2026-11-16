import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

# ==========================================
# 1. Define Symbols and 2D Expressions
# ==========================================
b, c, x, y = sp.symbols('b c x y', real=True)

# 2D Trial function for ReLU boundary: vanishes at x=1, y=1, x=0, y=0 AND on the diagonal x+y+b=0
u_theta = c * (x + y + b) * x * y * (1 - x) * (1 - y)

# Force function f(x,y). 
# Chosen to be negative near (0,0) and positive near (1,1) to push the boundary into the domain
f = 10 * (x + y) - 10

# ==========================================
# 2. Compute Functionals J1 and J2 (Diagonal ReLU Boundary)
# ==========================================
print("Calculating exact symbolic integrals over the diagonal domain (this takes a few seconds)...")
u_x = sp.diff(u_theta, x)
u_y = sp.diff(u_theta, y)

J1_integrand = 0.5 * (u_x**2 + u_y**2)
J2_integrand = u_theta * f

# Integrating over the region where x + y + b > 0 (assuming b is between -1 and 0)
# Region 1: x from 0 to -b, y from (-x - b) to 1
J1_part1 = sp.integrate(sp.integrate(J1_integrand, (y, -x - b, 1)), (x, 0, -b))
J2_part1 = sp.integrate(sp.integrate(J2_integrand, (y, -x - b, 1)), (x, 0, -b))

# Region 2: x from -b to 1, y from 0 to 1
J1_part2 = sp.integrate(sp.integrate(J1_integrand, (y, 0, 1)), (x, -b, 1))
J2_part2 = sp.integrate(sp.integrate(J2_integrand, (y, 0, 1)), (x, -b, 1))

J1 = J1_part1 + J1_part2
J2 = J2_part1 + J2_part2

# Total Functional
J = sp.simplify(J1 - J2)

print('Functional J computed successfully.')

# ==========================================
# 3. Plot Contour of J(b, c)
# ==========================================
J_func = sp.lambdify((b, c), J, "numpy")

# Notice b is now strictly negative!
b_vals = np.linspace(-0.99, -0.01, 100)
c_vals = np.linspace(-10, 10, 100)
B, C = np.meshgrid(b_vals, c_vals)
J_vals = J_func(B, C)

plt.figure(figsize=(8, 6))
contours = plt.contourf(B, C, J_vals, levels=50, cmap='viridis')
plt.colorbar(contours, label=r'$J(b, c)$')
plt.contour(B, C, J_vals, levels=50, colors='k', linewidths=0.5, alpha=0.5)
plt.xlabel(r'$b$')
plt.ylabel(r'$c$')
plt.title(r'Contour plot of Total Functional $J(b, c)$')
plt.show()

# ==========================================
# 4. Solve for c with respect to b
# ==========================================
J_D_c = sp.simplify(sp.diff(J, c))
eq2 = sp.Eq(J_D_c, 0)

# Isolate c as a function of b
c_respect_b = sp.solve(eq2, c)[0]
b_equation = sp.simplify(J.subs(c, c_respect_b))

# ==========================================
# 5. Optimize for b numerically and plot J(b)
# ==========================================
# Convert the simplified J(b) into a numpy function
b_func_num = sp.lambdify(b, b_equation, modules=['numpy'])

# Plot J(b) to visually confirm the local minimum in the negative domain
b_plot_vals = np.linspace(-0.99, -0.01, 200)
y_vals = b_func_num(b_plot_vals)

plt.figure(figsize=(8, 5))
plt.plot(b_plot_vals, y_vals, linewidth=2, color='blue')
plt.xlabel("Domain threshold $b$", fontsize=12)
plt.ylabel("$J(b)$", fontsize=12)
plt.title("Functional value vs. Domain threshold $b$", fontsize=13)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Optimize b specifically in the negative domain bounds
res = minimize_scalar(b_func_num, bounds=(-0.99, -0.01), method='bounded')
b_sol = res.x
c_sol = float(c_respect_b.subs(b, b_sol))

print(f"Optimal b = {b_sol:.4f}")
print(f"Optimal c = {c_sol:.4f}")

# ==========================================
# 6. Plot the 2D Spatial Solution
# ==========================================
u_t_expr = u_theta.subs({b: b_sol, c: c_sol})
u_t_func = sp.lambdify((x, y), u_t_expr, modules=['numpy'])

x_grid = np.linspace(0, 1, 100)
y_grid = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x_grid, y_grid)

# Evaluate the solution
U = u_t_func(X, Y)

# ReLU Logic: Set everything below the diagonal line to exactly 0.0
U = np.where((X + Y + b_sol) <= 0.0, 0.0, U)

plt.figure(figsize=(8, 6))
plt.contourf(X, Y, U, levels=50, cmap='magma')
plt.colorbar(label=r'$u(x,y)$')

# Plot the diagonal ReLU boundary line to highlight it
# The line is y = -x - b. We plot it over x in [0, -b]
x_line = np.linspace(0, -b_sol, 100)
y_line = -x_line - b_sol
plt.plot(x_line, y_line, color='cyan', linestyle='--', linewidth=2, label=f'ReLU Boundary (b={b_sol:.3f})')

plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.title("2D Piecewise Approximation Solution (Diagonal Boundary)", fontsize=13)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()

# ==========================================
# 7. Exact Solution & Error (Appended)
# ==========================================
print("Calculating exact symbolic Fourier integral (this takes a few seconds)...")
n, m = sp.symbols('n m', integer=True, positive=True)
F_nm_integrand = f * sp.sin(n * sp.pi * x) * sp.sin(m * sp.pi * y)
F_nm_expr = 4 * sp.integrate(F_nm_integrand, (x, 0, 1), (y, 0, 1))
A_nm_expr = F_nm_expr / (sp.pi**2 * (n**2 + m**2))
A_nm_func = sp.lambdify((n, m), A_nm_expr, modules=['numpy'])

U_exact = np.zeros_like(X)
N_terms = 30 
for n_val in range(1, N_terms + 1):
    for m_val in range(1, N_terms + 1):
        A = A_nm_func(n_val, m_val)
        if np.abs(A) > 1e-12:
            U_exact += A * np.sin(n_val * np.pi * X) * np.sin(m_val * np.pi * Y)

# Apply the same ReLU mask to the exact solution
U_exact_restricted = np.where((X + Y + b_sol) <= 0.0, 0.0, U_exact)

# Error plot
U_error = np.abs(U - U_exact_restricted)

plt.figure(figsize=(8, 6))
contours = plt.contourf(X, Y, U_error, levels=50, cmap='Reds')
plt.colorbar(contours, label=r'$|u_{approx} - u_{exact}|$')
plt.plot(x_line, y_line, color='blue', linestyle='--', linewidth=2, alpha=0.5, label=f'ReLU Boundary (b={b_sol:.3f})')
plt.xlabel("x", fontsize=12)
plt.ylabel("y", fontsize=12)
plt.title("Absolute Error Distribution", fontsize=13)
plt.legend(loc="upper right")
plt.tight_layout()
plt.show()