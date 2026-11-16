#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 23 12:25:56 2026

@author: sbahalouei
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def clip_polygon_with_line_diff(vertices, line_coeffs):
    """
    Differentiable clipping of a polygon against a single line.
    """
    # Ensure vertices are closed
    v_curr = vertices
    v_prev = tf.roll(vertices, shift=1, axis=0)
    

    # Signed distances: positive = inside
    d_prev = tf.reduce_sum(v_prev * line_coeffs[:2], axis=1) + line_coeffs[2]
    print(d_prev)
    d_curr = tf.reduce_sum(v_curr * line_coeffs[:2], axis=1) + line_coeffs[2]

    # Intersection points
    t = -d_prev / (d_curr - d_prev + 1e-8)
    intersections = v_prev + tf.expand_dims(t, axis=-1) * (v_curr - v_prev)

    curr_inside = d_curr >= 0
    prev_inside = d_prev >= 0

    output_points = []
    num_vertices = vertices.shape[0]

    for i in range(num_vertices):
        if curr_inside[i] != prev_inside[i]:
            output_points.append(intersections[i])
        if curr_inside[i]:
            output_points.append(v_curr[i])

    if len(output_points) == 0:
        return tf.zeros((0, 2), dtype=vertices.dtype)

    return tf.stack(output_points)

# --- Define Data ---
# A simple square
poly = tf.constant([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]], dtype=tf.float32)
# Clipping line: x = 0.5 (which is ax + by + c = 0 -> (a, b, c))
line_coeffs = tf.constant([1.0, -1.0, -0.5], dtype=tf.float32)

# Execute clipping
new_poly = clip_polygon_with_line_diff(poly, line_coeffs)
print("Clipped Vertices:\n", new_poly.numpy())

# with tf.GradientTape() as tape:
#     tape.watch(poly)  # <--- Add this line!
#     new_poly = clip_polygon_with_line_diff(poly, line_coeffs)
#     loss = tf.reduce_sum(new_poly[0])

# # Now grads will be a Tensor, not None
# grads = tape.gradient(loss, poly)


# if grads is not None:
#     print("\nGradient w.r.t original vertices:\n", grads.numpy())
# else:
#     print("\nGradient is None. Ensure 'poly' is a Variable or watched by the tape.")
    
    
    
    
    
    
    
    
    
    
    
    
    

# --- Plotting Code ---

def get_closed_coords(points):
    """Helper to close the polygon loop for plotting."""
    pts = points.numpy()
    return np.vstack([pts, pts[0]])

plt.figure(figsize=(8, 8))

# 1. Plot the Original Polygon (Dashed outline)
orig_coords = get_closed_coords(poly)
plt.plot(orig_coords[:, 0], orig_coords[:, 1], 'k--', label='Original Polygon', alpha=0.5)

# 2. Plot the Clipping Line
# ax + by + c = 0
a, b, c = line_coeffs.numpy()
x_vals = np.array([-1, 3])
if b != 0:
    y_vals = (-a * x_vals - c) / b
    plt.plot(x_vals, y_vals, 'blue', label=f'Line: {a}x + {b}y + {c} = 0')
else:
    # Handle vertical line case (like x = 0.5)
    x_intercept = -c / a
    plt.axvline(x_intercept, color='blue', label=f'Line: x = {x_intercept}')

# 3. Plot and Fill the Clipped Polygon
if new_poly.shape[0] > 0:
    clipped_coords = get_closed_coords(new_poly)
    plt.fill(clipped_coords[:, 0], clipped_coords[:, 1], 'red', alpha=0.3, label='Clipped Area')
    plt.plot(clipped_coords[:, 0], clipped_coords[:, 1], 'red', linewidth=2)

# Formatting
plt.xlim(-0.5, 2.5)
plt.ylim(-0.5, 2.5)
plt.axhline(0, color='black', lw=1)
plt.axvline(0, color='black', lw=1)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend()
plt.title('Differentiable Polygon Clipping')
plt.xlabel('X')
plt.ylabel('Y')

plt.show()