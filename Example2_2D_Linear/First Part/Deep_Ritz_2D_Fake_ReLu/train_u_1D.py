#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:06:08 2025

@author: sbahalouei
"""

import tensorflow as tf
import numpy as np
from My_integration import Lasserre_integration_dReLu_dReLu, Lasserre_integration_ReLu_ReLu_new
from Data import f_rhs

# from Symbolic import symbolic

@tf.function
def train_step(u_net, f_coeffs, f_degree, optimizer, num_neurons_u, domain_vertices, xy):

    with tf.GradientTape() as tape:
        out = loss_u(u_net, f_coeffs, f_degree, num_neurons_u, domain_vertices, xy)
    g = tape.gradient(out, u_net.trainable_variables)
    # for i, grad in enumerate(g):
    #     if grad is not None:
    #         print(f"Layer {i} gradient max: {np.max(grad.numpy())}")
    #     else:
    #         print(f"Layer {i} sgradient is NONE")
    optimizer.apply(g, u_net.trainable_variables)
    return out
    
@tf.function
def lr_schedule(epoch, initial_learning_rate, final_learning_rate, num_epochs):
    decay_rate = (final_learning_rate / initial_learning_rate) ** (1 / num_epochs)
    current_learning_rate = initial_learning_rate * (decay_rate ** epoch)
    return current_learning_rate

# @tf.function
# def interior_regularization(u_net, num_neurons_u, lambda_reg=0.001):
#     if tf.equal(lambda_reg, 0.0):
#         return tf.constant(0.0, dtype=tf.float64)
    
#     # In Fake_ReLu, u_net uses AnalyticalReLULayer with w (2, width) and b (width,)
#     layer = u_net.layers[0]
#     W = tf.cast(layer.w, tf.float64)  # shape (2, width)
#     b = tf.cast(layer.b, tf.float64)  # shape (width,)
    
#     w0 = W[0, :]
#     w1 = W[1, :]
    
#     norm_w = tf.sqrt(tf.square(w0) + tf.square(w1) + 1e-12)
    
#     # Distance to center
#     d_center = tf.abs(w0 * 0.5 + w1 * 0.5 + b) / norm_w
#     center_margin_penalty = tf.square(tf.nn.relu(d_center - 0.50))
    
#     reg_loss = tf.reduce_mean(5.0 * center_margin_penalty)
#     return tf.cast(lambda_reg, tf.float64) * reg_loss

@tf.function
def interior_regularization(u_net, num_neurons_u, lambda_reg=0.1):
    if tf.equal(lambda_reg, 0.0):
        return tf.constant(0.0, dtype=tf.float64)
    
    weights_1 = u_net.layers[0].weights[0]  # shape (2, num_neurons_u)
    biases_1 = u_net.layers[0].weights[1]   # shape (num_neurons_u,)
    
    w0 = tf.cast(weights_1[0, :], tf.float64)
    w1 = tf.cast(weights_1[1, :], tf.float64)
    b = tf.cast(biases_1, tf.float64)
    
    norm_w = tf.sqrt(tf.square(w0) + tf.square(w1) + 1e-12)
    
    # 1. Calculate the pre-activation Z at the 4 corners of the [0, 1]x[0, 1] domain
    v1 = b
    v2 = w0 + b
    v3 = w1 + b
    v4 = w0 + w1 + b
    
    # Stack to shape (4, num_neurons)
    corners_Z = tf.stack([v1, v2, v3, v4], axis=0)
    
    # Normalize the distances so the penalty is geometric and scale-invariant
    corners_Z = corners_Z / norm_w
    
    # 2. Get the maximum and minimum values at the corners for each neuron
    max_Z = tf.reduce_max(corners_Z, axis=0)
    min_Z = tf.reduce_min(corners_Z, axis=0)
    
    # We want the line to intersect the domain. 
    # An intersection means max_Z > 0 (not dead) AND min_Z < 0 (not purely linear).
    # We add a small margin (e.g. 0.05) to ensure it stays strictly inside.
    margin = 0.05
    
    # If max_Z < margin (domain is entirely dead), penalty pushes it up.
    penalty_dead = tf.square(tf.nn.relu(-max_Z + margin))
    
    # If min_Z > -margin (domain is entirely linear), penalty pushes it down.
    penalty_linear = tf.square(tf.nn.relu(min_Z + margin))
    
    # 3. Sum the penalties. If the line intersects the domain safely, this is exactly 0.0!
    reg_loss = tf.reduce_mean(penalty_dead + penalty_linear)
    
    return tf.cast(lambda_reg, tf.float64) * reg_loss

@tf.function
def loss_u(u_net, f_coeffs, f_degree, num_neurons_u, domain_vertices, xy):
    du_du = Lasserre_integration_dReLu_dReLu(u_net, num_neurons_u, domain_vertices, xy)
    u_f = Lasserre_integration_ReLu_ReLu_new(u_net, f_coeffs, f_degree, num_neurons_u, domain_vertices, xy)
    
    reg = interior_regularization(u_net, num_neurons_u, lambda_reg=0.1)
    
    out = du_du/2.0 - u_f #+ tf.cast(reg, du_du.dtype)
    return out
    
