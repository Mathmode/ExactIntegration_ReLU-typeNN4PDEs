#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 15:06:08 2025

@author: sbahalouei
"""

import tensorflow as tf
import numpy as np
from My_integration import my_integration_FB
from Data import f_rhs


@tf.function
def train_step_FB(u_net, f_coeffs, f_degree, optimizer, num_neurons_u, 
                  domain_vertices, batch_size, epoch, pts, wts):

    loss_total = tf.constant(0.0, dtype=tf.float64)
        
    with tf.GradientTape(persistent=True) as tape:
        # Pass the "tape" object directly into your loss function!
        out = loss_FB(u_net, f_coeffs, f_degree, num_neurons_u, domain_vertices, epoch, pts, wts)
        loss_total += tf.cast(out, tf.float64)
        
    vars_to_train = u_net.trainable_variables
    g = tape.gradient(out, vars_to_train)
    
    # Clean up the persistent tape manually to avoid memory leaks
    del tape 
    
    # FIX: Zip the gradients and variables together
    # It is also highly recommended to filter out any None gradients just in case
    grads_and_vars = [(grad, var) for grad, var in zip(g, vars_to_train) if grad is not None]
    
    if len(grads_and_vars) > 0:
        optimizer.apply_gradients(grads_and_vars)

    return loss_total


@tf.function
def lr_schedule(epoch, initial_learning_rate, final_learning_rate, num_epochs):
    # Calculate the total drop needed over the entire training duration
    total_reduction = initial_learning_rate - final_learning_rate
    
    # Calculate how much to drop per single epoch
    step_size = total_reduction / tf.cast(num_epochs, tf.float64)
    
    # Current LR = Start - (Drop per epoch * current epoch)
    current_learning_rate = initial_learning_rate - (step_size * tf.cast(epoch, tf.float64))
    
    # Safety: Ensure the learning rate doesn't drop below the final_learning_rate 
    # (useful if epoch > num_epochs)
    return tf.maximum(current_learning_rate, final_learning_rate)




@tf.function
def loss_FB(u_net, f_coeffs, f_degree, num_neurons_u, domain_vertices,  epoch, pts, wts):
    # Convert numpy arrays to tensors to preserve graph tracking
    pts_tf = tf.Variable(pts, dtype=tf.float64)
    wts_tf = tf.cast(wts, dtype=tf.float64)

    # Pass the tracking tensors into your integration engine
    du_du, u_f = my_integration_FB(u_net, num_neurons_u, domain_vertices, epoch, pts, wts, f_coeffs, f_degree)
    
    target_dtype = tf.float64 
    du_du = tf.cast(du_du, target_dtype)
    u_f = tf.cast(u_f, target_dtype)
    
    out = du_du / tf.cast(2.0, target_dtype) - u_f
    
    return out


