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

    loss_total = 0
        
    with tf.GradientTape() as tape:
        out = loss_FB(u_net, f_coeffs, f_degree, num_neurons_u, domain_vertices, epoch, pts, wts)
        loss_total += float(out)
        
    vars_to_train = u_net.trainable_variables
    g = tape.gradient(out, vars_to_train)
    del tape 
    
    # FIX: Zip the gradients and variables together
    # It is also highly recommended to filter out any None gradients just in case
    grads_and_vars = [(grad, var) for grad, var in zip(g, vars_to_train) if grad is not None]
    
    if len(grads_and_vars) > 0:
        optimizer.apply_gradients(grads_and_vars)
    return loss_total



@tf.function
def lr_schedule(epoch, initial_learning_rate, final_learning_rate, num_epochs):
    decay_rate = (final_learning_rate / initial_learning_rate) ** (1 / num_epochs)
    current_learning_rate = initial_learning_rate * (decay_rate ** epoch)
    return current_learning_rate


@tf.function
def loss_FB(u_net, f_coeffs, f_degree, num_neurons_u, domain_vertices,  epoch, pts, wts):
        

    du_du, u_f = my_integration_FB(u_net, num_neurons_u, domain_vertices, epoch, pts, wts, f_coeffs, f_degree)
    
    target_dtype = tf.float64 

    du_du = tf.cast(du_du, target_dtype)
    u_f = tf.cast(u_f, target_dtype)
    out = du_du/tf.cast(2.0, target_dtype) - tf.cast(u_f, target_dtype)
    
    
    return out

