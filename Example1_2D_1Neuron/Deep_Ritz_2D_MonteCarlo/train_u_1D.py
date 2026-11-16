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
    vars_to_train = u_net.trainable_variables
        # vars_to_train = [var for var in all_trainable_vars if var is not kernel_to_freeze]
    g = tape.gradient(out, vars_to_train)
    optimizer.apply_gradients(zip(g, vars_to_train))
    # g = tape.gradient(out, u_net.trainable_variables)
    # optimizer.apply(g, u_net.trainable_variables)
    return out
    
@tf.function
def lr_schedule(epoch, initial_learning_rate, final_learning_rate, num_epochs):
    decay_rate = (final_learning_rate / initial_learning_rate) ** (1 / num_epochs)
    current_learning_rate = initial_learning_rate * (decay_rate ** epoch)
    return current_learning_rate

@tf.function
def loss_u(u_net, f_coeffs, f_degree, num_neurons_u, domain_vertices, xy):
        

    du_du = Lasserre_integration_dReLu_dReLu(u_net, num_neurons_u, domain_vertices, xy)
    u_f = Lasserre_integration_ReLu_ReLu_new(u_net, f_coeffs, f_degree, num_neurons_u, domain_vertices, xy)
    
    out = du_du/2 - u_f 
    
    
    return out
    
