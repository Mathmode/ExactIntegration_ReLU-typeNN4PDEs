#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 17:44:28 2025

@author: sbahalouei
"""

import tensorflow as tf
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
keras.utils.set_random_seed(1234)

class net_u(tf.keras.Model):
    def __init__(self, depth, width, activation, **kwargs):
        (super(net_u, self).__init__)(**kwargs)
        self.depth = depth
        self.width = width
        self.activation = activation
        self.layers_list = list()

        self.layers_list.append(keras.layers.Dense(self.width, activation = self.activation, use_bias = True))
        self.layers_list.append(keras.layers.Dense(1, activation = None, use_bias = False))

    def call(self, inputs):

        # inputs is expected to be a tensor of shape (batch, 2) representing [x, y]
        x = inputs[:, 0:1]
        y = inputs[:, 1:2]
        
        # # normalization to [-1,1]
        # x_norm = 2.0 * x - 1.0
        # y_norm = 2.0 * y - 1.0
        
        # out = tf.concat([x_norm, y_norm], axis=1)
        out = inputs
        for layer in self.layers_list:
            out = layer(out)

        # Apply 2D Homogeneous Dirichlet Boundary Conditions 
        # This mask ensures the output is 0 at x=0,  y=0
        bc_mask = x * y * (x - 1) * (y - 1)
        return out * bc_mask
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "depth": self.depth,
            "width": self.width,
            "activation": self.activation,
        })
        return config
    