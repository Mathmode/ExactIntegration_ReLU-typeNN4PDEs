# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# Created on Wed Jan 22 17:44:28 2025

# @author: sbahalouei
# """

# import tensorflow as tf
# import numpy as np
# import os
# os.environ["KERAS_BACKEND"] = "tensorflow"
# import keras
# keras.utils.set_random_seed(1234)


# class DistributedKernelInitializer(keras.initializers.Initializer):
#     """
#     Initializes weights (normal vectors) for lines distributed across [0, 1]^2
#     with diverse orientations spanning from min_deg to max_deg.
#     """
#     def __init__(self, width, min_deg=15.0, max_deg=165.0):
#         self.width = width
#         self.min_deg = min_deg
#         self.max_deg = max_deg

#     def __call__(self, shape, dtype=None):
#         angles = np.linspace(np.deg2rad(self.min_deg), np.deg2rad(self.max_deg), self.width)
#         W = np.zeros(shape, dtype=np.float32)
#         W[0, :] = np.cos(angles)  # weights for x
#         W[1, :] = np.sin(angles)  # weights for y
#         return tf.convert_to_tensor(W, dtype=dtype)

#     def get_config(self):
#         return {
#             "width": self.width,
#             "min_deg": self.min_deg,
#             "max_deg": self.max_deg
#         }


# class DistributedBiasInitializer(keras.initializers.Initializer):
#     """
#     Initializes biases so that lines pass through distributed anchor points
#     inside [0.25, 0.75]^2, ensuring every line cuts deeply through the interior of [0, 1]^2.
#     """
#     def __init__(self, width, min_deg=15.0, max_deg=165.0):
#         self.width = width
#         self.min_deg = min_deg
#         self.max_deg = max_deg

#     def __call__(self, shape, dtype=None):
#         angles = np.linspace(np.deg2rad(self.min_deg), np.deg2rad(self.max_deg), self.width)
#         xs = np.linspace(0.25, 0.75, self.width)
#         ys = np.linspace(0.25, 0.75, self.width)
#         ys_perm = np.array([ys[i] if i % 2 == 0 else ys[self.width - 1 - i] for i in range(self.width)], dtype=np.float32)
        
#         w0 = np.cos(angles)
#         w1 = np.sin(angles)
#         b = -(w0 * xs + w1 * ys_perm)
#         return tf.convert_to_tensor(b, dtype=dtype)

#     def get_config(self):
#         return {
#             "width": self.width,
#             "min_deg": self.min_deg,
#             "max_deg": self.max_deg
#         }


# class FanInitializer(keras.initializers.Initializer):
#     """
#     Initializes weights so that every activation boundary line
#     starts at (0, 0) and enters the unit square (0, 1)^2.
#     """
#     def __init__(self, width, min_deg=5.0, max_deg=85.0):
#         self.width = width
#         self.min_deg = min_deg
#         self.max_deg = max_deg

#     def __call__(self, shape, dtype=None):
#         angles = np.linspace(np.deg2rad(self.min_deg), np.deg2rad(self.max_deg), self.width)
#         W = np.zeros(shape, dtype=np.float32)
#         W[0, :] = -np.sin(angles)
#         W[1, :] =  np.cos(angles)
#         return tf.convert_to_tensor(W, dtype=dtype)

#     def get_config(self):
#         return {
#             "width": self.width,
#             "min_deg": self.min_deg,
#             "max_deg": self.max_deg
#         }
    
    

# class net_u(tf.keras.Model):
#     def __init__(self, depth, width, activation, **kwargs):
#         (super(net_u, self).__init__)(**kwargs)
#         self.depth = depth
#         self.width = width
#         self.activation = activation
#         self.layers_list = list()

#         # Distributed line initializer crossing through the interior of [0, 1]^2
#         weight_init = DistributedKernelInitializer(width=self.width, min_deg=15.0, max_deg=165.0)
#         bias_init = DistributedBiasInitializer(width=self.width, min_deg=15.0, max_deg=165.0)

#         # Hidden layer(s)
#         self.layers_list.append(
#             keras.layers.Dense(
#                 self.width,
#                 activation=self.activation,
#                 kernel_initializer=weight_init,
#                 bias_initializer=bias_init,
#                 use_bias=True
#             )
#         )

#         # Output layer (linear readout, no bias)
#         self.layers_list.append(
#             keras.layers.Dense(
#                 1,
#                 activation=None,
#                 kernel_initializer=tf.keras.initializers.GlorotUniform(),
#                 use_bias=False
#             )
#         )

#     def call(self, inputs):

#         # inputs is expected to be a tensor of shape (batch, 2) representing [x, y]
#         x = inputs[:, 0:1]
#         y = inputs[:, 1:2]
#         out = inputs
#         for layer in self.layers_list:
#             out = layer(out)
#         bc_mask = x * y * (x - 1) * (y - 1)
#         return out * bc_mask
    
#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "depth": self.depth,
#             "width": self.width,
#             "activation": self.activation,
#         })
#         return config
    

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


class FanInitializer(keras.initializers.Initializer):
    """
    Initializes weights so that every activation boundary line
    starts at (0, 0) and enters the unit square (0, 1)^2.
    """
    def __init__(self, width, min_deg=5.0, max_deg=85.0):
        self.width = width
        self.min_deg = min_deg
        self.max_deg = max_deg

    def __call__(self, shape, dtype=None):
        # shape is (input_dim, width) -> (2, self.width)
        angles = np.linspace(np.deg2rad(self.min_deg), np.deg2rad(self.max_deg), self.width)
        W = np.zeros(shape, dtype=np.float32)
        # Normal vectors for lines y = tan(theta) * x: -sin(theta) * x + cos(theta) * y = 0
        W[0, :] = -np.sin(angles)  # weights for x
        W[1, :] =  np.cos(angles)  # weights for y
        return tf.convert_to_tensor(W, dtype=dtype)

    def get_config(self):
        return {
            "width": self.width,
            "min_deg": self.min_deg,
            "max_deg": self.max_deg
        }
    
    

class net_u(tf.keras.Model):
    def __init__(self, depth, width, activation, **kwargs):
        (super(net_u, self).__init__)(**kwargs)
        self.depth = depth
        self.width = width
        self.activation = activation
        self.layers_list = list()

        # weight_init = tf.keras.initializers.RandomUniform(minval=0.0, maxval=1.0)
        # bias_init = tf.keras.initializers.RandomUniform(minval=0.0, maxval=0.0)
        
        # # self.layers_list.append(
        # #     keras.layers.Dense(
        # #         self.width, 
        # #         activation=self.activation, 
        # #         kernel_initializer=weight_init,
        # #         bias_initializer=bias_init
        # #     )
        # # )
        
        # self.layers_list.append(keras.layers.Dense(self.width, activation = self.activation, kernel_initializer=weight_init,
        #          bias_initializer=bias_init, use_bias = True))
        # self.layers_list.append(keras.layers.Dense(1, activation = None, kernel_initializer=weight_init,
        #          bias_initializer=bias_init, use_bias = False))
        
        
        
        
        
        # Weight initializer fan-out from (0, 0) through the interior of [0, 1]^2
        weight_init = FanInitializer(width=self.width, min_deg=5.0, max_deg=85.0)
        # Bias set strictly to zero to force all lines through (0, 0)
        bias_init = tf.keras.initializers.Zeros()

        # Hidden layer(s)
        self.layers_list.append(
            keras.layers.Dense(
                self.width,
                activation=self.activation,
                kernel_initializer=weight_init,
                bias_initializer=bias_init,
                use_bias=True
            )
        )

        # Output layer (linear readout, no bias)
        self.layers_list.append(
            keras.layers.Dense(
                1,
                activation=None,
                kernel_initializer=tf.keras.initializers.GlorotUniform(),
                use_bias=False
            )
        )

    def call(self, inputs):

        # inputs is expected to be a tensor of shape (batch, 2) representing [x, y]
        x = inputs[:, 0:1]
        y = inputs[:, 1:2]
        out = inputs
        for layer in self.layers_list:
            out = layer(out)
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
    
