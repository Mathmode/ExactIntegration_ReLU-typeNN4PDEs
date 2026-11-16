

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import tensorflow as tf
import keras

scale = 0.25

@tf.custom_gradient
def my_heaviside(x):
    result = (tf.math.sign(x) + 1.0) / 2.0

    def grad(upstream):
        normalization = 1.0 / (scale * (2.0 * np.pi)**0.5)
        exponent = tf.math.exp(-(x**2) / (2.0 * (scale)))
        return normalization * exponent * upstream

    return result, grad

@tf.custom_gradient
def my_relu(x):
    result = tf.nn.relu(x)

    def grad(upstream):
        return my_heaviside(x) * upstream

    return result, grad

@tf.keras.utils.register_keras_serializable()
class MyReLU(keras.layers.Layer):
    def call(self, x):
        return my_relu(x)

class AnalyticalReLULayer(keras.layers.Layer):
    def __init__(self, init_alpha=0.0, init_b=0.0, init_c=1.0, use_custom_relu=True, **kwargs):
        super(AnalyticalReLULayer, self).__init__(**kwargs)
        self.init_alpha = init_alpha
        self.init_b = init_b
        self.init_c = init_c
        self.use_custom_relu = use_custom_relu

    def build(self, input_shape):
        self.alpha = self.add_weight(
            name="alpha",
            shape=(),
            initializer=keras.initializers.Constant(self.init_alpha),
            trainable=True
        )
        self.b = self.add_weight(
            name="b",
            shape=(),
            initializer=keras.initializers.Constant(self.init_b),
            trainable=True
        )
        self.c = self.add_weight(
            name="c",
            shape=(),
            initializer=keras.initializers.Constant(self.init_c),
            trainable=True
        )
        super(AnalyticalReLULayer, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch, 2) where inputs[:, 0] is x and inputs[:, 1] is y
        x = inputs[:, 0:1]
        y = inputs[:, 1:2]
        
        # Calculate: x * sin(alpha) + y * cos(alpha) + b
        inner_term = x * tf.sin(self.alpha) + y * tf.cos(self.alpha) + self.b
        
        # Apply c * my_relu(...) using the surrogate gradient
        relu_out = my_relu(inner_term) if self.use_custom_relu else tf.nn.relu(inner_term)
        return self.c * relu_out

    def get_config(self):
        config = super().get_config()
        config.update({
            "init_alpha": self.init_alpha,
            "init_b": self.init_b,
            "init_c": self.init_c,
            "use_custom_relu": self.use_custom_relu,
        })
        return config

class net_u(keras.Model):
    def __init__(self, init_alpha=0.0, init_b=0.0, init_c=1.0, use_custom_relu=True, **kwargs):
        super(net_u, self).__init__(**kwargs)
        self.analytical_layer = AnalyticalReLULayer(
            init_alpha=init_alpha, 
            init_b=init_b, 
            init_c=init_c,
            use_custom_relu=use_custom_relu
        )

    def call(self, inputs):
        x = inputs[:, 0:1]
        y = inputs[:, 1:2]
        
        # Compute the custom layer output
        out = self.analytical_layer(inputs)
        
        # Apply 2D Homogeneous Dirichlet boundary condition bubble function g(x, y)
        bc_mask = x * y * (x - 1.0) * (y - 1.0)
        return out * bc_mask