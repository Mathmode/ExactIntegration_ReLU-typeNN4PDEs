


import tensorflow as tf
import numpy as np
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

class AnalyticalReLULayer(keras.layers.Layer):
    def __init__(self, init_alpha=0.0, init_b=0.0, init_c=1.0, **kwargs):
        super(AnalyticalReLULayer, self).__init__(**kwargs)
        # Store initial values
        self.init_alpha = init_alpha
        self.init_b = init_b
        self.init_c = init_c

    def build(self, input_shape):
        # Create weights for alpha, b, and c as explicit scalar trainable variables
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
        
        # Apply c * ReLU(...)
        return self.c * tf.nn.relu(inner_term)

    def get_config(self):
        config = super().get_config()
        config.update({
            "init_alpha": self.init_alpha,
            "init_b": self.init_b,
            "init_c": self.init_c,
        })
        return config
    
    
    
class net_u(tf.keras.Model):
    def __init__(self, init_alpha, init_b, init_c, **kwargs):
        super(net_u, self).__init__(**kwargs)
        # Use our custom layer instead of self.layers_list
        self.analytical_layer = AnalyticalReLULayer(init_alpha, init_b, init_c)

    def call(self, inputs):
        x = inputs[:, 0:1]
        y = inputs[:, 1:2]
        
        # Compute the custom shape
        out = self.analytical_layer(inputs)
        
        # Apply your boundary condition mask
        bc_mask = x * y * (x - 1) * (y - 1)
        return out * bc_mask