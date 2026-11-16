# import os
# os.environ["KERAS_BACKEND"] = "tensorflow"
# import numpy as np
# import tensorflow as tf
# import keras

# scale = 1

# @tf.custom_gradient
# def my_heaviside(x):
#     result = (tf.math.sign(x) + 1.0) / 2.0

#     def grad(upstream):
#         normalization = 1.0 / (scale * (2.0 * np.pi)**0.5)
#         exponent = tf.math.exp(-(x**2) / (2.0 * (scale)))
#         return normalization * exponent * upstream

#     return result, grad

# @tf.custom_gradient
# def my_relu(x):
#     result = tf.nn.relu(x)

#     def grad(upstream):
#         return my_heaviside(x) * upstream

#     return result, grad

# @tf.keras.utils.register_keras_serializable()
# class MyReLU(keras.layers.Layer):
#     def call(self, x):
#         return my_relu(x)

# @tf.keras.utils.register_keras_serializable()
# class AnalyticalReLULayer(keras.layers.Layer):
#     def __init__(self, init_alpha=0.0, init_b=0.0, init_c=1.0, use_custom_relu=True, **kwargs):
#         super(AnalyticalReLULayer, self).__init__(**kwargs)
#         self.init_alpha = init_alpha
#         self.init_b = init_b
#         self.init_c = init_c
#         self.use_custom_relu = use_custom_relu

#     def build(self, input_shape):
#         self.alpha = self.add_weight(
#             name="alpha",
#             shape=(),
#             initializer=keras.initializers.Constant(self.init_alpha),
#             trainable=True
#         )
#         self.b = self.add_weight(
#             name="b",
#             shape=(),
#             initializer=keras.initializers.Constant(self.init_b),
#             trainable=True
#         )
#         self.c = self.add_weight(
#             name="c",
#             shape=(),
#             initializer=keras.initializers.Constant(self.init_c),
#             trainable=True
#         )
#         super(AnalyticalReLULayer, self).build(input_shape)

#     def call(self, inputs):
#         # inputs shape: (batch, 2) where inputs[:, 0] is x and inputs[:, 1] is y
#         x = inputs[:, 0:1]
#         y = inputs[:, 1:2]
        
#         # Calculate: x * sin(alpha) + y * cos(alpha) + b
#         inner_term = x * tf.sin(self.alpha) + y * tf.cos(self.alpha) + self.b
        
#         # Apply c * my_relu(...) using the surrogate gradient
#         relu_out = my_relu(inner_term) if self.use_custom_relu else tf.nn.relu(inner_term)
#         return self.c * relu_out

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "init_alpha": self.init_alpha,
#             "init_b": self.init_b,
#             "init_c": self.init_c,
#             "use_custom_relu": self.use_custom_relu,
#         })
#         return config

# @tf.keras.utils.register_keras_serializable()
# class net_u(keras.Model):
#     def __init__(self, init_alpha=0.0, init_b=0.0, init_c=1.0, use_custom_relu=True, **kwargs):
#         super(net_u, self).__init__(**kwargs)
#         self.init_alpha = init_alpha
#         self.init_b = init_b
#         self.init_c = init_c
#         self.use_custom_relu = use_custom_relu
        
#         self.analytical_layer = AnalyticalReLULayer(
#             init_alpha=init_alpha, 
#             init_b=init_b, 
#             init_c=init_c,
#             use_custom_relu=use_custom_relu
#         )

#     def call(self, inputs):
#         x = inputs[:, 0:1]
#         y = inputs[:, 1:2]
        
#         # Compute the custom layer output
#         out = self.analytical_layer(inputs)
        
#         # Apply 2D Homogeneous Dirichlet boundary condition bubble function g(x, y)
#         bc_mask = x * y * (x - 1.0) * (y - 1.0)
#         return out * bc_mask

#     def get_config(self):
#         config = super().get_config()
#         config.update({
#             "init_alpha": self.init_alpha,
#             "init_b": self.init_b,
#             "init_c": self.init_c,
#             "use_custom_relu": self.use_custom_relu,
#         })
#         return config



import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import tensorflow as tf
import keras

scale = 3

@tf.custom_gradient
def my_heaviside(x):
    result = (tf.math.sign(x) + 1.0) / 2.0

    def grad(upstream):
        normalization = 1.0 / (scale * (2.0 * np.pi)**0.5)
        exponent = tf.math.exp(-(x**2) / (2.0 * scale))
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


@tf.keras.utils.register_keras_serializable()
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


@tf.keras.utils.register_keras_serializable()
class AnalyticalReLULayer(keras.layers.Layer):
    def __init__(self, width, min_deg=5.0, max_deg=85.0, use_custom_relu=True, **kwargs):
        super(AnalyticalReLULayer, self).__init__(**kwargs)
        self.width = width
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_custom_relu = use_custom_relu

    def build(self, input_shape):
        # Initialize W using the FanInitializer
        self.w = self.add_weight(
            name="w",
            shape=(2, self.width),
            initializer=FanInitializer(width=self.width, min_deg=self.min_deg, max_deg=self.max_deg),
            trainable=True
        )
        # Initialize biases to exactly zero
        self.b = self.add_weight(
            name="b",
            shape=(self.width,),
            initializer=keras.initializers.Zeros(),
            trainable=True
        )
        # Initialize readout weights (c) for the final sum
        self.c = self.add_weight(
            name="c",
            shape=(self.width, 1),
            initializer=keras.initializers.GlorotUniform(),
            trainable=True
        )
        super(AnalyticalReLULayer, self).build(input_shape)

    def call(self, inputs):
        # inputs shape: (batch, 2)
        # 1. Affine transformation: (batch, 2) @ (2, width) + (width,) -> (batch, width)
        inner_term = tf.matmul(inputs, self.w) + self.b
        
        # 2. Apply surrogate custom ReLU activation
        relu_out = my_relu(inner_term) if self.use_custom_relu else tf.nn.relu(inner_term)
        
        # 3. Readout multiplication: (batch, width) @ (width, 1) -> (batch, 1)
        return tf.matmul(relu_out, self.c)

    def get_config(self):
        config = super().get_config()
        config.update({
            "width": self.width,
            "min_deg": self.min_deg,
            "max_deg": self.max_deg,
            "use_custom_relu": self.use_custom_relu,
        })
        return config


@tf.keras.utils.register_keras_serializable()
class net_u(keras.Model):
    def __init__(self, depth=1, width=10, min_deg=5.0, max_deg=85.0, use_custom_relu=True, **kwargs):
        super(net_u, self).__init__(**kwargs)
        self.depth = depth  # kept for compatibility with your generic load_model calls
        self.width = width
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_custom_relu = use_custom_relu
        
        self.analytical_layer = AnalyticalReLULayer(
            width=width, 
            min_deg=min_deg, 
            max_deg=max_deg,
            use_custom_relu=use_custom_relu
        )

    def call(self, inputs):
        x = inputs[:, 0:1]
        y = inputs[:, 1:2]
        
        # Compute the custom wide layer output
        out = self.analytical_layer(inputs)
        
        # Apply 2D Homogeneous Dirichlet boundary condition bubble function g(x, y)
        bc_mask = x * y * (x - 1.0) * (y - 1.0)
        return out * bc_mask

    def get_config(self):
        config = super().get_config()
        config.update({
            "depth": self.depth,
            "width": self.width,
            "min_deg": self.min_deg,
            "max_deg": self.max_deg,
            "use_custom_relu": self.use_custom_relu,
        })
        return config
