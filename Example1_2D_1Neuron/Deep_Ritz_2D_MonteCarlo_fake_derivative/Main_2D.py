import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

from NN_Architecture_2D import net_u
from train_u_1D import train_step, lr_schedule
from Data import exact_u, f_rhs, exact_dx, exact_dy
from Plot import plot_model_2D
import keras
keras.utils.set_random_seed(1234)
import time as time
from numpy.polynomial.polynomial import Polynomial
import scipy.io
from numpy.polynomial.polynomial import Polynomial


import warnings
import numpy as np

# This will stop the console flood from Shapely/GEOS
warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")
# Some versions of Shapely also trigger through numpy
warnings.filterwarnings("ignore", message="invalid value encountered in intersects")
warnings.filterwarnings("ignore", message="invalid value encountered in intersection")



scale = 0.25

@tf.custom_gradient
def my_heaviside(x):
    result = (tf.math.sign(x) + 1) / 2

    def grad(upstream):
        normalization = 1.0 / (scale * (2.0 * np.pi)**0.5)
        exponent = tf.math.exp(-(x)**2 / (2.0 * (scale)**2))
        return normalization * exponent * upstream

    return result, grad


@tf.custom_gradient
def my_relu(x):
    result = tf.nn.relu(x)

    def grad(upstream):
        return my_heaviside(x) * upstream

    return result, grad


# ✅ SERIALIZABLE WRAPPER (THIS IS THE KEY FIX)
@tf.keras.utils.register_keras_serializable()
class MyReLU(tf.keras.layers.Layer):
    def call(self, x):
        return my_relu(x)
    

num_epochs = 500
num_neurons_u = 1
num_layers = 1
initial_learning_rate = 1.e-1
final_learning_rate = 1.e-2
n_MC = 1000 # in each dimention


domain_vertices = [(0, 0), (1,0), (1, 1), (0, 1)]

## ---------
## ---------
## ---------
# Step 1: approximate the right hand side function f by a polynomial of degree N:
## ---------
## ---------
## ---------

import numpy as np
from numpy.polynomial import Chebyshev, Polynomial

def fit_2d_poly_stable(x_coords_1d, y_coords_1d, z_data_2d, deg):
    """
    Separable 1D stable fit to avoid 2D domain bugs and Runge's Phenomenon.
    """
    num_y = len(y_coords_1d)
    x_deg_p1 = deg[0] + 1
    y_deg_p1 = deg[1] + 1
    
    # Step 1: Fit each row (fixed y) along the x-dimension
    intermediate_coeffs = np.zeros((num_y, x_deg_p1))
    for j in range(num_y):
        row_data = z_data_2d[j, :]
        # 1D Chebyshev fit is robust against domain errors
        c_fit = Chebyshev.fit(x_coords_1d, row_data, deg[0], domain=[0, 1])
        # Convert to power basis (x^i)
        p_fit = c_fit.convert(kind=Polynomial)
        c_x = p_fit.coef
        intermediate_coeffs[j, :min(len(c_x), x_deg_p1)] = c_x[:x_deg_p1]
    
    # Step 2: Fit each x-coefficient across the y-dimension
    final_coeffs = np.zeros((x_deg_p1, y_deg_p1))
    for i in range(x_deg_p1):
        col_data = intermediate_coeffs[:, i]
        c_fit = Chebyshev.fit(y_coords_1d, col_data, deg[1], domain=[0, 1])
        # Convert to power basis (y^j)
        p_fit = c_fit.convert(kind=Polynomial)
        c_y = p_fit.coef
        final_coeffs[i, :min(len(c_y), y_deg_p1)] = c_y[:y_deg_p1]
        
    return np.transpose(final_coeffs)

# Main_2D.py
x_coords = np.linspace(0, 1, 100) # Use odd number for perfect center
y_coords = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
Z = f_rhs(X, Y) 

# Pass the 1D arrays to the stable fitting function
f_degree = [10, 10]
f_coeffs = fit_2d_poly_stable(x_coords, y_coords, Z, f_degree)


# Verify Symmetry
from numpy.polynomial.polynomial import polyval2d
Z_poly = polyval2d(X, Y, f_coeffs)
sym_err = np.abs(Z_poly[25, 5] - Z_poly[25, 45])
print(f"New Symmetry Error: {sym_err}") 





u_net = net_u(init_alpha=np.pi/2, init_b=-0.7, init_c=1.0)
u = u_net(tf.random.uniform(shape=[1,2] , minval=0., maxval=1.))


#===================================================
#===================================================
#===================================================

u_net.save('Saved_Models/u_net_initialize.keras')
u_net.summary()

#===================================================
#===================================================
#===================================================

optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

x_eval = tf.linspace(0.0, 1.0, 1000)[:, None]  # 100 evenly spaced points
x_eval_tf = tf.expand_dims(tf.linspace(0.0, 1.0, 1000), axis=1)


# Start the timer
start_time = time.time()
## training u_net
Loss = []
k = 0



weights_a1 = []
weights_b = []
weights_c = []

Error_a1 = []
Error_b = []
Error_c = []


# 1. Access the parameters directly from your custom layer
current_alpha = u_net.analytical_layer.alpha.numpy()
current_b     = u_net.analytical_layer.b.numpy()
current_c     = u_net.analytical_layer.c.numpy()

# 2. Append the target values to your tracking lists
weights_a1.append((current_alpha)) # Equivalent to your old W[0, 0]

alpha_o = 0.7854
b_o = -0.5988
c_o = 7.4323

# Log b and c directly
weights_b.append(current_b)
weights_c.append(current_c) 
weights_a1.append(current_alpha)

for epoch in range(num_epochs):
    
    #========= MC points =========
    x = tf.random.uniform((n_MC,), 0.0, 1.0)
    X, Y = tf.meshgrid(x, x, indexing='ij')
    xy = tf.stack([tf.reshape(X, [-1]), tf.reshape(Y, [-1])], axis=1)
    #=============================
    
    new_learning_rate = lr_schedule(epoch, initial_learning_rate, \
                                    final_learning_rate, num_epochs)
    optimizer.learning_rate.assign(new_learning_rate)
    
    Loss += [train_step(u_net, f_coeffs, f_degree, optimizer, num_neurons_u, 
                        domain_vertices, xy)]
    k += 1
    print('===========================================')
    print('=============', 'Epoch:', epoch, '/', num_epochs,'=============')
    print('Loss_training:', float(Loss[-1]))
    
    
    current_alpha = u_net.analytical_layer.alpha.numpy()
    current_b     = u_net.analytical_layer.b.numpy()
    current_c     = u_net.analytical_layer.c.numpy()
    
    
    weights_a1.append((current_alpha)) 
    weights_b.append(current_b)
    weights_c.append(current_c)

    Error_a1.append(np.abs(current_alpha - alpha_o)/np.abs(alpha_o)) 
    Error_b.append(np.abs(current_b- b_o)/np.abs(b_o))
    Error_c.append(np.abs(current_c - c_o)/np.abs(c_o))

# Print the weights
    print('[b, c]:', weights_b[-1], weights_c[-1])
    print('[W1]:', weights_a1[-1])
    
    print('===========================================')

    
# End the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")


u_net.save('Saved_Models/u_net_FMC.keras')

mat_dict_history = {
    'num_epochs': num_epochs,
    'history': Loss,
    'weights_b': weights_b,
    'weights_c': weights_c,
    'weights_a1': weights_a1,
    'Error_b': Error_b,
    'Error_c': Error_c,
    'Error_a1': Error_a1,
}

scipy.io.savemat('Saved_Models/Data_history_FMC.mat', mat_dict_history)
## Ploting the results of u
plot_model_2D(u_net, Loss, weights_b, weights_c,num_epochs)
u_net = keras.models.load_model(
    'Saved_Models/u_net_FMC.keras',
    compile=False,
    custom_objects={
        'my_relu': my_relu,
        'my_heaviside': my_heaviside
    }
)












