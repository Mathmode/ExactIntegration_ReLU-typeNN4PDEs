import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
# This script works in float32. Set it explicitly: Spyder keeps the precision
# left behind by the previous script (e.g. float64 from the *_fast.py runs).
keras.backend.set_floatx('float32')
keras.config.set_dtype_policy('float32')

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


num_epochs = 30000
num_neurons_u = 20
num_layers = 1
activation = tf.nn.relu
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

def get_MC_quad(n_points):
    """
    Generates n_points random uniform points inside the reference triangle
    (0,0), (1,0), (0,1) using the folding method, with uniform weights.
    """
    # 1. Generate random pairs in the unit square [0, 1] x [0, 1]
    u = np.random.uniform(0.0, 1.0, n_points)
    v = np.random.uniform(0.0, 1.0, n_points)
    
    # # 2. Fold points that fall outside the triangle (u + v > 1) back inside
    # outside = (u + v) > 1.0
    # u[outside] = 1.0 - u[outside]
    # v[outside] = 1.0 - v[outside]
    
    # Stack into an (n_points, 2) array
    pts = np.column_stack((u, v))
    
    # 3. Assign uniform weights
    # Note: The area of the reference triangle is 0.5. 
    # Because your original Gauss weights summed to 0.5, these weights 
    # MUST be 0.5 / n_points to act as a mathematically correct drop-in replacement.
    wts = np.full(n_points, 0.5 / n_points)
    
    return pts

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





## ---------
## ---------
## ---------
# Step 2: we should construct an NN for u and use Deep Ritz method 
#         and exact integrations
## ---------
## ---------
## ---------
u_net = net_u(num_layers, num_neurons_u, activation = activation)
u = u_net(tf.random.uniform(shape=[1,2] , minval=0., maxval=1.))
u_net.summary()


#===================================================
#===================================================
#===================================================

w_values = np.zeros((2, num_neurons_u), dtype=np.float64)
w_values[0, :] = 0.0  # Coefficient for x
w_values[1, :] = 1.0  # Coefficient for y

# 2. Initialize biases: some uniform number between 0 and 1
b_values = np.random.uniform(0, 1, size=(num_neurons_u,)).astype(np.float64)

# 3. Apply these to the first layer in your layers_list
# layers_list[0] is the first Dense layer
u_net.layers_list[0].weights[0].assign(w_values)
u_net.layers_list[0].weights[1].assign(b_values)

#===================================================
#===================================================
#===================================================

optimizer = tf.keras.optimizers.SGD(learning_rate=initial_learning_rate)

x_eval = tf.linspace(0.0, 1.0, 1000)[:, None]  # 100 evenly spaced points
x_eval_tf = tf.expand_dims(tf.linspace(0.0, 1.0, 1000), axis=1)


# Start the timer
start_time = time.time()
## training u_net
Loss = []
k = 0
errors = []
d_errors = []


n_points = 400
x_lin = tf.linspace(0.0, 1.0, n_points)
X, Y = tf.meshgrid(x_lin, x_lin, indexing='ij')
x_eval_tf = tf.stack([tf.reshape(X, [-1]), tf.reshape(Y, [-1])], axis=1)

# Pre-compute exact values once outside the loop
u_exact = exact_u(x_eval_tf[:, 0], x_eval_tf[:, 1])
dux_exact = exact_dx(x_eval_tf[:, 0], x_eval_tf[:, 1])
duy_exact = exact_dy(x_eval_tf[:, 0], x_eval_tf[:, 1])




for epoch in range(num_epochs):
    
    #========= MC points =========
    # x = tf.random.uniform((n_MC,), 0.0, 1.0)
    # X, Y = tf.meshgrid(x, x, indexing='ij')
    # xy = tf.stack([tf.reshape(X, [-1]), tf.reshape(Y, [-1])], axis=1)
    xy = get_MC_quad(n_MC)
    xy = tf.convert_to_tensor(xy, dtype=tf.float32)
    
    # print(shima)
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
    
    # Compute exact and predicted solutions
    
    with tf.GradientTape() as t2:
        t2.watch(x_eval_tf)
        u_pred_tensor = u_net(x_eval_tf)
    
    grad = t2.gradient(u_pred_tensor, x_eval_tf)
    
    # Flattening ensures shape [N] matches u_exact shape [N]
    u_pred = tf.squeeze(u_pred_tensor).numpy()
    dux_pred = grad[:, 0].numpy()
    duy_pred = grad[:, 1].numpy()

    # Calculation of errors with explicit shape matching
    error = tf.sqrt(tf.reduce_mean(tf.math.square(u_exact - u_pred)))/tf.sqrt(tf.reduce_mean(tf.math.square(u_exact)))*100
    d_error = tf.sqrt(tf.reduce_mean(tf.math.square(dux_exact - dux_pred)+tf.math.square(duy_exact - duy_pred)))/tf.sqrt(tf.reduce_mean(tf.math.square(dux_exact)+tf.math.square(duy_exact)))*100
    print('L2_Error:',error.numpy())
    print('L2_dError:',d_error.numpy())
    errors.append(error.numpy())
    d_errors.append(d_error.numpy())

    
# End the timer
end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")


import os
script_dir = os.path.dirname(os.path.abspath(__file__))
save_dir = os.path.join(script_dir, 'Saved_Models')
os.makedirs(save_dir, exist_ok=True)

keras_path = os.path.join(save_dir, 'u_net_MC.keras')
mat_path = os.path.join(save_dir, 'Data_history_MC.mat')

u_net.save(keras_path)

mat_dict_history = {
    'num_epochs': num_epochs,
    'history': Loss,
    'errors': errors,
    'd_errors': d_errors,
}

scipy.io.savemat(mat_path, mat_dict_history)
## Ploting the results of u
plot_model_2D(u_net, Loss, exact_u, errors, d_errors, num_epochs)
u_net = keras.models.load_model(keras_path, compile=False)












