import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

from NN_Architecture_2D import net_u
from Integrate_Points import get_exact_conical_quad
from train_u_1D import train_step_FB, train_step_NW, train_step_DNW, lr_schedule
from Data import exact_u, f_rhs, exact_dx, exact_dy
from Plot import plot_model_2D
import keras
keras.utils.set_random_seed(1234)
import time as time
from numpy.polynomial.polynomial import Polynomial
import scipy.io
from numpy.polynomial.polynomial import Polynomial
from Fit_2D_Poly_Stable import fit_2d_poly_stable

import warnings
import numpy as np

# This will stop the console flood from Shapely/GEOS
warnings.filterwarnings("ignore", category=RuntimeWarning, module="shapely")
# Some versions of Shapely also trigger through numpy
warnings.filterwarnings("ignore", message="invalid value encountered in intersects")
warnings.filterwarnings("ignore", message="invalid value encountered in intersection")


num_epochs = 10000
num_neurons_u = 10
batch_size_i = 1
num_layers = 1
activation = tf.nn.relu
initial_learning_rate = 0.1
final_learning_rate = 0.0001
n_points = 10

# Method = 'FB'  #Full-Batched
Method = 'NW'  #Neuron-Wise
# Method = 'DNW' # Double Neuron-Wise

alpha = 0.0  # Reaction term coefficient
lambda_reg = 0.001  # Interior regularization penalty weight to keep lines inside [0, 1]^2


domain_vertices = [(0, 0), (1,0), (1, 1), (0, 1)]

pts, wts = get_exact_conical_quad(n_points)




f_degree = [5, 5]
f_coeffs = fit_2d_poly_stable(f_degree)


u_net = net_u(num_layers, num_neurons_u, activation = activation)
u = u_net(tf.random.uniform(shape=[1,2] , minval=0., maxval=1.))


#===================================================
#======== setting weights and biases ===============
#===================================================

# w_values = np.zeros((2, num_neurons_u), dtype=np.float64)
# w_values[0, :] = 1.0  # Coefficient for x
# w_values[1, :] = 1.0  # Coefficient for y

# b_values = 0*np.random.uniform(0, 1, size=(num_neurons_u,)).astype(np.float64)

# u_net.layers_list[0].weights[0].assign(w_values)
# u_net.layers_list[0].weights[1].assign(b_values)

#===================================================
#===================================================
script_dir = os.path.dirname(os.path.abspath(__file__))

u_net.save(os.path.join(script_dir, 'Saved_Models/u_net_initialize.keras'))
u_net.summary()


optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)

# p1 = tf.ones([dat_len, 1], dtype=dtype) * 30.0
# p2 = tf.ones([dat_len, 1], dtype=dtype) * 1.0

# Shape: (dat_len, 1, 2)
# P_val = tf.stack([p1, p2], axis=-1)



x_eval = tf.linspace(0.0, 1.0, 1000)[:, None]  # 100 evenly spaced points
x_eval_tf = tf.expand_dims(tf.linspace(0.0, 1.0, 1000), axis=1)


# Start the timer
start_time = time.time()
## training u_net
Loss = []
k = 0
errors = []
d_errors = []


n_points = 100
x_lin = tf.linspace(0.0, 1.0, n_points)
X, Y = tf.meshgrid(x_lin, x_lin, indexing='ij')
x_eval_tf = tf.stack([tf.reshape(X, [-1]), tf.reshape(Y, [-1])], axis=1)

# Pre-compute exact values once outside the loop
u_exact = exact_u(x_eval_tf[:, 0], x_eval_tf[:, 1])
dux_exact = exact_dx(x_eval_tf[:, 0], x_eval_tf[:, 1])
duy_exact = exact_dy(x_eval_tf[:, 0], x_eval_tf[:, 1])


if Method =='FB':
    
    for epoch in range(num_epochs):
        new_learning_rate = lr_schedule(epoch, initial_learning_rate, \
                                        final_learning_rate, num_epochs)
        optimizer.learning_rate.assign(new_learning_rate)
        
        Loss += [train_step_FB(u_net, f_coeffs, f_degree, optimizer, num_neurons_u, domain_vertices, batch_size_i, pts, wts, alpha, lambda_reg)]
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


    u_net.save(os.path.join(script_dir, 'Saved_Models/u_net.keras'))

    mat_dict_history = {
        'elapsed_time': elapsed_time,
        'num_epochs': num_epochs,
        'history': Loss,
        'errors': errors,
        'd_errors': d_errors,
    }

    scipy.io.savemat(os.path.join(script_dir, 'Saved_Models/Data_history.mat'), mat_dict_history)
    ## Ploting the results of u
    plot_model_2D(u_net, Loss, exact_u, errors, d_errors, num_epochs)
    u_net = keras.models.load_model(os.path.join(script_dir, 'Saved_Models/u_net.keras'), compile=False)
    
elif Method == 'NW':
    
    for epoch in range(num_epochs):
        new_learning_rate = lr_schedule(epoch, initial_learning_rate, \
                                        final_learning_rate, num_epochs)
        optimizer.learning_rate.assign(new_learning_rate)
        
        Loss += [train_step_NW(u_net, f_coeffs, f_degree, optimizer, num_neurons_u, domain_vertices, batch_size_i, pts, wts, new_learning_rate, alpha, lambda_reg)]
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


    u_net.save(os.path.join(script_dir, 'Saved_Models/u_net_S_i.keras'))

    mat_dict_history = {
        'elapsed_time': elapsed_time,
        'num_epochs': num_epochs,
        'history': Loss,
        'errors': errors,
        'd_errors': d_errors,
    }

    scipy.io.savemat(os.path.join(script_dir, 'Saved_Models/Data_history_S_i.mat'), mat_dict_history)
    ## Ploting the results of u
    plot_model_2D(u_net, Loss, exact_u, errors, d_errors, num_epochs)
    u_net = keras.models.load_model(os.path.join(script_dir, 'Saved_Models/u_net_S_i.keras'), compile=False)
    
elif Method == 'DNW':
    
    for epoch in range(num_epochs):
        new_learning_rate = lr_schedule(epoch, initial_learning_rate, \
                                        final_learning_rate, num_epochs)
        optimizer.learning_rate.assign(new_learning_rate)
        
        Loss += [train_step_DNW(u_net, f_coeffs, f_degree, optimizer, num_neurons_u, domain_vertices, batch_size_i, pts, wts, new_learning_rate, alpha, lambda_reg)]
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


    u_net.save(os.path.join(script_dir, 'Saved_Models/u_net_S_ij.keras'))

    mat_dict_history = {
        'elapsed_time': elapsed_time,
        'num_epochs': num_epochs,
        'history': Loss,
        'errors': errors,
        'd_errors': d_errors,
    }

    scipy.io.savemat(os.path.join(script_dir, 'Saved_Models/Data_history_S_ij.mat'), mat_dict_history)
    ## Ploting the results of u
    plot_model_2D(u_net, Loss, exact_u, errors, d_errors, num_epochs)
    u_net = keras.models.load_model(os.path.join(script_dir, 'Saved_Models/u_net_S_ij.keras'), compile=False)



    













