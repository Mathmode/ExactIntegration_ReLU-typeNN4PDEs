import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras

from NN_Architecture_2D import net_u
from Integrate_Points import get_exact_conical_quad
from train_u_1D import train_step_FB, lr_schedule
from Data import f_rhs
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


num_epochs = 500
num_neurons_u = 1
batch_size_i = 1
num_layers = 1
activation = tf.nn.relu
initial_learning_rate = 1.e-1
final_learning_rate = 1.e-2
n_points = 100


domain_vertices = [(0, 0), (1,0), (1, 1), (0, 1)]

pts, wts = get_exact_conical_quad(n_points)

Method = 'FB'  #Full-Batched
# Method = 'NW'  #Neuron-Wise
# Method = 'DNW' # Double Neuron-Wise



f_degree = [5, 5]
f_coeffs = fit_2d_poly_stable(f_degree)

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
errors = []
d_errors = []


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

if Method =='FB':
    
    for epoch in range(num_epochs):
        new_learning_rate = lr_schedule(epoch, initial_learning_rate, \
                                        final_learning_rate, num_epochs)
        optimizer.learning_rate.assign(new_learning_rate)
        
        Loss += [train_step_FB(u_net, f_coeffs, f_degree, optimizer, num_neurons_u, domain_vertices, batch_size_i, epoch, pts, wts)]
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


    u_net.save('Saved_Models/u_net.keras')

    mat_dict_history = {
        'elapsed_time': elapsed_time,
        'num_epochs': num_epochs,
        'history': Loss,
        'weights_b': weights_b,
        'weights_c': weights_c,
        'weights_a1': weights_a1,
        'Error_b': Error_b,
        'Error_c': Error_c,
        'Error_a1': Error_a1,
    }

    scipy.io.savemat('Saved_Models/Data_history.mat', mat_dict_history)
    ## Ploting the results of u
    plot_model_2D(u_net, Loss, weights_b, weights_c, num_epochs)
    u_net = keras.models.load_model('Saved_Models/u_net.keras', compile=False)
 













