#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:40:19 2026

@author: sbahalouei
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:04:42 2025

@author: sbahalouei
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from Data import f_rhs, exact_u, exact_dx, exact_dy
import matplotlib.colors as mcolors
import keras
from NN_Architecture_2D import net_u
import scipy.io


My_Blue=(0, 0/255, 102/255)
# My_Green = (8/255,8/255,137/255)
# My_Green = (0/255,0/255,102/255)
My_Orange = (255/255, 178/255, 102/255)
My_Green = (10/255, 102/255, 28/255)


data = scipy.io.loadmat('Models/Data_history.mat')
data_S_i = scipy.io.loadmat('Models/Data_history_S_i.mat')
data_S_ij = scipy.io.loadmat('Models/Data_history_S_ij.mat')
data_MC = scipy.io.loadmat('Models/Data_history_MC.mat')

num_epochs = data['num_epochs']
history = data['history']
errors = data['errors'].flatten()
d_errors = data['d_errors'].flatten()


num_epochs_MC_i = data_S_i['num_epochs']
history_MC_i = data_S_i['history']
errors_MC_i = data_S_i['errors'].flatten()
d_errors_MC_i = data_S_i['d_errors'].flatten()


num_epochs_MC_ij = data_S_ij['num_epochs']
history_MC_ij = data_S_ij['history']
errors_MC_ij = data_S_ij['errors'].flatten()
d_errors_MC_ij = data_S_ij['d_errors'].flatten()

num_epochs_MC = data_MC['num_epochs']
history_MC = data_MC['history']
errors_MC = data_MC['errors'].flatten()
d_errors_MC = data_MC['d_errors'].flatten()


u_net_MC_i = keras.models.load_model('Models/u_net_S_i.keras', compile=False)
u_net_MC_ij = keras.models.load_model('Models/u_net_S_ij.keras', compile=False)
u_net = keras.models.load_model('Models/u_net.keras', compile=False)
u_net_MC = keras.models.load_model('Models/u_net_MC.keras', compile=False)
u_net_initialize = keras.models.load_model('Models/u_net_initialize.keras', compile=False)


weights_1_MC_i = u_net_MC_i.layers[0].weights[0]
biases_1_MC_i = u_net_MC_i.layers[0].weights[1]
weights_2_MC_i = u_net_MC_i.layers[-1].weights[0]

weights_1_MC_ij = u_net_MC_ij.layers[0].weights[0]
biases_1_MC_ij = u_net_MC_ij.layers[0].weights[1]
weights_2_MC_ij = u_net_MC_ij.layers[-1].weights[0]

weights_1 = u_net.layers[0].weights[0]
biases_1 = u_net.layers[0].weights[1]
weights_2 = u_net.layers[-1].weights[0]

weights_1_MC = u_net_MC.layers[0].weights[0]
biases_1_MC = u_net_MC.layers[0].weights[1]
weights_2_MC = u_net_MC.layers[-1].weights[0]

weights_1_initialize = u_net_initialize.layers[0].weights[0]
biases_1_initialize = u_net_initialize.layers[0].weights[1]
weights_2_initialize = u_net_initialize.layers[-1].weights[0]

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_relu_boundaries_and_save(W1, b1, title, filename, x_lim=(-.5, 1.5), y_lim=(-.5, 1.5)):
    x_grid = np.linspace(x_lim[0], x_lim[1], 400)
    y_grid = np.linspace(y_lim[0], y_lim[1], 400)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Font settings
    font = {'family': 'Times',
            'weight': 'normal',
            'size': 13}
    plt.rc('font', **font)
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{mathrsfs} \usepackage{amssymb}'
    

    plt.figure(figsize=(4.8, 4))
    
    # 1. Plot the unit square (0, 1) x (0, 1) for reference
    # This helps visualize which neurons actually cross the integration domain
    square = patches.Rectangle((0, 0), 1, 1, linewidth=3, edgecolor='black', facecolor='none', zorder=10)
    plt.gca().add_patch(square)
    
    # 2. Loop through each neuron
    for i in range(W1.shape[1]):
        w1, w2 = W1[0, i], W1[1, i]
        b = b1[i]
        
        # Calculate activation Z = w1*X + w2*Y + b
        Z = w1 * X + w2 * Y + b
        
        # Plot the boundary line (where Z = 0)
        # Handle vertical lines (w2 approx 0) separately to avoid division by zero
        if abs(w2) > 1e-8:
            y_line = -(w1 * x_grid + b) / w2
            plt.plot(x_grid, y_line, linewidth=1.5, alpha=0.8)
        else:
            x_line = -b / w1
            plt.axvline(x=x_line, linewidth=1.5, linestyle='--', alpha=0.8)

        # Fill the positive side (Z > 0) with a semi-transparent color
        plt.contourf(X, Y, Z, levels=[0, np.inf], colors=[plt.cm.tab10(i % 10)], alpha=0.05)

    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    
    # Place legend outside to keep the plot clean
    # plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save the figure
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    plt.show()

# --- Apply to your models with the requested range (-1 to 2) ---

# Model 1: Standard u_net
W1, b1 = u_net.layers[0].get_weights()
plot_relu_boundaries_and_save(W1, b1, "Full-Bached", "Saved_Pics/boundaries_u_net.pdf")

# Model 2: u_net_MC_i
W1_i, b1_i = u_net_MC_i.layers[0].get_weights()
plot_relu_boundaries_and_save(W1_i, b1_i, "Neuron-Wise", "Saved_Pics/boundaries_MC_i.pdf")

# Model 3: u_net_MC_ij
W1_ij, b1_ij = u_net_MC_ij.layers[0].get_weights()
plot_relu_boundaries_and_save(W1_ij, b1_ij, "Double Neuron-Wise", "Saved_Pics/boundaries_MC_ij.pdf")

# Model 4: u_net_MC
W1_mc, b1_mc = u_net_MC.layers[0].get_weights()
plot_relu_boundaries_and_save(W1_mc, b1_mc, "Monte Carlo", "Saved_Pics/boundaries_MC.pdf")

# Model 4: u_net_MC
W1_initialize, b1_initialize = u_net_initialize.layers[0].get_weights()
plot_relu_boundaries_and_save(W1_initialize, b1_initialize, "Initialization", "Saved_Pics/boundaries_Initial.pdf")


def count_active_lines(W1, b1, x_range=[0, 1], y_range=[0, 1]):
    active_count = 0
    
    for i in range(W1.shape[1]):
        w1, w2 = W1[0, i], W1[1, i]
        b = b1[i]
        
        # Values at the 4 corners of the square
        corners = [
            w1 * x_range[0] + w2 * y_range[0] + b,
            w1 * x_range[1] + w2 * y_range[0] + b,
            w1 * x_range[0] + w2 * y_range[1] + b,
            w1 * x_range[1] + w2 * y_range[1] + b
        ]
        
        # Count if ANY part of square is in positive half-space
        if max(corners) >= 0:
            active_count += 1
            
    return active_count

# Usage for your models:
print(f"Active lines in u_net: {count_active_lines(*u_net.layers[0].get_weights())}")
print(f"Active lines in u_net_MC_i: {count_active_lines(*u_net_MC_i.layers[0].get_weights())}")
print(f"Active lines in u_net_MC_ij: {count_active_lines(*u_net_MC_ij.layers[0].get_weights())}")
print(f"Active lines in u_net_MC: {count_active_lines(*u_net_MC.layers[0].get_weights())}")