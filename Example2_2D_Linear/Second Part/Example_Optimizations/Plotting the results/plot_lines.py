#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 21:40:19 2026

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
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


My_Blue=(0, 0/255, 102/255)
My_Orange = (255/255, 178/255, 102/255)
My_Green = (10/255, 102/255, 28/255)


data = scipy.io.loadmat('Models/Data_history_fast_explicit.mat')
data_S_i = scipy.io.loadmat('Models/Data_history_S_i_fast_explicit.mat')
data_S_ij = scipy.io.loadmat('Models/Data_history_S_ij_fast_explicit.mat')
data_MC = scipy.io.loadmat('Models/Data_history_MC.mat')
data_CMC = scipy.io.loadmat('Models/Data_history_CMC_fast_explicit.mat')
data_FR = scipy.io.loadmat('Models/Data_history_FR.mat')

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

num_epochs_CMC = data_CMC['num_epochs']
history_CMC = data_CMC['history']
errors_CMC = data_CMC['errors'].flatten()
d_errors_CMC = data_CMC['d_errors'].flatten()

num_epochs_FR = data_FR['num_epochs']
history_FR = data_FR['history']
errors_FR = data_FR['errors'].flatten()
d_errors_FR = data_FR['d_errors'].flatten()


import sys
import importlib.util

# # Load the specific Fake_ReLu architecture
# fr_arch_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../Deep_Ritz_2D_Fake_ReLu/NN_Architecture_2D.py')
# spec = importlib.util.spec_from_file_location("fr_arch", fr_arch_path)
# fr_arch = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(fr_arch)

# Load the specific CMC architecture
cmc_arch_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../Deep_Ritz_2D_CMC/NN_Architecture_2D.py')
spec_cmc = importlib.util.spec_from_file_location("cmc_arch", cmc_arch_path)
cmc_arch = importlib.util.module_from_spec(spec_cmc)
spec_cmc.loader.exec_module(cmc_arch)

# ==========================================
# HELPER TO FIX KERAS GLOBAL REGISTRY
# ==========================================
def load_with_registry_override(filepath, class_map):
    for k, v in class_map.items():
        keras.saving.get_custom_objects()[k] = v
        keras.saving.get_custom_objects()[f"Custom>{k}"] = v
    return keras.models.load_model(filepath, compile=False)

u_net_MC_i = load_with_registry_override('Models/u_net_S_i_fast_explicit.keras', {'net_u': net_u})
u_net_MC_ij = load_with_registry_override('Models/u_net_S_ij_fast_explicit.keras', {'net_u': net_u})
u_net = load_with_registry_override('Models/u_net_fast_explicit.keras', {'net_u': net_u})
u_net_MC = load_with_registry_override('Models/u_net_MC.keras', {'net_u': net_u})

# Prepare cmc custom objects dynamically based on what exists
cmc_custom_objects = {}
if hasattr(cmc_arch, 'DistributedKernelInitializer'):
    cmc_custom_objects['DistributedKernelInitializer'] = cmc_arch.DistributedKernelInitializer
if hasattr(cmc_arch, 'DistributedBiasInitializer'):
    cmc_custom_objects['DistributedBiasInitializer'] = cmc_arch.DistributedBiasInitializer
if hasattr(cmc_arch, 'FanInitializer'):
    cmc_custom_objects['FanInitializer'] = cmc_arch.FanInitializer
if hasattr(cmc_arch, 'net_u'):
    cmc_custom_objects['net_u'] = cmc_arch.net_u

u_net_CMC = load_with_registry_override('Models/u_net_CMC_fast_explicit.keras', cmc_custom_objects)

# u_net_FR = load_with_registry_override('Models/u_net_FR.keras', {
#     'my_relu': fr_arch.my_relu,
#     'my_heaviside': fr_arch.my_heaviside,
#     'AnalyticalReLULayer': fr_arch.AnalyticalReLULayer,
#     'net_u': fr_arch.net_u
# })

u_net_initialize = load_with_registry_override('Models/u_net_initialize_fast_explicit.keras', cmc_custom_objects)


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

weights_1_CMC = u_net_CMC.layers[0].weights[0]
biases_1_CMC = u_net_CMC.layers[0].weights[1]
weights_2_CMC = u_net_CMC.layers[-1].weights[0]

# Extracting variables carefully from AnalyticalReLULayer for FR
# analytical_layer = u_net_FR.layers[0]

# # Since we updated Fake_ReLu to use FanInitializer, it now has W, b, c
# W1_FR = analytical_layer.w.numpy()
# b1_FR = analytical_layer.b.numpy()
# weights_2_FR = analytical_layer.c.numpy()


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
    square = patches.Rectangle((0, 0), 1, 1, linewidth=3, edgecolor='black', facecolor='none', zorder=10)
    plt.gca().add_patch(square)
    
    # 2. Loop through each neuron
    for i in range(W1.shape[1]):
        w1, w2 = W1[0, i], W1[1, i]
        
        # Handle scalar b1 gracefully
        if np.isscalar(b1):
            b = b1
        else:
            b = b1[i]
        
        # Calculate activation Z = w1*X + w2*Y + b
        Z = w1 * X + w2 * Y + b
        
        if abs(w2) > 1e-15:
            y_line = -(w1 * x_grid + b) / w2
            plt.plot(x_grid, y_line, linewidth=1.5, alpha=0.8)
        else:
            x_line = -b / w1
            plt.axvline(x=x_line, linewidth=1.5, linestyle='--', alpha=0.8)

        plt.contourf(X, Y, Z, levels=[0, np.inf], colors=[plt.cm.tab10(i % 10)], alpha=0.05)

    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True, which='both', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, dpi=300)
    print(f"Saved: {filename}")
    plt.show()

# --- Apply to your models with the requested range (-1 to 2) ---

# Model 1: Standard u_net
W1, b1 = u_net.layers[0].get_weights()
plot_relu_boundaries_and_save(W1, b1, "Exact Integration", "Saved_Pics/boundaries_u_net.pdf")

# # Model 2: u_net_MC_i
# W1_i, b1_i = u_net_MC_i.layers[0].get_weights()
# plot_relu_boundaries_and_save(W1_i, b1_i, "Neuron-Wise", "Saved_Pics/boundaries_MC_i.pdf")

# # Model 3: u_net_MC_ij
# W1_ij, b1_ij = u_net_MC_ij.layers[0].get_weights()
# plot_relu_boundaries_and_save(W1_ij, b1_ij, "Double Neuron-Wise", "Saved_Pics/boundaries_MC_ij.pdf")

# Model 4: u_net_MC
W1_mc, b1_mc = u_net_MC.layers[0].get_weights()
plot_relu_boundaries_and_save(W1_mc, b1_mc, "Monte Carlo", "Saved_Pics/boundaries_MC.pdf")

# Model 4: u_net_CMC
W1_cmc, b1_cmc = u_net_CMC.layers[0].get_weights()
plot_relu_boundaries_and_save(W1_cmc, b1_cmc, "Consistent Monte Carlo", "Saved_Pics/boundaries_CMC.pdf")

# # Model 4: u_net_FR
# plot_relu_boundaries_and_save(W1_FR, b1_FR, "Surrogate Derivative", "Saved_Pics/boundaries_FR.pdf")

# Model 4: u_net_FR
W1_initialize, b1_initialize = u_net_initialize.layers[0].get_weights()
plot_relu_boundaries_and_save(W1_initialize, b1_initialize, "Initialization", "Saved_Pics/boundaries_Initial.pdf")


def count_active_lines(W1, b1, x_range=[0, 1], y_range=[0, 1]):
    active_count = 0
    
    for i in range(W1.shape[1]):
        w1, w2 = W1[0, i], W1[1, i]
        if np.isscalar(b1):
            b = b1
        else:
            b = b1[i]
        
        # Values at the 4 corners of the square
        corners = [
            w1 * x_range[0] + w2 * y_range[0] + b,
            w1 * x_range[1] + w2 * y_range[0] + b,
            w1 * x_range[0] + w2 * y_range[1] + b,
            w1 * x_range[1] + w2 * y_range[1] + b
        ]
        
        if max(corners) >= 0:
            active_count += 1
            
    return active_count

# Usage for your models:
print(f"Active lines in u_net: {count_active_lines(*u_net.layers[0].get_weights())}")
print(f"Active lines in u_net_MC_i: {count_active_lines(*u_net_MC_i.layers[0].get_weights())}")
print(f"Active lines in u_net_MC_ij: {count_active_lines(*u_net_MC_ij.layers[0].get_weights())}")
print(f"Active lines in u_net_MC: {count_active_lines(*u_net_MC.layers[0].get_weights())}")
print(f"Active lines in u_net_CMC: {count_active_lines(*u_net_CMC.layers[0].get_weights())}")
# print(f"Active lines in u_net_FR: {count_active_lines(W1_FR, b1_FR)}")
