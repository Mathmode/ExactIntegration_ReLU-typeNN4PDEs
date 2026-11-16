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
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))


My_Blue=(0, 0/255, 102/255)
My_Orange = (255/255, 178/255, 102/255)
My_Green = (10/255, 102/255, 28/255)



data_CMC = scipy.io.loadmat('Models/Data_history_CMC_fast_explicit.mat')
data_CMC_i = scipy.io.loadmat('Models/Data_history_CMC_i_fast_explicit.mat')
data_CMC_ij = scipy.io.loadmat('Models/Data_history_CMC_ij_fast_explicit.mat')


num_epochs_CMC = data_CMC['num_epochs']
history_CMC = data_CMC['history']
errors_CMC = data_CMC['errors'].flatten()
d_errors_CMC = data_CMC['d_errors'].flatten()

num_epochs_CMC_i = data_CMC_i['num_epochs']
history_CMC_i = data_CMC_i['history']
errors_CMC_i = data_CMC_i['errors'].flatten()
d_errors_CMC_i = data_CMC_i['d_errors'].flatten()

num_epochs_CMC_ij = data_CMC_ij['num_epochs']
history_CMC_ij = data_CMC_ij['history']
errors_CMC_ij = data_CMC_ij['errors'].flatten()
d_errors_CMC_ij = data_CMC_ij['d_errors'].flatten()

# the first SKIP iterations are not plotted; the x-axis keeps the iteration numbers
SKIP = 200

def curve(y):
    return np.arange(SKIP, len(y)), y[SKIP:]


import sys
import importlib.util

def load_with_registry_override(filepath, class_map):
    for k, v in class_map.items():
        keras.saving.get_custom_objects()[k] = v
        keras.saving.get_custom_objects()[f"Custom>{k}"] = v
    return keras.models.load_model(filepath, compile=False)

u_net_CMC_i = load_with_registry_override('Models/u_net_CMC_i_fast_explicit.keras', {'net_u': net_u})
u_net_CMC_ij = load_with_registry_override('Models/u_net_CMC_ij_fast_explicit.keras', {'net_u': net_u})
u_net_CMC = load_with_registry_override('Models/u_net_CMC_fast_explicit.keras', {'net_u': net_u})
# u_net = load_with_registry_override('Models/u_net.keras', {'net_u': net_u})
# u_net_S_i = load_with_registry_override('Models/u_net_S_i.keras', {'net_u': net_u})
# u_net_S_ij = load_with_registry_override('Models/u_net_S_ij.keras', {'net_u': net_u})


def plot_model_2D(
    u_net_CMC_i,
    u_net_CMC_ij,
    u_net_CMC,
    errors_CMC,
    derrors_CMC,
    errors_CMC_i,
    derrors_CMC_i,
    errors_CMC_ij,
    derrors_CMC_ij,
    num_epochs,
    N=100
):
    """
    Plot 2D Deep Ritz results:
    - u(x,y)
    - gradients
    - pointwise errors
    - training curves
    """
    
    # -------------------------------------------------
    # Error history Plots
    # -------------------------------------------------
    
    font = {'family': 'Times',
            'weight': 'normal',
            'size': 17}
    plt.rc('font', **font)
    plt.rcParams['text.usetex'] = True
    # We can safely use standard amssymb and mathrsfs
    plt.rcParams['text.latex.preamble'] = r'\usepackage{mathrsfs} \usepackage{amssymb} \usepackage{amsmath}'
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.subplots_adjust(left=0.1, bottom=0.2, right=0.98, top=0.85, wspace=0.3, hspace=0.4)
    
    # # --- Top Left [0, 0]: Exact Integration (u errors) ---
    # axes[0, 0].plot(errors, linewidth=1.5, color=My_Blue, label='Full-Batched')
    # axes[0, 0].plot(errors_S_i, linewidth=1.5, color=My_Orange, label='Neuron-Wise')
    # axes[0, 0].plot(errors_S_ij, linewidth=1.5, color=My_Green, label='Double Neuron-Wise')
    
    # axes[0, 0].set_title('Exact Integration', fontsize=17)
    # axes[0, 0].set_xlabel('Iterations')
    # axes[0, 0].set_ylabel(r'Relative error of $u_\theta$ (in \%)')
    # # axes[0, 0].set_yscale('log')
    # axes[0, 0].set_xscale('log')
    # axes[0, 0].set_yticks([1, 10, 20, 30, 40, 50, 60])
    # axes[0, 0].set_yticklabels([r'$1$', r'$10$', r'$20$', r'$30$', r'$40$', r'$50$', r'$60$'])
    # axes[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # --- Top Right [0, 1]: Conforming Monte Carlo (u errors) ---
    axes[0].plot(*curve(errors_CMC), linewidth=1.5, color=My_Blue, label='Full-Batched')
    axes[0].plot(*curve(errors_CMC_i), linewidth=1.5, color=My_Orange, label='Neuron-Wise')
    axes[0].plot(*curve(errors_CMC_ij), linewidth=1.5, color=My_Green, label='Double Neuron-Wise')

    axes[0].set_title('Conforming Monte Carlo', fontsize=17)
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel(r'Relative error of $u_\theta$ (in \%)')
    # axes[0].set_yscale('log')
    # axes[0].set_xscale('log')
    # axes[0].set_yticks([1, 10, 100])
    # axes[0].set_yticklabels([r'$1$', r'$20$', r'$40$', r'$60$', r'$80$', r'$100$'])
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # # --- Bottom Left [1, 0]: Exact Integration (gradient errors) ---
    # axes[0, 1].plot(derrors, linewidth=1.5, color=My_Blue, label='Full-Batched')
    # axes[0, 1].plot(derrors_S_i, linewidth=1.5, color=My_Orange, label='Neuron-Wise')
    # axes[0, 1].plot(derrors_S_ij, linewidth=1.5, color=My_Green, label='Double Neuron-Wise')
    
    # axes[0, 1].set_title('Exact Integration', fontsize=17)
    # axes[0, 1].set_xlabel('Iterations')
    # axes[0, 1].set_ylabel(r'Relative error of $|\nabla u_\theta|$ (in \%)')
    # axes[0, 1].set_yscale('log')
    # axes[0, 1].set_xscale('log')
    # axes[0, 1].set_yticks([10, 30, 50, 70, 90])
    # axes[0, 1].set_yticklabels([r'$10$', r'$30$', r'$50$', r'$70$', r'$90$'])
    # axes[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # --- Bottom Right [1, 1]: Conforming Monte Carlo (gradient errors) ---
    axes[1].plot(*curve(derrors_CMC), linewidth=1.5, color=My_Blue, label='Full-Batched')
    axes[1].plot(*curve(derrors_CMC_i), linewidth=1.5, color=My_Orange, label='Neuron-Wise')
    axes[1].plot(*curve(derrors_CMC_ij), linewidth=1.5, color=My_Green, label='Double Neuron-Wise')

    axes[1].set_title('Conforming Monte Carlo', fontsize=17)
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel(r'Relative error of $|\nabla u_\theta|$ (in \%)')
    # axes[1].set_yscale('log')
    # axes[1].set_xscale('log')
    # axes[1].set_yticks([5, 10, 100])
    # axes[1].set_yticklabels([r'$1$', r'$20$', r'$40$', r'$60$', r'$80$', r'$100$'])
    axes[1].grid(True, linestyle='--', alpha=0.7)
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, 0.90), fontsize=17, frameon=False)
    
    # Ensure Saved_Pics directory exists
    if not os.path.exists('Saved_Pics'):
        os.makedirs('Saved_Pics')

    plt.savefig('Saved_Pics/L2_derror_2D_nonpolynomial.pdf', dpi=300)
    plt.show()
    
plot_model_2D(
    u_net_CMC_i,
    u_net_CMC_ij,
    u_net_CMC,
    errors_CMC,
    d_errors_CMC,
    errors_CMC_i,
    d_errors_CMC_i,
    errors_CMC_ij,
    d_errors_CMC_ij,
    num_epochs_CMC,
    N=100)