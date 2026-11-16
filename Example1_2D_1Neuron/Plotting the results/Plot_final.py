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
My_Green = (50/255, 92/255, 18/255)
My_purple = (191/255, 64/255, 191/255)


data = scipy.io.loadmat('Models/Data_history.mat')
data_MC = scipy.io.loadmat('Models/Data_history_MC.mat')
data_CMC = scipy.io.loadmat('Models/Data_history_CMC.mat')
data_FMC_05 = scipy.io.loadmat('Models/Data_history_FMC_05.mat')
data_FMC_01 = scipy.io.loadmat('Models/Data_history_FMC_01.mat')
data_FMC_1 = scipy.io.loadmat('Models/Data_history_FMC_1.mat')

num_epochs = data['num_epochs']
history = data['history']
Error_b = data['weights_b'].flatten()
Error_c = tf.squeeze(data['weights_c'].flatten())
Error_a1 = tf.squeeze(data['weights_a1'].flatten())

num_epochs_MC = data_MC['num_epochs']
history_MC = data_MC['history']
Error_b_MC = data_MC['weights_b'].flatten()
Error_c_MC = tf.squeeze(data_MC['weights_c'].flatten())
Error_a1_MC = tf.squeeze(data_MC['weights_a1'].flatten())

num_epochs_CMC = data_CMC['num_epochs']
history_CMC = data_CMC['history']
Error_b_CMC = data_CMC['weights_b'].flatten()
Error_c_CMC = tf.squeeze(data_CMC['weights_c'].flatten())
Error_a1_CMC = tf.squeeze(data_CMC['weights_a1'].flatten())

num_epochs_FMC_05 = data_FMC_05['num_epochs']
history_FMC_05 = data_FMC_05['history']
Error_b_FMC_05 = data_FMC_05['weights_b'].flatten()
Error_c_FMC_05 = tf.squeeze(data_FMC_05['weights_c'].flatten())
Error_a1_FMC_05 = tf.squeeze(data_FMC_05['weights_a1'].flatten())

num_epochs_FMC_01 = data_FMC_01['num_epochs']
history_FMC_01 = data_FMC_01['history']
Error_b_FMC_01 = data_FMC_01['weights_b'].flatten()
Error_c_FMC_01 = tf.squeeze(data_FMC_01['weights_c'].flatten())
Error_a1_FMC_01 = tf.squeeze(data_FMC_01['weights_a1'].flatten())

num_epochs_FMC_1 = data_FMC_1['num_epochs']
history_FMC_1 = data_FMC_1['history']
Error_b_FMC_1 = data_FMC_1['weights_b'].flatten()
Error_c_FMC_1 = tf.squeeze(data_FMC_1['weights_c'].flatten())
Error_a1_FMC_1 = tf.squeeze(data_FMC_1['weights_a1'].flatten())


# u_net_MC = keras.models.load_model('Models/u_net_MC.keras', compile=False)
# u_net_CMC = keras.models.load_model('Models/u_net_CMC.keras', compile=False)
# u_net = keras.models.load_model('Models/u_net.keras', compile=False)



import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_model_2D(
    Error_b,
    Error_c,
    Error_a1,
    Error_b_MC,
    Error_c_MC,
    Error_a1_MC,
    Error_b_CMC,
    Error_c_CMC,
    Error_a1_CMC,
    Error_b_FMC_05,
    Error_c_FMC_05,
    Error_a1_FMC_05,
    Error_b_FMC_01,
    Error_c_FMC_01,
    Error_a1_FMC_01,
    Error_b_FMC_1,
    Error_c_FMC_1,
    Error_a1_FMC_1,
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
    
    
    # font = {'family' : 'Times',
    #             'weight' : 'normal',
    #             'size'   : 16}
    # plt.rc('font', **font)
    # plt.figure(figsize=(4, 3))
    # plt.rcParams['text.usetex'] = True
    
    # x = np.linspace(0, 500, 500)
    # b = np.ones_like(x)*(-0.5988)
    # c = np.ones_like(x)*7.4323
    # alpha = np.ones_like(x)*0.7854
    
    # # --- First Figure ---
    # plt.figure(figsize=(8.5, 3.5))
    # plt.plot(Error_b, linewidth=1.5,linestyle='-', color=My_Orange, label='Exact Integration')
    # plt.plot(Error_b_MC, linewidth=1.5,linestyle='--', color='red', label='Standard Monte-Carlo')
    # plt.plot(Error_b_CMC, linewidth=1.5,linestyle='-.', color=My_Blue, label='Conforming Monte-Carlo')
    # plt.plot(Error_b_FMC, linewidth=1.5,linestyle=':', color='blue', label='Fake ReLu')
    # plt.plot(x, b, '--', color='black')
    # plt.text(300 , -0.7, r'$b^* \approx -0.60$', horizontalalignment='center', fontsize=12)
    # plt.xlabel('Iterations')
    # plt.ylabel(r'Values of $b$')
    # # plt.yscale('log')
    
    # # Place the legend exactly above the plot in 2 columns
    # plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize='small')
    
    # plt.tight_layout()
    # plt.savefig('Saved_Pics/b_values.pdf', dpi=300) # Changed name slightly to avoid overwriting
    # plt.show()
    
    
    # # --- Second Figure ---
    # font = {'family' : 'Times',
    #             'weight' : 'normal',
    #             'size'   : 16}
    # plt.rc('font', **font)
    # plt.figure(figsize=(8.5, 3.5))
    # plt.rcParams['text.usetex'] = True
    # plt.plot(Error_c, linewidth=2.5, color=My_Orange,
    #          linestyle='-', label='Exact Integration')      # solid

    # plt.plot(Error_c_MC, linewidth=1.5, color='red',
    #          linestyle='--', label='Standard  Monte-Carlo')           # dashed
    
    # plt.plot(Error_c_CMC, linewidth=1.5, color=My_Blue,
    #          linestyle='-.', label='Conforming Monte-Carlo') # dash-dot
    
    # plt.plot(Error_c_FMC, linewidth=1.5, color='blue',
    #          linestyle=':', label='Fake ReLu')             # dotted
    # plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize='small')
    # plt.plot(x, c, '--', color='black')
    # plt.text(50 , 6.8, r'$c^* \approx 7.43$', horizontalalignment='center', fontsize=12)
    # plt.xlabel('Iterations')
    # plt.ylabel(r'Values of $c$')
    # # plt.yscale('log')
    # # plt.xscale('log')
    
    # plt.tight_layout()
    # plt.savefig('Saved_Pics/c_values.pdf', dpi=300) 
    # plt.show()
    
    # # --- Second Figure ---
    # font = {'family' : 'Times',
    #             'weight' : 'normal',
    #             'size'   : 16}
    # plt.rc('font', **font)
    # plt.figure(figsize=(8.5, 3.5))
    # plt.rcParams['text.usetex'] = True
    # plt.plot(Error_a1, linewidth=2.5,linestyle='-', color=My_Orange, label='Exact Integration')
    # plt.plot(Error_a1_MC, linewidth=1.5,linestyle='--',  color='red', label='Standard Monte-Carlo')
    # plt.plot(Error_a1_CMC, linewidth=1.5,linestyle='-.', color=My_Blue, label='Conforming Monte-Carlo')
    # plt.plot(Error_a1_FMC,linestyle=':', linewidth=1.5, color='blue', label='Fake ReLu')
    # plt.plot(x, alpha, '--', color='black')
    # plt.text(300 , 0.6, r'$\alpha ^* \approx 0.79$', horizontalalignment='center', fontsize=12)
    # plt.xlabel('Iterations')
    # plt.ylabel(r'Values of $\alpha$')
    # # plt.yscale('log')
    # # plt.xscale('log')
    
    # # Place the legend exactly above the plot in 2 columns
    # plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, fontsize='small')
    
    # plt.tight_layout()
    # plt.savefig('Saved_Pics/alpha_values.pdf', dpi=300) 
    # plt.show()
    
    
    font = {
        'family': 'Times',
        'weight': 'normal',
        'size': 12
    }
    plt.rc('font', **font)
    plt.rcParams['text.usetex'] = True
    
    x = np.linspace(0, 500, 500)
    
    b = np.ones_like(x) * (-0.5988)
    c = np.ones_like(x) * 7.4323
    alpha = np.ones_like(x) * 0.7854
    
    # =====================================================
    # Create 4-row figure:
    # row 0 = legend
    # row 1 = b
    # row 2 = c
    # row 3 = alpha
    # =====================================================
    
    fig = plt.figure(figsize=(7.5, 7))
    
    gs = fig.add_gridspec(
        nrows=4,
        ncols=1,
        height_ratios=[-0.15, 1, 1, 1]
        
    )
    
    ax_leg = fig.add_subplot(gs[0])
    ax_b   = fig.add_subplot(gs[1])
    ax_c   = fig.add_subplot(gs[2])
    ax_a   = fig.add_subplot(gs[3])
    
    # -----------------------------------------------------
    # b plot
    # -----------------------------------------------------
    l1, = ax_b.plot(Error_b, linewidth=1, linestyle='-',
                    color=My_purple, label='Exact Integration')
    
    l2, = ax_b.plot(Error_b_MC, linewidth=1, linestyle='--',
                    color='red', label='Standard Monte-Carlo')
    
    l3, = ax_b.plot(Error_b_CMC, linewidth=1, linestyle=':',
                    color=My_Blue, label='Conforming Monte-Carlo')
    
    l4, = ax_b.plot(Error_b_FMC_05, linewidth=1, linestyle='-', 
    marker='o',markersize=2.5, markevery=30,
                    color=My_Green, label=r'Surrogate Gradient ($\beta=0.5$)')
    
    l5, = ax_b.plot(Error_b_FMC_01, linewidth=1, linestyle='--', 
    marker='o',markersize=2.5, markevery=35,
                    color=My_Orange, label=r'Surrogate Gradient ($\beta=0.1$)')
    
    l6, = ax_b.plot(Error_b_FMC_1, linewidth=1, linestyle='-.', 
    marker='o',markersize=2.5, markevery=40,
                    color='blue', label=r'Surrogate Gradient ($\beta=0.25$)')
    
    ax_b.plot(x, b, linewidth=1, color='black')
    
    ax_b.text(
        300, -0.7,
        r'$b^*_1 \approx -0.60$',
        ha='center',
        fontsize=12
    )
    
    ax_b.set_ylabel(r'$b$')
    ax_b.set_xlabel('Iterations')
    # -----------------------------------------------------
    # c plot
    # -----------------------------------------------------
    ax_c.plot(Error_c, linewidth=1, linestyle='-',
              color=My_purple)
    
    ax_c.plot(Error_c_MC, linewidth=1, linestyle='--',
              color='red')
    
    ax_c.plot(Error_c_CMC, linewidth=1, linestyle='-', 
              color=My_Blue)
    
    ax_c.plot(Error_c_FMC_05, linewidth=1, linestyle='-', 
    marker='o',markersize=2.5, markevery=30,
              color=My_Green)
    
    ax_c.plot(Error_c_FMC_01, linewidth=1, linestyle='--', 
    marker='o',markersize=2.5, markevery=35,
              color=My_Orange)
    
    ax_c.plot(Error_c_FMC_1, linewidth=1, linestyle='-.', 
    marker='o',markersize=2.5, markevery=40,
              color='blue')
    
    ax_c.plot(x, c, linewidth=1, color='black')
    
    ax_c.text(
        50, 6.5,
        r'$c^*_1 \approx 7.43$',
        ha='center',
        fontsize=12
    )
    
    ax_c.set_ylabel(r'$c$')
    ax_c.set_xlabel('Iterations')
    # -----------------------------------------------------
    # alpha plot
    # -----------------------------------------------------
    ax_a.plot(Error_a1, linewidth=1, linestyle='-',
              color=My_purple)
    
    ax_a.plot(Error_a1_MC, linewidth=1, linestyle='--',
              color='red')
    
    ax_a.plot(Error_a1_CMC, linewidth=1, linestyle=':',
              color=My_Blue)
    
    ax_a.plot(Error_a1_FMC_05, linewidth=1, linestyle='-', 
    marker='o',markersize=2.5, markevery=30, 
              color=My_Green)
    
    ax_a.plot(Error_a1_FMC_01, linewidth=1, linestyle='--', 
    marker='o',markersize=2.5, markevery=35,
              color=My_Orange)
    
    ax_a.plot(Error_a1_FMC_1, linewidth=1, linestyle='-.', 
    marker='o',markersize=2.5, markevery=40,
              color='blue')
    
    ax_a.plot(x, alpha, linewidth=1, color='black')
    
    ax_a.text(
        300, 0.6,
        r'$\alpha^* \approx 0.79$',
        ha='center',
        fontsize=12
    )
    
    ax_a.set_ylabel(r'$\alpha$')
    ax_a.set_xlabel('Iterations')
    
    # -----------------------------------------------------
    # Common legend at the top
    # -----------------------------------------------------
    ax_leg.axis('off')
    
    ax_leg.legend(
        handles=[l1, l2, l3, l4, l6, l5],
        labels=[
            'Exact Integration',
            'Standard Monte-Carlo',
            'Conforming Monte-Carlo',
            r'Surrogate Gradient ($\beta=0.5$)',
            r'Surrogate Gradient ($\beta=0.25$)',
            r'Surrogate Gradient ($\beta=0.1$)'
        ],
        loc='center',
        ncol=2,
        frameon=False
    )
    
    # -----------------------------------------------------
    # Layout
    # -----------------------------------------------------
    plt.tight_layout()
    plt.savefig('Saved_Pics/all_parameters.pdf',
                dpi=300,
                bbox_inches='tight')
    plt.show()
    
    # =====================================================
    # Create 1x3 zoomed figure for the last 50 iterations
    # =====================================================
    
    fig_zoom = plt.figure(figsize=(9, 3))
    
    gs_zoom = fig_zoom.add_gridspec(
        nrows=2,
        ncols=3,
        height_ratios=[0, 1]
    )
    
    ax_leg_zoom = fig_zoom.add_subplot(gs_zoom[0, :])
    ax_b_zoom = fig_zoom.add_subplot(gs_zoom[1, 0])
    ax_c_zoom = fig_zoom.add_subplot(gs_zoom[1, 1])
    ax_a_zoom = fig_zoom.add_subplot(gs_zoom[1, 2])
    
    # We want the last 50 iterations
    last_50 = slice(-50, None)
    x_zoom = x[last_50]
    
    # -----------------------------------------------------
    # b plot zoom
    # -----------------------------------------------------
    l1_zoom, = ax_b_zoom.plot(x_zoom, Error_b[last_50], linewidth=2, linestyle='-', color=My_purple, label='Exact Integration')
    l2_zoom, = ax_b_zoom.plot(x_zoom, Error_b_MC[last_50], linewidth=1, linestyle='--', color='red', label='Standard Monte-Carlo')
    l3_zoom, = ax_b_zoom.plot(x_zoom, Error_b_CMC[last_50], linewidth=1, linestyle=':', color=My_Blue, label='Conforming Monte-Carlo')
    l4_zoom, = ax_b_zoom.plot(x_zoom, Error_b_FMC_05[last_50], linewidth=1, linestyle='-', marker='o', markersize=2.5, markevery=5, color=My_Green, label=r'Surrogate Gradient ($\beta=0.5$)')
    l5_zoom, = ax_b_zoom.plot(x_zoom, Error_b_FMC_01[last_50], linewidth=1, linestyle='--', marker='o', markersize=2.5, markevery=5, color=My_Orange, label=r'Surrogate Gradient ($\beta=0.1$)')
    l6_zoom, = ax_b_zoom.plot(x_zoom, Error_b_FMC_1[last_50], linewidth=1, linestyle='-.', marker='o', markersize=2.5, markevery=5, color='blue', label=r'Surrogate Gradient ($\beta=0.25$)')
    
    ax_b_zoom.plot(x_zoom, b[last_50], linewidth=2, color='black')
    
    ax_b_zoom.text(
        475, -0.5988 - 0.020,
        r'$b^*_1 \approx -0.60$',
        ha='center',
        va='bottom',
        fontsize=12
    )
    
    ax_b_zoom.set_ylabel(r'$b$')
    ax_b_zoom.set_xlabel('Iterations')
    
    # -----------------------------------------------------
    # c plot zoom
    # -----------------------------------------------------
    ax_c_zoom.plot(x_zoom, Error_c[last_50], linewidth=1, linestyle='-', color=My_purple)
    ax_c_zoom.plot(x_zoom, Error_c_MC[last_50], linewidth=1, linestyle='--', color='red')
    ax_c_zoom.plot(x_zoom, Error_c_CMC[last_50], linewidth=1, linestyle='-', color=My_Blue)
    ax_c_zoom.plot(x_zoom, Error_c_FMC_05[last_50], linewidth=1, linestyle='-', marker='o', markersize=2.5, markevery=5, color=My_Green)
    ax_c_zoom.plot(x_zoom, Error_c_FMC_01[last_50], linewidth=1, linestyle='--', marker='o', markersize=2.5, markevery=5, color=My_Orange)
    ax_c_zoom.plot(x_zoom, Error_c_FMC_1[last_50], linewidth=1, linestyle='-.', marker='o', markersize=2.5, markevery=5, color='blue')
    
    ax_c_zoom.plot(x_zoom, c[last_50], linewidth=1, color='black')
    
    ax_c_zoom.text(
        475, 7.4323 + 0.07,
        r'$c^*_1 \approx 7.43$',
        ha='center',
        va='bottom',
        fontsize=12
    )
    
    ax_c_zoom.set_ylabel(r'$c$')
    ax_c_zoom.set_xlabel('Iterations')
    
    # -----------------------------------------------------
    # alpha plot zoom
    # -----------------------------------------------------
    ax_a_zoom.plot(x_zoom, Error_a1[last_50], linewidth=1, linestyle='-', color=My_purple)
    ax_a_zoom.plot(x_zoom, Error_a1_MC[last_50], linewidth=1, linestyle='--', color='red')
    ax_a_zoom.plot(x_zoom, Error_a1_CMC[last_50], linewidth=1, linestyle=':', color=My_Blue)
    ax_a_zoom.plot(x_zoom, Error_a1_FMC_05[last_50], linewidth=1, linestyle='-', marker='o', markersize=2.5, markevery=5, color=My_Green)
    ax_a_zoom.plot(x_zoom, Error_a1_FMC_01[last_50], linewidth=1, linestyle='--', marker='o', markersize=2.5, markevery=5, color=My_Orange)
    ax_a_zoom.plot(x_zoom, Error_a1_FMC_1[last_50], linewidth=1, linestyle='-.', marker='o', markersize=2.5, markevery=5, color='blue')
    
    ax_a_zoom.plot(x_zoom, alpha[last_50], linewidth=1, color='black')
    
    ax_a_zoom.text(
        475, 0.7854 - 0.002,
        r'$\alpha^* \approx 0.79$',
        ha='center',
        va='bottom',
        fontsize=12
    )
    
    ax_a_zoom.set_ylabel(r'$\alpha$')
    ax_a_zoom.set_xlabel('Iterations')
    
    # -----------------------------------------------------
    # Common legend at the top
    # -----------------------------------------------------
    ax_leg_zoom.axis('off')
    
    # ax_leg_zoom.legend(
    #     handles=[l1_zoom, l2_zoom, l3_zoom, l4_zoom, l6_zoom, l5_zoom],
    #     labels=[
    #         'Exact Integration',
    #         'Standard Monte-Carlo',
    #         'Conforming Monte-Carlo',
    #         r'Surrogate Gradient ($\beta=0.5$)',
    #         r'Surrogate Gradient ($\beta=0.25$)',
    #         r'Surrogate Gradient ($\beta=0.1$)'
    #     ],
    #     loc='center',
    #     ncol=3,
    #     frameon=False
    # )
    
    # -----------------------------------------------------
    # Layout
    # -----------------------------------------------------
    plt.tight_layout()
    plt.savefig('Saved_Pics/all_parameters_zoom.pdf',
                dpi=300,
                bbox_inches='tight')
    plt.show()

plot_model_2D(Error_b,
Error_c,
Error_a1,
Error_b_MC,
Error_c_MC,
Error_a1_MC,
Error_b_CMC,
Error_c_CMC,
Error_a1_CMC,
Error_b_FMC_05,
Error_c_FMC_05,
Error_a1_FMC_05,
Error_b_FMC_01,
Error_c_FMC_01,
Error_a1_FMC_01,
Error_b_FMC_1,
Error_c_FMC_1,
Error_a1_FMC_1,
300,
N=100)