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

My_Blue=(0, 0/255, 102/255)
#My_Green = (8/255,8/255,137/255)
My_Green = (0/255,0/255,102/255)
My_Orange = (255/255, 178/255, 102/255)
My_Green_Light = (146/255, 220/255, 181/255)


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_model_2D(
    net,
    Loss,
    weights_b, 
    weights_c,
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
    font = {'family' : 'Times',
                'weight' : 'normal',
                'size'   : 15}
    plt.rc('font', **font)
    plt.figure(figsize=(4.5, 3.2))
    plt.rcParams['text.usetex'] = True
    
    
    x = np.linspace(0, num_epochs, num_epochs)
    y1 = np.ones_like(x)*(-0.8459)
    y2 = np.ones_like(x)*5.2676
    
    plt.figure(figsize=(4.5, 3.2))
    plt.plot(weights_b, linewidth=1.5, color=My_Blue)
    plt.plot(tf.squeeze(weights_c), linewidth=1.5, color=My_Orange)
    plt.plot(x, y1, '--', color='black')
    plt.plot(x, y2, '--', color='black')
    plt.text(num_epochs -10, 0, r'$b^* = -0.8459$', horizontalalignment='center', fontsize=12)
    plt.text(num_epochs-10, 4., r'$c^* = 5.2676$', horizontalalignment='center', fontsize=12)
    plt.xlabel('Iterations')
    plt.ylabel('Values of c and b')
    # plt.yscale('log')
    plt.tight_layout()
    plt.savefig('Saved_Pics/b-c.pdf', dpi=300)
    plt.show()

    # -------------------------------------------------
    # Loss history
    # -------------------------------------------------
    # plt.figure(figsize=(4.5, 3.2))
    # plt.plot(Loss, linewidth=1.5, color=My_Blue)
    # plt.xlabel('Iterations')
    # plt.ylabel('Loss')
    # plt.yscale('log')
    # plt.tight_layout()
    # plt.savefig('Saved_Pics/Loss_2D.pdf', dpi=300)
    # plt.show()
