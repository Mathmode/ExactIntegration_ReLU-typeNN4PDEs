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
# def plot_model(net, Loss, exact, dx_errors, dy_errors, num_epochs):
    
#     xlist = np.expand_dims(np.array([1/1000 * i for i in range(1000)]),axis = 1)
#     dux_exact_test = exact_du(xlist)
#     ux_exact_test = exact_u(xlist)
#     xlist = tf.expand_dims(tf.linspace(0.0, 1.0, 1000), axis=1)
#     with tf.GradientTape(watch_accessed_variables=False) as t2:
#         t2.watch(xlist)
#         u_test = net(xlist)
#     dux_test = t2.gradient(u_test,xlist)
    
#     ## ---------
#     # solutions_u
#     ## ---------
#     # Generate a list of x values for visualization
#     xlist = np.expand_dims(np.array([1/1000 * i for i in range(1000)]),axis = 1)
    
#     font = {'family' : 'Times',
#             'weight' : 'normal',
#             'size'   : 15}
#     plt.rc('font', **font)
#     plt.figure(figsize=(4.5, 3.2))
#     plt.rcParams['text.usetex'] = True
    
#     plt.plot(xlist, u_test,'-', linewidth=1.5, label = 'Approximation')
#     plt.plot(xlist, ux_exact_test,'-', linewidth=1.5, label = 'Exact')
#     # plt.plot(Loss, color='r')
#     plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=False)
#     plt.xlabel('x')
#     plt.ylabel('u')
#     # plt.xticks([1, 10, 100, 1000, 10000], ['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])#, r"$\displaystyle 5 \cdot 10^3$"])
#     plt.subplots_adjust(left  =0.2,
#                         bottom=0.19,
#                         right =0.95,
#                         top   =0.85,
#                         wspace=0.1,
#                         hspace=0.2)
#     plt.savefig('Saved_Pics/solution.pdf',dpi=300)
#     plt.show()
    
#     ## ---------
#     # solutions_du
#     ## ---------
#     # Generate a list of x values for visualization
#     xlist = np.expand_dims(np.array([1/1000 * i for i in range(1000)]),axis = 1)
    
#     font = {'family' : 'Times',
#             'weight' : 'normal',
#             'size'   : 15}
#     plt.rc('font', **font)
#     plt.figure(figsize=(4.5, 3.2))
#     plt.rcParams['text.usetex'] = True
    
#     plt.plot(xlist, dux_test,'-', linewidth=1.5, label = 'Approximation')
#     plt.plot(xlist, dux_exact_test,'-', linewidth=1.5, label = 'Exact')
#     # plt.plot(Loss, color='r')
#     plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=False)
#     plt.xlabel('x')
#     plt.ylabel('du')
#     # plt.xticks([1, 10, 100, 1000, 10000], ['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])#, r"$\displaystyle 5 \cdot 10^3$"])
#     plt.subplots_adjust(left  =0.2,
#                         bottom=0.19,
#                         right =0.95,
#                         top   =0.85,
#                         wspace=0.1,
#                         hspace=0.2)
#     plt.savefig('Saved_Pics/d_solution.pdf',dpi=300)
#     plt.show()
    
    
#     ## ---------
#     # Error_du
#     ## ---------
#     # Generate a list of x values for visualization
#     xlist = np.expand_dims(np.array([1/1000 * i for i in range(1000)]),axis = 1)
    
#     font = {'family' : 'Times',
#             'weight' : 'normal',
#             'size'   : 15}
#     plt.rc('font', **font)
#     plt.figure(figsize=(4.5, 3.2))
#     plt.rcParams['text.usetex'] = True
    
#     plt.plot(xlist, tf.abs(dux_exact_test - dux_test.numpy()),'-', linewidth=1.5)

#     plt.xlabel('x')
#     plt.ylabel(r'$|u^\prime - u^\prime_\theta|$')

#     # plt.xticks([1, 10, 100, 1000, 10000], ['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])#, r"$\displaystyle 5 \cdot 10^3$"])
#     plt.subplots_adjust(left  =0.2,
#                         bottom=0.19,
#                         right =0.95,
#                         top   =0.85,
#                         wspace=0.1,
#                         hspace=0.2)
#     plt.savefig('Saved_Pics/error.pdf',dpi=300)
#     plt.show()
    
    
#     ## ---------
#     # Error_u
#     ## ---------
#     # Generate a list of x values for visualization
#     xlist = np.expand_dims(np.array([1/1000 * i for i in range(1000)]),axis = 1)
    
#     font = {'family' : 'Times',
#             'weight' : 'normal',
#             'size'   : 15}
#     plt.rc('font', **font)
#     plt.figure(figsize=(4.5, 3.2))
#     plt.rcParams['text.usetex'] = True
    
#     ue = tf.cast(exact(xlist), dtype=tf.float32)
#     u = tf.cast(net(xlist), dtype=tf.float32)
#     plt.plot(xlist, tf.abs(ue - u),'-', linewidth=1.5)

#     plt.xlabel('x')
#     plt.ylabel(r'$|u - u_\theta|$')
#     # plt.xticks([1, 10, 100, 1000, 10000], ['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])#, r"$\displaystyle 5 \cdot 10^3$"])
#     plt.subplots_adjust(left  =0.2,
#                         bottom=0.19,
#                         right =0.95,
#                         top   =0.85,
#                         wspace=0.1,
#                         hspace=0.2)
#     plt.savefig('Saved_Pics/derror.pdf',dpi=300)
#     plt.show()
    
#     ## ---------
#     # Loss plot
#     ## ---------
    
#     font = {'family' : 'Times',
#             'weight' : 'normal',
#             'size'   : 15}
#     plt.rc('font', **font)
#     plt.figure(figsize=(4.5, 3.2))
#     plt.rcParams['text.usetex'] = True
    
#     plt.plot(Loss,'-', linewidth=1.5, label = 'Training')
#     # plt.plot(Loss, color='r')
#     plt.title('')
#     plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=False)
#     plt.xlabel('Iterations')
#     plt.ylabel('Loss')
#     # plt.yscale('log')
#     # plt.xticks([1, 10, 100, 1000, 10000], ['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])#, r"$\displaystyle 5 \cdot 10^3$"])
#     plt.subplots_adjust(left  =0.2,
#                         bottom=0.19,
#                         right =0.95,
#                         top   =0.85,
#                         wspace=0.1,
#                         hspace=0.2)
#     plt.savefig('Saved_Pics/Loss_0.pdf',dpi=300)
#     plt.show()


#     ## ---------
#     # L2_error plot
#     ## ---------
    
#     font = {'family' : 'Times',
#             'weight' : 'normal',
#             'size'   : 15}
#     plt.rc('font', **font)
#     plt.figure(figsize=(4.5, 3.2))
#     plt.rcParams['text.usetex'] = True
    
#     plt.plot(errors,'-', linewidth=1.5)
#     # plt.plot(Loss, color='r')
#     plt.title('')
#     plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=False)
#     plt.xlabel('Iterations')
#     plt.ylabel(r'$||u - u_\theta||_{L^2}$')
#     plt.yscale('log')
#     plt.yticks([1, 0.1, 0.01, 0.001], ['$10^0$', '$10^{-1}$', '$10^{-2}$', '$10^{-3}$'])#, r"$\displaystyle 5 \cdot 10^3$"])
#     # plt.xticks([1, 10, 100, 1000, 10000], ['$10^0$', '$10^1$', '$10^2$', '$10^3$', '$10^4$'])#, r"$\displaystyle 5 \cdot 10^3$"])
#     plt.subplots_adjust(left  =0.2,
#                         bottom=0.19,
#                         right =0.95,
#                         top   =0.85,
#                         wspace=0.1,
#                         hspace=0.2)
#     plt.savefig('Saved_Pics/L2_error.pdf',dpi=300)
#     plt.show()
    
#     ## ---------
#     # L2_derror plot
#     ## ---------
    
#     font = {'family' : 'Times',
#             'weight' : 'normal',
#             'size'   : 15}
#     plt.rc('font', **font)
#     plt.figure(figsize=(4.5, 3.2))
#     plt.rcParams['text.usetex'] = True
    
#     plt.plot(derrors,'-', linewidth=1.5)
#     # plt.plot(Loss, color='r')
#     plt.title('')
#     plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.22), ncol=2, frameon=False)
#     plt.xlabel('Iterations')
#     plt.ylabel(r'$||u ^\prime- u^\prime_\theta||_{L^2}$')
#     plt.yscale('log')
#     plt.yticks([2, 1, 0.1], ['', '$10^0$', '$10^{-1}$'])#, r"$\displaystyle 5 \cdot 10^3$"])
#     plt.subplots_adjust(left  =0.2,
#                         bottom=0.19,
#                         right =0.95,
#                         top   =0.85,
#                         wspace=0.1,
#                         hspace=0.2)
#     plt.savefig('Saved_Pics/L2_derror.pdf',dpi=300)
#     plt.show()



import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

def plot_model_2D(
    net,
    Loss,
    exact_u,
    errors,
    derrors,
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
    # Create 2D grid
    # -------------------------------------------------
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.flatten(), Y.flatten()], axis=1)
    XY_tf = tf.convert_to_tensor(XY, dtype=tf.float32)

    # -------------------------------------------------
    # Network prediction + gradients
    # -------------------------------------------------
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(XY_tf)
        u_pred = net(XY_tf)

    du_pred = tape.gradient(u_pred, XY_tf)
    u_pred = u_pred.numpy().reshape(N, N)
    dux_pred = du_pred[:, 0].numpy().reshape(N, N)
    duy_pred = du_pred[:, 1].numpy().reshape(N, N)

    # -------------------------------------------------
    # Exact solution
    # -------------------------------------------------
    x = XY[:, 0]
    y = XY[:, 1]
    
    u_exact = exact_u(x, y).reshape(N, N)
    dux_exact = exact_dx(x, y).reshape(N, N)
    duy_exact = exact_dy(x, y).reshape(N, N)

    # -------------------------------------------------
    # Plot helper
    # -------------------------------------------------
    def contour_plot(Z, title, fname, cmap='viridis'):
        colors = [My_Orange, 'white', My_Blue]  # Blue -> Cyan -> Green -> Yellow -> Red
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors)

        plt.figure(figsize=(4.5, 3.6))
        plt.contourf(X, Y, Z, 100, cmap=cmap)
        plt.colorbar()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(fname, dpi=300)
        plt.show()

    # -------------------------------------------------
    # u(x,y)
    # -------------------------------------------------
    contour_plot(u_pred, r'$u_\theta(x,y)$', 'Saved_Pics/u_approx_2D.pdf')
    contour_plot(u_exact, r'$u(x,y)$', 'Saved_Pics/u_exact_2D.pdf')

    # -------------------------------------------------
    # Gradients
    # -------------------------------------------------
    contour_plot(dux_pred, r'$\partial_x u_\theta$', 'Saved_Pics/dux_approx_2D.pdf')
    contour_plot(duy_pred, r'$\partial_y u_\theta$', 'Saved_Pics/duy_approx_2D.pdf')

    contour_plot(dux_exact, r'$\partial_x u$', 'Saved_Pics/dux_exact_2D.pdf')
    contour_plot(duy_exact, r'$\partial_y u$', 'Saved_Pics/duy_exact_2D.pdf')

    # -------------------------------------------------
    # Pointwise errors
    # -------------------------------------------------
    contour_plot(np.abs(u_exact - u_pred),
                 r'$|u - u_\theta|$',
                 'Saved_Pics/error_u_2D.pdf',
                 cmap='magma')

    grad_error = np.sqrt((dux_exact - dux_pred)**2 +
                          (duy_exact - duy_pred)**2)

    contour_plot(grad_error,
                 r'$|\nabla u - \nabla u_\theta|$',
                 'Saved_Pics/error_du_2D.pdf',
                 cmap='magma')

    # -------------------------------------------------
    # Loss history
    # -------------------------------------------------
    plt.figure(figsize=(4.5, 3.2))
    plt.plot(Loss, linewidth=1.5, color=My_Blue)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('Saved_Pics/Loss_2D.pdf', dpi=300)
    plt.show()

    # -------------------------------------------------
    # L2 error history
    # -------------------------------------------------
    plt.figure(figsize=(4.5, 3.2))
    plt.plot(errors, linewidth=1., color=My_Blue)
    plt.xlabel('Iterations')
    plt.ylabel(r'$||u - u_\theta||_{L^2}$')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('Saved_Pics/L2_error_2D.pdf', dpi=300)
    plt.show()

    plt.figure(figsize=(4.5, 3.2))
    plt.plot(derrors, linewidth=1.5, color=My_Blue)
    plt.xlabel('Iterations')
    plt.ylabel(r'$||\nabla u - \nabla u_\theta||_{L^2}$')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('Saved_Pics/L2_derror_2D.pdf', dpi=300)
    plt.show()
