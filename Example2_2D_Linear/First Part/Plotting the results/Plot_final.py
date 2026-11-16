





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 16:04:42 2025

@author: sbahalouei
"""

import os
import importlib.util

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import keras
import scipy.io

from Data import f_rhs, exact_u, exact_dx, exact_dy
from NN_Architecture_2D import net_u

os.chdir(os.path.dirname(os.path.abspath(__file__)))


My_Blue = (0, 0/255, 102/255)
My_Orange = (255/255, 178/255, 102/255)
My_Green = (10/255, 102/255, 28/255)


# -------------------------------------------------
# Load training histories
# -------------------------------------------------
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


# -------------------------------------------------
# Load the specific architectures
# -------------------------------------------------
here = os.path.dirname(os.path.abspath(__file__))

# Fake_ReLu architecture
fr_arch_path = os.path.join(here, '../Deep_Ritz_2D_Fake_ReLu/NN_Architecture_2D.py')
spec_fr = importlib.util.spec_from_file_location("fr_arch", fr_arch_path)
fr_arch = importlib.util.module_from_spec(spec_fr)
spec_fr.loader.exec_module(fr_arch)

# CMC architecture
cmc_arch_path = os.path.join(here, '../Deep_Ritz_2D_CMC/NN_Architecture_2D.py')
spec_cmc = importlib.util.spec_from_file_location("cmc_arch", cmc_arch_path)
cmc_arch = importlib.util.module_from_spec(spec_cmc)
spec_cmc.loader.exec_module(cmc_arch)


def load_with_registry_override(filepath, class_map):
    for k, v in class_map.items():
        keras.saving.get_custom_objects()[k] = v
        keras.saving.get_custom_objects()[f"Custom>{k}"] = v
    return keras.models.load_model(filepath, compile=False)


# u_net_MC_i = load_with_registry_override('Models/u_net_S_i_fast.keras', {'net_u': net_u})
# u_net_MC_ij = load_with_registry_override('Models/u_net_S_ij_fast.keras', {'net_u': net_u})
u_net_MC = load_with_registry_override('Models/u_net_CMC_fast_explicit.keras', {'net_u': net_u})
u_net = load_with_registry_override('Models/u_net_fast_explicit.keras', {'net_u': net_u})

u_net_CMC = load_with_registry_override('Models/u_net_CMC_fast_explicit.keras', {
    'FanInitializer': cmc_arch.FanInitializer,
    'net_u': cmc_arch.net_u
})

u_net_FR = load_with_registry_override('Models/u_net_FR.keras', {
    'my_relu': fr_arch.my_relu,
    'my_heaviside': fr_arch.my_heaviside,
    'AnalyticalReLULayer': fr_arch.AnalyticalReLULayer,
    'net_u': fr_arch.net_u
})


# -------------------------------------------------
# Evaluate a model and its gradient in the model's own dtype
# -------------------------------------------------
def predict_with_grad(model, XY, N):
    """Return u, du/dx, du/dy on the N x N grid as float64 numpy arrays.

    Some saved models have a float64 outer policy but float32 Dense layers,
    so Keras casts the input to float64 and `out * bc_mask` mixes dtypes.
    Align the outer policy with the inner layers before evaluating.
    """
    inner_dtype = model.weights[-1].dtype if model.weights else 'float32'
    if model.dtype_policy.compute_dtype != inner_dtype:
        model.dtype_policy = keras.DTypePolicy(inner_dtype)

    XY_tf = tf.convert_to_tensor(XY, dtype=inner_dtype)
    with tf.GradientTape() as tape:
        tape.watch(XY_tf)
        u = model(XY_tf)
    du = tape.gradient(u, XY_tf)

    u = tf.cast(u, tf.float64).numpy().reshape(N, N)
    du = tf.cast(du, tf.float64).numpy()
    return u, du[:, 0].reshape(N, N), du[:, 1].reshape(N, N)


def plot_model_2D(
    u_net,
    u_net_MC,
    u_net_CMC,
    u_net_FR,
    errors,
    derrors,
    errors_MC_i,
    derrors_MC_i,
    errors_MC_ij,
    derrors_MC_ij,
    errors_MC,
    derrors_MC,
    errors_CMC,
    derrors_CMC,
    errors_FR,
    derrors_FR,
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

    # -------------------------------------------------
    # Network predictions + gradients
    # -------------------------------------------------
    u_pred,       dux_pred,       duy_pred       = predict_with_grad(u_net,       XY, N)
    # u_pred_MC_i,  dux_pred_MC_i,  duy_pred_MC_i  = predict_with_grad(u_net_MC_i,  XY, N)
    # u_pred_MC_ij, dux_pred_MC_ij, duy_pred_MC_ij = predict_with_grad(u_net_MC_ij, XY, N)
    u_pred_MC,    dux_pred_MC,    duy_pred_MC    = predict_with_grad(u_net_MC,    XY, N)
    u_pred_CMC,   dux_pred_CMC,   duy_pred_CMC   = predict_with_grad(u_net_CMC,   XY, N)
    u_pred_FR,    dux_pred_FR,    duy_pred_FR    = predict_with_grad(u_net_FR,    XY, N)

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
        colors = [My_Orange, 'white', My_Blue]
        cmap = mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors)
        font = {'family': 'Times',
                'weight': 'normal',
                'size': 13}
        plt.rc('font', **font)
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{mathrsfs} \usepackage{amssymb}'

        plt.figure(figsize=(4.8, 4))
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
    # contour_plot(u_pred, r'$u_\theta(x,y)$', 'Saved_Pics/u_approx_2D.pdf')
    # contour_plot(u_pred_MC_i, r'$u_\theta(x,y)$', 'Saved_Pics/u_approx_2D_MC_i.pdf')
    # contour_plot(u_pred_MC_ij, r'$u_\theta(x,y)$', 'Saved_Pics/u_approx_2D_MC_ij.pdf')
    # contour_plot(u_exact, r'$u(x,y)$', 'Saved_Pics/u_exact_2D.pdf')

    # -------------------------------------------------
    # Gradients
    # -------------------------------------------------
    # contour_plot(dux_pred, r'$\partial_x u_\theta$', 'Saved_Pics/dux_approx_2D.pdf')
    # contour_plot(duy_pred, r'$\partial_y u_\theta$', 'Saved_Pics/duy_approx_2D.pdf')

    # contour_plot(dux_pred_MC_i, r'$\partial_x u_\theta$', 'Saved_Pics/dux_approx_2D_MC_i.pdf')
    # contour_plot(duy_pred_MC_i, r'$\partial_y u_\theta$', 'Saved_Pics/duy_approx_2D_MC_i.pdf')

    # contour_plot(dux_pred_MC_ij, r'$\partial_x u_\theta$', 'Saved_Pics/dux_approx_2D_MC_ij.pdf')
    # contour_plot(duy_pred_MC_ij, r'$\partial_y u_\theta$', 'Saved_Pics/duy_approx_2D_MC_ij.pdf')

    # contour_plot(dux_exact, r'$\partial_x u$', 'Saved_Pics/dux_exact_2D.pdf')
    # contour_plot(duy_exact, r'$\partial_y u$', 'Saved_Pics/duy_exact_2D.pdf')

    # contour_plot(dux_pred_MC, r'$\partial_x u$', 'Saved_Pics/dux_2D_MC.pdf')
    # contour_plot(duy_pred_MC, r'$\partial_y u$', 'Saved_Pics/duy_2D_MC.pdf')

    # contour_plot(dux_pred_CMC, r'$\partial_x u$', 'Saved_Pics/dux_2D_CMC.pdf')
    # contour_plot(duy_pred_CMC, r'$\partial_y u$', 'Saved_Pics/duy_2D_CMC.pdf')

    # contour_plot(dux_pred_FR, r'$\partial_x u$', 'Saved_Pics/dux_2D_FR.pdf')
    # contour_plot(duy_pred_FR, r'$\partial_y u$', 'Saved_Pics/duy_2D_FR.pdf')

    # -------------------------------------------------
    # Pointwise errors
    # -------------------------------------------------
    # contour_plot(np.abs(u_exact - u_pred), r'$|u - u_\theta|$',
    #              'Saved_Pics/error_u_2D.pdf', cmap='magma')
    # contour_plot(np.abs(u_exact - u_pred_MC_i), r'$|u - u_\theta|$',
    #              'Saved_Pics/error_u_2D_MC_i.pdf', cmap='magma')
    # contour_plot(np.abs(u_exact - u_pred_MC_ij), r'$|u - u_\theta|$',
    #              'Saved_Pics/error_u_2D_MC_ij.pdf', cmap='magma')
    # contour_plot(np.abs(u_exact - u_pred_MC), r'$|u - u_\theta|$',
    #              'Saved_Pics/error_u_2D_MC.pdf', cmap='magma')
    # contour_plot(np.abs(u_exact - u_pred_CMC), r'$|u - u_\theta|$',
    #              'Saved_Pics/error_u_2D_CMC.pdf', cmap='magma')
    # contour_plot(np.abs(u_exact - u_pred_FR), r'$|u - u_\theta|$',
    #              'Saved_Pics/error_u_2D_FR.pdf', cmap='magma')

    grad_error = np.sqrt((dux_exact - dux_pred)**2 +
                         (duy_exact - duy_pred)**2)

    # grad_error_MC_i = np.sqrt((dux_exact - dux_pred_MC_i)**2 +
    #                           (duy_exact - duy_pred_MC_i)**2)

    # grad_error_MC_ij = np.sqrt((dux_exact - dux_pred_MC_ij)**2 +
    #                            (duy_exact - duy_pred_MC_ij)**2)

    grad_error_MC = np.sqrt((dux_exact - dux_pred_MC)**2 +
                            (duy_exact - duy_pred_MC)**2)

    grad_error_CMC = np.sqrt((dux_exact - dux_pred_CMC)**2 +
                             (duy_exact - duy_pred_CMC)**2)

    grad_error_FR = np.sqrt((dux_exact - dux_pred_FR)**2 +
                            (duy_exact - duy_pred_FR)**2)

    # contour_plot(grad_error, r'$|\nabla u - \nabla u_\theta|$',
    #              'Saved_Pics/error_du_2D.pdf', cmap='magma')
    # contour_plot(grad_error_MC_i, r'$|\nabla u - \nabla u_\theta|$',
    #              'Saved_Pics/error_du_2D_MC_i.pdf', cmap='magma')
    # contour_plot(grad_error_MC_ij, r'$|\nabla u - \nabla u_\theta|$',
    #              'Saved_Pics/error_du_2D_MC_ij.pdf', cmap='magma')
    # contour_plot(grad_error_MC, r'$|\nabla u - \nabla u_\theta|$',
    #              'Saved_Pics/error_du_2D_MC.pdf', cmap='magma')
    # contour_plot(grad_error_CMC, r'$|\nabla u - \nabla u_\theta|$',
    #              'Saved_Pics/error_du_2D_CMC.pdf', cmap='magma')
    # contour_plot(grad_error_FR, r'$|\nabla u - \nabla u_\theta|$',
    #              'Saved_Pics/error_du_2D_FR.pdf', cmap='magma')

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

    # -------------------------------------------------
    # Relative error histories
    # -------------------------------------------------
    font = {'family': 'Times',
            'weight': 'normal',
            'size': 16}
    plt.rc('font', **font)
    plt.rcParams['text.usetex'] = True
    plt.rcParams['text.latex.preamble'] = r'\usepackage{mathrsfs} \usepackage{amssymb}'

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    # Increase `left` (e.g. 0.15) if the y-label gets cropped
    fig.subplots_adjust(left=0.12, bottom=0.27, right=0.98, top=0.8, wspace=0.3, hspace=0.1)

    # Error in u
    axes[0].plot(errors, linewidth=1.5, color=My_Blue, label='Exact Integration')
    # axes[0].plot(errors_MC_i, linewidth=1.5, color=My_Orange, label='Neuron-Wise')
    # axes[0].plot(errors_MC_ij, linewidth=1.5, color=My_Green, label='Double Neuron-Wise')
    axes[0].plot(errors_MC, linewidth=1.5, color='red', label='Standard Monte Carlo')
    axes[0].plot(errors_CMC, linewidth=1.5, color=My_Orange, label='Conforming Monte Carlo')
    axes[0].plot(errors_FR, linewidth=1.5, color=My_Green, label=r'Surrogate gradient ($\beta=3$)')
    axes[0].set_xlabel('Iterations')
    axes[0].set_ylabel(r'Relative error of $u_\theta$ (in \%)')
    axes[0].set_yscale('log')
    # axes[0].set_xscale('log')
    # axes[0].set_yticks([0.01, 1, 10, 100, 1000])
    # axes[0].set_yticklabels([r'$0.01$', r'$1$', r'$10$', r'$100$', r'$1000$'])
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # Error in grad u
    axes[1].plot(derrors, linewidth=1.5, color=My_Blue, label='Exact Integration')
    # axes[1].plot(derrors_MC_i, linewidth=1.5, color=My_Orange, label='Neuron-Wise')
    # axes[1].plot(derrors_MC_ij, linewidth=1.5, color=My_Green, label='Double Neuron-Wise')
    axes[1].plot(derrors_MC, linewidth=1.5, color='red', label='Standard Monte Carlo')
    axes[1].plot(derrors_CMC, linewidth=1.5, color=My_Orange, label='Conforming Monte Carlo')
    axes[1].plot(derrors_FR, linewidth=1.5, color=My_Green, label=r'Surrogate gradient ($\beta=3$)')
    axes[1].set_xlabel('Iterations')
    axes[1].set_ylabel(r'Relative error of $\nabla u_\theta$ (in \%)')
    axes[1].set_yscale('log')
    # axes[1].set_xscale('log')
    axes[1].grid(True, linestyle='--', alpha=0.7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2,
               bbox_to_anchor=(0.5, 0.8), fontsize=17, frameon=False)
    plt.savefig('Saved_Pics/L2_derror_2D.pdf', dpi=300)
    plt.show()


plot_model_2D(
    u_net,
    u_net_MC,
    u_net_CMC,
    u_net_FR,
    errors,
    d_errors,
    errors_MC_i,
    d_errors_MC_i,
    errors_MC_ij,
    d_errors_MC_ij,
    errors_MC,
    d_errors_MC,
    errors_CMC,
    d_errors_CMC,
    errors_FR,
    d_errors_FR,
    300,
    N=100
)