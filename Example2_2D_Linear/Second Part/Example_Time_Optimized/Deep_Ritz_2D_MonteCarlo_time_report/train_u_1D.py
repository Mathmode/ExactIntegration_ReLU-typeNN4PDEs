#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimized Standard Monte Carlo training step for the Deep Ritz method.

Key optimizations over the original:
  1. Single forward pass: u_net(xy) is called only once, inside nested tapes,
     so both ∇_x u (spatial gradient) and ∇_θ loss (weight gradient) are
     computed in the same graph — eliminating the second redundant forward pass.
  2. f_rhs is NOT evaluated here; the caller passes f_val_fixed (precomputed
     once before the training loop), saving one expensive Python function call
     and its associated tf.function re-tracing per epoch.
"""

import tensorflow as tf


@tf.function
def train_step(u_net, f_val_fixed, optimizer, xy):
    """
    One gradient-descent step for the Deep Ritz loss

        L(u) = (1/2) * ∫|∇u|² dΩ  −  ∫f·u dΩ

    approximated by Monte Carlo:

        L ≈ (1/2) * mean(||∇u(xᵢ)||²)  −  mean(f(xᵢ) · u(xᵢ))

    Parameters
    ----------
    u_net       : tf.keras.Model  — the network (boundary mask already applied inside)
    f_val_fixed : tf.Tensor, shape (n_MC², 1)  — f(xᵢ) precomputed outside the loop
    optimizer   : tf.keras.optimizers.Optimizer
    xy          : tf.Tensor, shape (n_MC², 2)  — fixed collocation grid

    Returns
    -------
    loss : scalar tf.Tensor
    """
    xy          = tf.cast(xy,          tf.float32)
    f_val_fixed = tf.cast(f_val_fixed, tf.float32)

    with tf.GradientTape() as outer_tape:
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(xy)
            u = u_net(xy)                               # shape (n, 1)  — single forward pass

        grad_u = inner_tape.gradient(u, xy)             # shape (n, 2)  ∂u/∂(x,y)

        # ∫|∇u|² ≈ mean(||∇u||²)
        integral_grad = tf.reduce_mean(
            tf.reduce_sum(tf.square(grad_u), axis=1)    # shape (n,)
        )

        # ∫f·u ≈ mean(f · u)  — reuses the u already computed above
        integral_uf = tf.reduce_mean(f_val_fixed * u)

        loss = 0.5 * integral_grad - integral_uf

    grads = outer_tape.gradient(loss, u_net.trainable_variables)
    optimizer.apply_gradients(zip(grads, u_net.trainable_variables))
    return loss


@tf.function
def lr_schedule(epoch, initial_learning_rate, final_learning_rate, num_epochs):
    decay_rate = (final_learning_rate / initial_learning_rate) ** (1 / num_epochs)
    current_learning_rate = initial_learning_rate * (decay_rate ** epoch)
    return current_learning_rate
