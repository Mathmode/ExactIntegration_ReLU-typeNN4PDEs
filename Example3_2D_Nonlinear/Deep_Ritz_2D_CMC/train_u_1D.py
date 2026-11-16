import tensorflow as tf
import numpy as np
from My_integration import my_integration_FB, my_integration_NW, my_integration_DNW
from Data import f_rhs


@tf.function
def lr_schedule(epoch, initial_learning_rate, final_learning_rate, num_epochs):
    decay_rate = (final_learning_rate / initial_learning_rate) ** (1 / num_epochs)
    current_learning_rate = initial_learning_rate * (decay_rate ** epoch)
    return current_learning_rate


def interior_regularization(u_net, num_neurons_u, lambda_reg=0.1):
    if tf.equal(lambda_reg, 0.0):
        return tf.constant(0.0, dtype=tf.float64)

    weights_1 = u_net.layers[0].weights[0]  # shape (2, num_neurons_u)
    biases_1 = u_net.layers[0].weights[1]   # shape (num_neurons_u,)

    w0 = tf.cast(weights_1[0, :], tf.float64)
    w1 = tf.cast(weights_1[1, :], tf.float64)
    b = tf.cast(biases_1, tf.float64)

    norm_w = tf.sqrt(tf.square(w0) + tf.square(w1) + 1e-12)

    # Calculate the pre-activation Z at the 4 corners of the [0, 1]x[0, 1] domain
    v1 = b
    v2 = w0 + b
    v3 = w1 + b
    v4 = w0 + w1 + b

    # Stack to shape (4, num_neurons)
    corners_Z = tf.stack([v1, v2, v3, v4], axis=0)

    # Normalize the distances so the penalty is geometric and scale-invariant
    corners_Z = corners_Z / norm_w

    # Get the maximum and minimum values at the corners for each neuron
    max_Z = tf.reduce_max(corners_Z, axis=0)
    min_Z = tf.reduce_min(corners_Z, axis=0)

    margin = 0.05
    penalty_dead   = tf.square(tf.nn.relu(-max_Z + margin))
    penalty_linear = tf.square(tf.nn.relu(min_Z + margin))

    reg_loss = tf.reduce_mean(penalty_dead + penalty_linear)
    return tf.cast(lambda_reg, tf.float64) * reg_loss


# ─── CRITICAL FIX: No @tf.function on any train_step or loss function ────────
# With @tf.function, TF AutoGraph traces through my_integration_* which calls
# clip_polygon_with_line_diff — a function with data-dependent output shape
# (tf.boolean_mask).  This causes N=128 to fail.  Running these eagerly lets
# the inner @tf.function kernels (clip_polygon, compute_MC_integrals_*, etc.)
# still compile fast while outer Python control flow stays dynamic.

def train_step_FB(u_net, optimizer, num_neurons_u,
                  domain_vertices, batch_size, pts, wts, alpha, lambda_reg=0.1):
    loss_total = tf.constant(0.0, dtype=tf.float64)
    with tf.GradientTape() as tape:
        out = loss_FB(u_net, num_neurons_u, domain_vertices, pts, wts, alpha, lambda_reg)
        loss_total += tf.cast(out, tf.float64)
    g = tape.gradient(out, u_net.trainable_variables)
    if g is not None and all(grad is not None for grad in g):
        optimizer.apply_gradients(zip(g, u_net.trainable_variables))
    return loss_total


# def train_step_NW(u_net, optimizer, num_neurons_u,
#                   domain_vertices, batch_size, pts, wts, new_learning_rate,
#                   alpha, lambda_reg=0.1):
#     # Sample i-indices ONCE before the tape; all invariants computed once inside
#     num_to_sample_i = tf.random.uniform(shape=[], minval=1,
#                                         maxval=num_neurons_u + 1, dtype=tf.int32)
#     sampled_i_np = tf.random.shuffle(tf.range(num_neurons_u))[:num_to_sample_i].numpy()
#     with tf.GradientTape() as tape:
#         du_du, u_f = my_integration_NW(u_net, num_neurons_u, domain_vertices,
#                                         sampled_i_np, pts, wts, alpha)
#         reg = interior_regularization(u_net, num_neurons_u, lambda_reg)
#         loss_total = (tf.cast(du_du, tf.float64) / tf.cast(2.0, tf.float64)
#                       - tf.cast(u_f, tf.float64) + reg)
#     g = tape.gradient(loss_total, u_net.trainable_variables)
#     if g is not None and all(grad is not None for grad in g):
#         optimizer.apply_gradients(zip(g, u_net.trainable_variables))
#     return loss_total

def train_step_NW(u_net, optimizer, num_neurons_u,
                  domain_vertices, batch_size, pts, wts, new_learning_rate,
                  alpha, lambda_reg=0.1):
    # Sample i-indices ONCE before the tape; all invariants computed once inside
    num_to_sample_i = tf.random.uniform(shape=[], minval=1,
                                        maxval=num_neurons_u + 1, dtype=tf.int32)
    sampled_i_np = tf.random.shuffle(tf.range(num_neurons_u))[:num_to_sample_i].numpy()
    loss_total =0
    for k in range(0, num_neurons_u):
        with tf.GradientTape() as tape:
            du_du, u_f = my_integration_NW(u_net, num_neurons_u, domain_vertices,
                                            sampled_i_np, pts, wts, alpha, k)
            reg = interior_regularization(u_net, num_neurons_u, lambda_reg)
            loss = (tf.cast(du_du, tf.float64) / tf.cast(2.0, tf.float64)
                          - tf.cast(u_f, tf.float64) + reg)
            loss_total = loss_total + loss
        g = tape.gradient(loss, u_net.trainable_variables)
        if g is not None and all(grad is not None for grad in g):
            optimizer.apply_gradients(zip(g, u_net.trainable_variables))
    return loss_total


# def train_step_DNW(u_net, optimizer, num_neurons_u,
#                    domain_vertices, batch_size, pts, wts, new_learning_rate,
#                    alpha, lambda_reg=0.1):
#     # Sample i-indices ONCE before the tape; all invariants computed once inside
#     num_to_sample_i = tf.random.uniform(shape=[], minval=1,
#                                         maxval=num_neurons_u + 1, dtype=tf.int32)
#     sampled_i_np = tf.random.shuffle(tf.range(num_neurons_u))[:num_to_sample_i].numpy()
#     with tf.GradientTape() as tape:
#         du_du, u_f, n = my_integration_DNW(u_net, num_neurons_u, domain_vertices,
#                                             sampled_i_np, pts, wts, alpha)
#         reg = interior_regularization(u_net, num_neurons_u, lambda_reg)
#         loss_total = (tf.cast(du_du, tf.float64) / tf.cast(2.0, tf.float64)
#                       - tf.cast(u_f, tf.float64) + reg)
#     g = tape.gradient(loss_total, u_net.trainable_variables)
#     if g is not None and all(grad is not None for grad in g):
#         optimizer.apply_gradients(zip(g, u_net.trainable_variables))
#     return loss_total



def train_step_DNW(u_net, optimizer, num_neurons_u,
                   domain_vertices, batch_size, pts, wts, new_learning_rate,
                   alpha, lambda_reg=0.1):
    # Sample i-indices ONCE before the tape; all invariants computed once inside
    num_to_sample_i = tf.random.uniform(shape=[], minval=1,
                                        maxval=num_neurons_u + 1, dtype=tf.int32)
    sampled_i_np = tf.random.shuffle(tf.range(num_neurons_u))[:num_to_sample_i].numpy()
    loss_total =0
    for k in range(0, num_neurons_u):
        for k1 in range(0, num_neurons_u):
            with tf.GradientTape() as tape:
                du_du, u_f, n = my_integration_DNW(u_net, num_neurons_u, domain_vertices,
                                                    sampled_i_np, pts, wts, alpha, k, k1)
                reg = interior_regularization(u_net, num_neurons_u, lambda_reg)
                loss=  (tf.cast(du_du, tf.float64) / tf.cast(2.0, tf.float64)
                              - tf.cast(u_f, tf.float64) + reg)
                loss_total = loss_total + loss
            g = tape.gradient(loss, u_net.trainable_variables)
            if g is not None and all(grad is not None for grad in g):
                optimizer.apply_gradients(zip(g, u_net.trainable_variables))
    return loss_total


def loss_FB(u_net, num_neurons_u, domain_vertices, pts, wts, alpha, lambda_reg=0.1):
    du_du, u_f = my_integration_FB(u_net, num_neurons_u, domain_vertices, pts, wts, alpha)
    target_dtype = tf.float64
    du_du = tf.cast(du_du, target_dtype)
    u_f   = tf.cast(u_f,   target_dtype)
    reg   = interior_regularization(u_net, num_neurons_u, lambda_reg)
    return du_du / tf.cast(2.0, target_dtype) - tf.cast(u_f, target_dtype) + reg


