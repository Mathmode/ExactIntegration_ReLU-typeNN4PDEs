"""
Main_2D_fast.py  —  Conforming Monte Carlo Deep Ritz, compiled version of Main_2D.py.

Same problem, settings, schedule, error measures, saved files and plots as Main_2D.py.
The training updates are those of train_u_1D.py (FB, NW, DNW with the explicit neuron
interactions of My_integration.py), compiled in ../Fast_integration.py.
Output files get the suffix SUFFIX, so earlier results are not overwritten.
"""
import os, sys, time
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import tensorflow as tf
import keras
import scipy.io
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PRECISION = 'float64'                      # Main_2D.py works in float32 (the integrals are always float64)
keras.backend.set_floatx(PRECISION)        # before any network is built
keras.config.set_dtype_policy(PRECISION)   # set_floatx alone is ignored once any layer exists (e.g. in Spyder)
from Fast_integration import FastIntegrator, make_train_steps

from NN_Architecture_2D import net_u
from Integrate_Points import get_MC_quad
from Data import exact_u, f_rhs, exact_dx, exact_dy
from Plot import plot_model_2D
keras.utils.set_random_seed(1234)

num_epochs = 30000
num_neurons_u = 50
num_layers = 1
activation = tf.nn.relu
initial_learning_rate = 1.e-3
final_learning_rate = 1.e-4
n_MC = 1000                    # random points in the reference triangle, redrawn every epoch (as Main_2D.py)

Method = 'FB'    # Full-Batched
# Method = 'NW'    # Neuron-Wise
# Method = 'DNW'     # Double Neuron-Wise

alpha = 0.0                    # reaction term coefficient
lambda_reg = 0.1               # train_u_1D.py default
domain_vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
SUFFIX = '_fast_explicit'      # output suffix; earlier '_fast' results are not overwritten
TAG = {'FB': '_CMC', 'NW': '_CMC_i', 'DNW': '_CMC_ij'}
n_batched_1 = 20                # NW / DNW: random set K1 of neurons k, redrawn every epoch
n_batched_2 = 20                # DNW: random set K2 of neurons k1, redrawn every epoch
                               # (one gradient and one update per epoch on the summed loss)   # same file names as Main_2D.py (+ SUFFIX)

pts, wts = get_MC_quad(n_MC)
integrator = FastIntegrator(num_neurons_u, domain_vertices, alpha, f_rhs, pts, wts,
                            load_degree=None)   # f_rhs is a polynomial of degree 4: exact compressed load

# ════════════════════════════════════════════════════════════════════════════
#  Setup (compiled training steps; see ../Fast_integration.py)
# ════════════════════════════════════════════════════════════════════════════
script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(script_dir, 'Saved_Models'), exist_ok=True)
os.makedirs('Saved_Pics', exist_ok=True)          # Plot.py saves to ./Saved_Pics

u_net = net_u(num_layers, num_neurons_u, activation=activation)
u_net(tf.random.uniform(shape=[1, 2], minval=0., maxval=1., dtype=PRECISION))
u_net.save(os.path.join(script_dir, f'Saved_Models/u_net_initialize{SUFFIX}.keras'))
u_net.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
optimizer.build(u_net.trainable_variables)
train_epoch = make_train_steps(u_net, optimizer, integrator, Method, lambda_reg,
                               n_batched_1=n_batched_1, n_batched_2=n_batched_2)

# ---- error evaluation grid (same as Main_2D.py) ---------------------------
n_points = 100
x_lin = tf.cast(tf.linspace(0.0, 1.0, n_points), PRECISION)
X, Y = tf.meshgrid(x_lin, x_lin, indexing='ij')
x_eval_tf = tf.stack([tf.reshape(X, [-1]), tf.reshape(Y, [-1])], axis=1)
u_exact = exact_u(x_eval_tf[:, 0], x_eval_tf[:, 1])
dux_exact = exact_dx(x_eval_tf[:, 0], x_eval_tf[:, 1])
duy_exact = exact_dy(x_eval_tf[:, 0], x_eval_tf[:, 1])
u_norm = tf.sqrt(tf.reduce_mean(tf.square(u_exact)))
du_norm = tf.sqrt(tf.reduce_mean(tf.square(dux_exact) + tf.square(duy_exact)))

@tf.function
def eval_errors():
    """Relative L2 errors (%) of u and grad u — same formulas as Main_2D.py."""
    with tf.GradientTape() as t2:
        t2.watch(x_eval_tf)
        u_pred = u_net(x_eval_tf)
    grad = t2.gradient(u_pred, x_eval_tf)
    u_pred = tf.squeeze(u_pred)
    err = tf.sqrt(tf.reduce_mean(tf.square(u_exact - u_pred))) / u_norm * 100
    derr = tf.sqrt(tf.reduce_mean(tf.square(dux_exact - grad[:, 0]) +
                                  tf.square(duy_exact - grad[:, 1]))) / du_norm * 100
    return err, derr

def lr_schedule(epoch, initial_learning_rate, final_learning_rate, num_epochs):
    """Same exponential decay as train_u_1D.lr_schedule, in plain Python (the original is a
    @tf.function called with a Python int, which re-traces at every epoch)."""
    decay_rate = (final_learning_rate / initial_learning_rate) ** (1 / num_epochs)
    return initial_learning_rate * decay_rate ** epoch

# ════════════════════════════════════════════════════════════════════════════
#  Training loop (same structure as Main_2D.py)
# ════════════════════════════════════════════════════════════════════════════
Loss, errors, d_errors = [], [], []
start_time = time.time()
for epoch in range(num_epochs):
    optimizer.learning_rate.assign(lr_schedule(epoch, initial_learning_rate, final_learning_rate, num_epochs))
    pts, wts = get_MC_quad(n_MC)           # new MC points every epoch, as in Main_2D.py
    integrator.set_points(pts, wts)
    loss = train_epoch()
    err, derr = eval_errors()

    Loss.append(float(loss))
    errors.append(float(err))
    d_errors.append(float(derr))

    if epoch % 100 == 0:
        print('===========================================')
        print('=============', 'Epoch:', epoch, '/', num_epochs, '=============')
        print('Loss_training:', Loss[-1])
        print('L2_Error:', errors[-1])
        print('L2_dError:', d_errors[-1])

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time} seconds")

model_path = os.path.join(script_dir, f'Saved_Models/u_net{TAG[Method]}{SUFFIX}.keras')
u_net.save(model_path)
scipy.io.savemat(os.path.join(script_dir, f'Saved_Models/Data_history{TAG[Method]}{SUFFIX}.mat'), {
    'elapsed_time': elapsed_time,
    'num_epochs': num_epochs,
    'history': Loss,
    'errors': errors,
    'd_errors': d_errors,
})
plot_model_2D(u_net, Loss, exact_u, errors, d_errors, num_epochs)
# (the trained model is already in memory; reload it later with keras.models.load_model(model_path))
