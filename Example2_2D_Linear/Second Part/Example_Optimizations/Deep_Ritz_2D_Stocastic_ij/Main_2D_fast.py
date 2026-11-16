"""
Main_2D_fast.py  —  Exact integration Deep Ritz (FB / NW / DNW), compiled version of Main_2D.py.

Same problem, settings, schedule, error measures, saved files and plots as Main_2D.py.
Integration: ../Fast_integration.py with the exact Gauss rule of get_exact_conical_quad.
The stiffness is moment-compressed from that rule (exact for its polynomial integrands); the load
uses a compressed rule of degree 15 (f_poly has total degree 10).  The updates are the explicit
neuron-wise ones of train_u_1D.py (no random selection; DNW load weighted 1/N per step),
or, with n_batched_1 / n_batched_2, random batches with one gradient and one update per epoch.
Output files get the suffix SUFFIX, so earlier results are not overwritten.
"""
import os, sys, time
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import tensorflow as tf
import keras
import scipy.io
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
PRECISION = 'float64'                    # same precision as the timing runs
keras.backend.set_floatx(PRECISION)      # before any network is built
keras.config.set_dtype_policy(PRECISION)  # set_floatx alone is ignored once any layer exists (e.g. in Spyder)
from Fast_integration import FastIntegrator, make_train_steps

from NN_Architecture_2D import net_u
from Integrate_Points import get_exact_conical_quad
from Data import exact_u, exact_dx, exact_dy
from Fit_2D_Poly_Stable import fit_2d_poly_stable
from Plot import plot_model_2D
keras.utils.set_random_seed(1234)

num_epochs = 60000
num_neurons_u = 20
num_layers = 1
activation = tf.nn.relu
initial_learning_rate = 1.e-3
final_learning_rate = 1.e-5
n_points = 10                 # Gauss points per direction (exact rule for the stiffness and the load)

# Method = 'FB'   # Full-Batched
Method = 'NW'   # Neuron-Wise
# Method = 'DNW'    # Double Neuron-Wise
n_batched_1 = 10                # NW / DNW: random set K1 of neurons k, redrawn every epoch
n_batched_2 = 10                # DNW: random set K2 of neurons k1, redrawn every epoch
                               # (one gradient and one update per epoch on the summed loss)

alpha = 0.0                   # reaction term coefficient
lambda_reg = 0.1               # interior regularisation weight
domain_vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
SUFFIX = '_fast_explicit'     # output suffix; earlier '_fast' results are not overwritten
TAG = {'FB': '', 'NW': '_S_i', 'DNW': '_S_ij'}     # same file names as Main_2D.py (+ SUFFIX)

f_degree = [5, 5]
f_coeffs = fit_2d_poly_stable(f_degree)

def f_poly(x, y):
    out = tf.zeros_like(x)
    for j in range(f_degree[0] + 1):
        for k in range(f_degree[1] + 1):
            out += f_coeffs[j, k] * x**j * y**k
    return out

pts, wts = get_exact_conical_quad(n_points)
integrator = FastIntegrator(num_neurons_u, domain_vertices, alpha, f_poly, pts, wts,
                            load_degree=f_degree[0] + f_degree[1] + 5)

# ════════════════════════════════════════════════════════════════════════════
#  Setup (compiled training epoch; see ../Fast_integration.py)
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
n_eval = 100
x_lin = tf.cast(tf.linspace(0.0, 1.0, n_eval), PRECISION)
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
