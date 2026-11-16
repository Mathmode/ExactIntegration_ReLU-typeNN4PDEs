"""
Main_2D_fast.py  —  Conforming Monte Carlo Deep Ritz (FB / NW / DNW), fast version.

Same problem, settings, schedule, error measures, saved files and plots as Main_2D.py,
but each training step is one compiled graph from ../Fast_integration.py
FB / NW / DNW follow the paper exactly (Algorithms 3, 4, 5; eq. (30), (34), (37)).
Output files get the suffix SUFFIX (set SUFFIX = '' to overwrite the original results).
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
from Fast_integration import FastIntegrator, make_train_step, get_MC_square

from NN_Architecture_2D import net_u
from Integrate_Points import get_MC_quad
from Data import exact_u, f_rhs, exact_dx, exact_dy
from Plot import plot_model_2D
keras.utils.set_random_seed(1234)

num_epochs = 10000
num_neurons_u = 50
num_layers = 1
activation = tf.nn.relu
initial_learning_rate = 1.e-2
final_learning_rate = 1.e-5
# CMC_MODE = 'density': SAME POINT DENSITY as Standard MC with n_MC points: every clipped polygon
#                     (every neuron interaction D_ij) gets on average n_MC*|D_ij|/|Omega| points,
#                     i.e. exactly the points Standard MC puts inside it
# CMC_MODE = 'budget': EQUAL BUDGET with Standard MC — n_MC points IN TOTAL per epoch, shared
#                     among all clipped polygons integrated in that epoch (on average
#                     proportional to area; unbiased), each mapped directly onto its polygon
# CMC_MODE = 'sweep' : n_MC points in EVERY clipped polygon (direct map, no triangles)
# CMC_MODE = 'fan'   : original: centroid triangulation, n_MC points in EVERY triangle
CMC_MODE = 'density'
CMC_MAP = 'fan' if CMC_MODE == 'fan' else 'sweep'
n_MC = 1000                   # Standard MC points: 'density' -> same density, 'budget' -> same total
RESAMPLE_EACH_EPOCH = True    # Main_2D.py draws new MC points at every epoch

Method = 'FB'   # Full-Batched
# Method = 'NW'   # Neuron-Wise
# Method = 'DNW'    # Double Neuron-Wise

alpha = 0.0                   # reaction term coefficient
lambda_reg = 0.1              # the original CMC train steps use their default lambda_reg = 0.1
domain_vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]
GEOMETRY_GRADIENT = True      # paper, Corollary 4.2: gradient through the clipped polygons
DNW_UNBIASED = True          # paper, end of Sec. 5.3: stiffness sub-sum rescaled by n / r_ki.
                              # False = Algorithm 5 without it: biased by (n+1)/(2n), converges
                              # to a WRONG solution (about 1/0.55 times u for n = 10)
DNW_UPDATE = 'batched'        # 'batched'   : ONE optimizer update per epoch with sum_{i in I_k} J~_ki
                              #               (as in your original code; fewest interactions -> fastest)
                              # 'sequential': Algorithm 5 literally, one update per i in I_k
                              #               (r_k updates per epoch -> slower per epoch)
SUFFIX = '_fast' + {'density': '_density', 'budget': '_budget', 'sweep': '_sweep', 'fan': ''}[CMC_MODE]
TAG = {'FB': '_CMC', 'NW': '_CMC_i', 'DNW': '_CMC_ij'}   # same file names as Main_2D.py (+ SUFFIX)

def draw_points(n):
    return get_MC_square(n) if CMC_MAP == 'sweep' else get_MC_quad(n)

pts, wts = draw_points(n_MC)
integrator = FastIntegrator('CMC', num_neurons_u, domain_vertices, pts, wts, alpha, f_rhs,
                            cmc_map=CMC_MAP, budget=n_MC if CMC_MODE in ('budget', 'density') else None,
                            allocation='density' if CMC_MODE == 'density' else 'budget',
                            geometry_gradient=GEOMETRY_GRADIENT, dnw_unbiased=DNW_UNBIASED, dnw_update=DNW_UPDATE)

# ════════════════════════════════════════════════════════════════════════════
#  Fast training (one compiled graph per step; see Fast_integration.py)
# ════════════════════════════════════════════════════════════════════════════
script_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(script_dir, 'Saved_Models'), exist_ok=True)
os.makedirs('Saved_Pics', exist_ok=True)          # Plot.py saves to ./Saved_Pics

u_net = net_u(num_layers, num_neurons_u, activation=activation)
u = u_net(tf.random.uniform(shape=[1, 2], minval=0., maxval=1., dtype=PRECISION))
u_net.save(os.path.join(script_dir, f'Saved_Models/u_net_initialize{SUFFIX}.keras'))
u_net.summary()

optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
optimizer.build(u_net.trainable_variables)
train_step = make_train_step(u_net, optimizer, integrator, Method, lambda_reg)

# ---- error evaluation grid (same as the original Main_2D.py) ---------------
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
    """Relative L2 errors (%) of u and grad u — same formulas as the original loop."""
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
    """Same exponential decay as train_u_1D.lr_schedule, but in plain Python: the original is a
    @tf.function called with a Python int, which re-traces (recompiles) it at EVERY epoch."""
    decay_rate = (final_learning_rate / initial_learning_rate) ** (1 / num_epochs)
    return initial_learning_rate * decay_rate ** epoch

# ---- training loop ----------------------------------------------------------
Loss, errors, d_errors = [], [], []
start_time = time.time()
for epoch in range(num_epochs):
    new_learning_rate = lr_schedule(epoch, initial_learning_rate, final_learning_rate, num_epochs)
    optimizer.learning_rate.assign(new_learning_rate)

    if RESAMPLE_EACH_EPOCH and CMC_MODE not in ('budget', 'density'):   # 'budget' draws new points inside the step
        pts, wts = draw_points(n_MC)          # new MC points every epoch, as in Main_2D.py
        integrator.set_points(pts, wts)
    loss = train_step()
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
