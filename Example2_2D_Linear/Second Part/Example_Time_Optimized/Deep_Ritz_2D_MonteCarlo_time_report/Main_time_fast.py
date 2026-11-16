"""Full-epoch timing — Standard Monte Carlo, with the SAME protocol as Exact integration and CMC
(../Timing_protocol.py): same precision, same number of epochs, synchronised timer,
NEW uniform points in the whole square Omega = [0,1]^2 at every epoch (paper, Sec. 3).

Output: timing_results_MC_full_epoch.json (next to this file), read by Plot_time_figures.py.
Run:    python Main_time_fast.py            (python Main_time_fast.py --quick  for a short test)
"""
import os, sys, json, gc
os.environ["KERAS_BACKEND"] = "tensorflow"
import numpy as np
import tensorflow as tf
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
# this folder's own modules first; drop same-named modules that an earlier run in the same IDE console
# loaded from another folder
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)
for _name in ("Data", "NN_Architecture_2D"):
    _mod = sys.modules.get(_name)
    if _mod is None:
        continue
    _file = getattr(_mod, "__file__", None)
    if _file is None or os.path.dirname(os.path.abspath(_file)) != THIS_DIR:
        del sys.modules[_name]
from Timing_protocol import set_precision, time_variants
PRECISION = "float64"
set_precision(PRECISION)
from NN_Architecture_2D import net_u
from Data import f_rhs

HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = "--quick" in sys.argv
n_MC = 10000
num_layers = 1
activation = tf.nn.relu
initial_learning_rate = 1.e-1
area_Omega = 1.0
DT = tf.as_dtype(PRECISION)
neurons_list = [16, 32] if QUICK else [16, 32, 64, 128, 256, 512, 1024, 2048]
num_epochs = 2 if QUICK else 100
OUT = os.path.join(HERE, 'timing_results_MC_full_epoch_quick.json' if QUICK
                   else 'timing_results_MC_full_epoch.json')


def make_smc_step(u_net, optimizer):
    @tf.function
    def train_step():
        xy = tf.random.uniform([n_MC, 2], dtype=DT)              # new points every epoch
        f_val = tf.cast(tf.reshape(f_rhs(xy[:, 0], xy[:, 1]), (-1, 1)), DT)
        with tf.GradientTape() as outer:
            with tf.GradientTape() as inner:
                inner.watch(xy)
                u = tf.cast(u_net(xy), DT)   # same dtype as f_val in any Keras version
            grad_u = tf.cast(inner.gradient(u, xy), DT)
            loss = area_Omega * (0.5 * tf.reduce_mean(tf.reduce_sum(grad_u**2, 1))
                                 - tf.reduce_mean(f_val * u))
        g = outer.gradient(loss, u_net.trainable_variables)
        optimizer.apply_gradients(zip(g, u_net.trainable_variables))
        return loss
    return train_step


results = {'neurons': neurons_list, 'SMC': []}
for N in neurons_list:
    def build(variant):
        net = net_u(num_layers, N, activation=activation)
        _ = net(tf.zeros([1, 2], dtype=PRECISION))
        opt = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
        opt.build(net.trainable_variables)
        return net, make_smc_step(net, opt)
    el = time_variants(build, ['SMC'], 'SMC', num_epochs)
    results['SMC'].append(el['SMC'])
    print(f"N = {N:4d}:  SMC {el['SMC']:.5f} s", flush=True)
    with open(OUT, 'w') as f:
        json.dump(results, f)
    gc.collect()

print(f"Standard MC full-epoch timing finished: {OUT}", flush=True)
