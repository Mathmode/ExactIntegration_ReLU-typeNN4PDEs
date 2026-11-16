"""Full-epoch timing — Conforming Monte Carlo (CMC) with the FB, NW and DNW updates.

Common protocol: ../Timing_protocol.py (precision, same states, synchronised timer, warm-up excluded).
Integration: ../Fast_integration.py (CMC points, moment-compressed stiffness, load of degree 10).
A full epoch is one training update, including the point evaluations and the optimizer step.
Random-subset schedules: every epoch a random set K1 of r neurons (and K2 of rbar neurons for DNW) is drawn;
one gradient and one update on the summed loss.

For every n in neurons_list, all variants are timed in ONE call of time_variants (same states, same
conditions), so that ratios between them are comparable:
  * FB;
  * NW and DNW with fixed sizes r = rbar = 10;
  * NW with r = alpha n and DNW with r = alpha n, rbar = beta n, for (alpha, beta) in grid_settings.

Output: timing_results_CMC_full_epoch.json (next to this file), read by Plot_time_figures.py.
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
for _name in ("Integrate_Points", "Data", "NN_Architecture_2D"):
    _mod = sys.modules.get(_name)
    if _mod is None:
        continue
    _file = getattr(_mod, "__file__", None)
    if _file is None or os.path.dirname(os.path.abspath(_file)) != THIS_DIR:
        del sys.modules[_name]
from Timing_protocol import set_precision, time_variants
PRECISION = "float64"
set_precision(PRECISION)                                  # before any network is built
from Fast_integration import FastIntegrator, make_train_steps
from NN_Architecture_2D import net_u
from Integrate_Points import get_MC_quad
from Data import f_rhs

HERE = os.path.dirname(os.path.abspath(__file__))
QUICK = "--quick" in sys.argv
n_MC = 10000                  # CMC points per triangle (fan), as in My_integration.py
num_layers = 1
activation = tf.nn.relu
initial_learning_rate = 1.e-1
alpha = 0.0
lambda_reg = 0.01              # timing runs: no interior regularisation
domain_vertices = [(0, 0), (1, 0), (1, 1), (0, 1)]

fixed_size = 10                                           # r = rbar = 10
neurons_list = [16, 32] if QUICK else [16, 32, 64, 128, 256, 512, 1024, 2048]
grid_settings = []#[(0.5, 0.5), (0.5, 0.25)]               # (alpha, beta) of the random-subset runs
OUT = os.path.join(HERE, 'timing_results_CMC_full_epoch_quick.json' if QUICK
                   else 'timing_results_CMC_full_epoch.json')


def epochs_for(N):
    """Mean seconds per epoch over this many timed epochs."""
    if QUICK:
        return 2
    return 100 if N <= 32 else (20 if N <= 128 else (5 if N <= 512 else 3))


def spec_for(N):
    """name -> (method, r, rbar) of every variant timed at size N."""
    spec = {'FB': ('FB', None, None),
            'NW_r10': ('NW', fixed_size, None),
            'DNW_r10': ('DNW', fixed_size, fixed_size)}
    for a in sorted({a for a, _ in grid_settings}):
        spec[f'NW_a{a}'] = ('NW', int(round(a * N)), None)
    for a, b in grid_settings:
        spec[f'DNW_a{a}_b{b}'] = ('DNW', int(round(a * N)), int(round(b * N)))
    return spec


pts, wts = get_MC_quad(n_MC)


def timed_block(N, spec, epochs):
    """Mean seconds per epoch of every variant of spec, timed with the common protocol."""
    integ = FastIntegrator(N, domain_vertices, alpha, f_rhs, pts, wts, load_degree=10)

    def build(name):
        method, r, rbar = spec[name]
        net = net_u(num_layers, N, activation=activation)
        _ = net(tf.zeros([1, 2], dtype=PRECISION))
        opt = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
        opt.build(net.trainable_variables)
        if method == 'FB':
            return net, make_train_steps(net, opt, integ, 'FB', lambda_reg)
        return net, make_train_steps(net, opt, integ, method, lambda_reg, n_batched_1=r, n_batched_2=rbar)

    return time_variants(build, list(spec), 'FB', epochs)


results = {'neurons': neurons_list, 'settings': [list(s) for s in grid_settings], 'by_n': {}}
for N in neurons_list:
    el = timed_block(N, spec_for(N), epochs_for(N))
    results['by_n'][str(N)] = el
    print(f"N = {N:4d}:  " + "   ".join(f"{k} {v:.5f} s" for k, v in el.items()), flush=True)
    with open(OUT, 'w') as f:
        json.dump(results, f)
    gc.collect()

print(f"CMC full-epoch timing finished: {OUT}", flush=True)
