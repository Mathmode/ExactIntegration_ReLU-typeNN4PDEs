"""
Timing_protocol.py — ONE timing protocol for Exact integration, CMC and Standard MC.

  * precision : every network and every computation in PRECISION (default float64)
  * epochs    : the same number of timed epochs for every method
  * states    : at every epoch, each variant is timed for ONE epoch starting from the SAME
                weights, taken from a reference training run (FB of the same method; Standard MC
                uses its own run).  Differences in time come from the algorithm only.
  * sync      : the timer stops only after the epoch has finished (float(loss))
  * warm-up   : graph tracing (first call) is excluded
  * points    : Monte Carlo methods draw NEW points every epoch (paper, Sec. 3 and eq. (32))
One "epoch" is one pass of the training loop of the paper: one update for FB / NW / Standard MC,
one update for DNW with dnw_update="batched" (default), r_k updates with "sequential" (Algorithm 5 literally).
"""
import time
import numpy as np
import keras


def set_precision(precision="float64"):
    """Call BEFORE building any network.  Sets the default dtype for BOTH Keras front-ends
    (`keras` and `tf.keras` can be different packages, e.g. Keras 3 + tf_keras / TF <= 2.15)."""
    keras.backend.set_floatx(precision)
    try:
        import tensorflow as tf
        tf.keras.backend.set_floatx(precision)
    except Exception:
        pass


def check_precision(net, precision="float64"):
    """Warn if a network was built in a different precision than requested."""
    dts = {v.dtype if isinstance(v.dtype, str) else v.dtype.name for v in net.trainable_variables}
    dts = {str(d).replace("tf.", "").replace("<dtype: '", "").replace("'>", "") for d in dts}
    if dts != {precision}:
        print(f"WARNING: network variables are {sorted(dts)}, requested {precision}")


def time_variants(build, variants, ref_variant, num_epochs):
    """build(variant) -> (net, train_step).  Returns {variant: mean seconds per epoch}."""
    ref_net, ref_step = build(ref_variant)
    check_precision(ref_net, keras.backend.floatx())
    float(ref_step())                                   # warm-up of the reference
    runs = {v: build(v) for v in variants}
    w0 = [x.numpy() for x in ref_net.trainable_variables]
    for v, (net, step) in runs.items():                 # warm-up / tracing (not timed)
        for a, x in zip(net.trainable_variables, w0):
            a.assign(x)
        float(step())
    tot = {v: 0.0 for v in variants}
    for epoch in range(num_epochs):
        w = [x.numpy() for x in ref_net.trainable_variables]
        for v, (net, step) in runs.items():
            for a, x in zip(net.trainable_variables, w):
                a.assign(x)
            t0 = time.perf_counter()
            float(step())                               # synchronised
            tot[v] += time.perf_counter() - t0
        float(ref_step())                               # advance the reference state
    return {v: tot[v] / num_epochs for v in variants}
