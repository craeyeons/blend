"""
Thin MLP builder for scalar-output PINNs (Helmholtz), with optional
Fourier-feature input embedding (Tancik et al. 2020).

Fourier features are essential for fitting oscillatory solutions like
sin(kx)sin(ky) at moderate-to-high k; vanilla tanh MLPs exhibit strong
spectral bias and fail even at k=4 pi.
"""

import numpy as np
import tensorflow as tf


class FourierFeatures(tf.keras.layers.Layer):
    """Random Fourier feature embedding: x -> [sin(2 pi B x), cos(2 pi B x)].

    B has shape (m, d) with entries drawn from N(0, scale^2). B is fixed
    (not trained); it is stored as a non-trainable weight so it is saved
    with the model.
    """

    def __init__(self, num_features, scale, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.num_features = int(num_features)
        self.scale = float(scale)
        self.seed = int(seed)

    def build(self, input_shape):
        d = int(input_shape[-1])
        rng = np.random.default_rng(self.seed)
        B = rng.normal(loc=0.0, scale=self.scale,
                       size=(self.num_features, d)).astype(np.float32)
        self.B = self.add_weight(
            name='B', shape=B.shape, trainable=False,
            initializer=tf.keras.initializers.Constant(B))
        super().build(input_shape)

    def call(self, x):
        proj = 2.0 * np.pi * tf.matmul(x, self.B, transpose_b=True)
        return tf.concat([tf.sin(proj), tf.cos(proj)], axis=-1)


def build_pinn(num_inputs=2, layers=(64, 64, 64, 64), activation='tanh',
               input_range=((0.0, 1.0), (0.0, 1.0)),
               fourier_m=64, fourier_scale=2.0, fourier_seed=0):
    """Build an MLP with input rescaling to [-1, 1] and optional Fourier features.

    Parameters
    ----------
    fourier_m : int
        Number of Fourier frequencies. Output embedding dim is 2 * fourier_m.
        Set to 0 to disable Fourier features (plain MLP).
    fourier_scale : float
        Std of Gaussian from which frequency matrix B is drawn. Rule of
        thumb for Helmholtz: fourier_scale ~ k / (2 pi), i.e. a few times
        the problem's dominant wavenumber (in cycles-per-unit-domain).
    fourier_seed : int
        Seed for B.
    """
    assert len(input_range) == num_inputs, \
        f"input_range has {len(input_range)} entries for num_inputs={num_inputs}"

    inp = tf.keras.Input(shape=(num_inputs,), dtype=tf.float32)

    los = tf.constant([r[0] for r in input_range], dtype=tf.float32)
    his = tf.constant([r[1] for r in input_range], dtype=tf.float32)
    h = 2.0 * (inp - los) / (his - los) - 1.0

    if fourier_m and fourier_m > 0:
        h = FourierFeatures(num_features=fourier_m, scale=fourier_scale,
                            seed=fourier_seed, name='fourier')(h)

    for width in layers:
        h = tf.keras.layers.Dense(
            width, activation=activation,
            kernel_initializer='glorot_normal')(h)
    out = tf.keras.layers.Dense(1, activation=None,
                                kernel_initializer='glorot_normal')(h)

    return tf.keras.Model(inp, out)
