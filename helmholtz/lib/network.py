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


class KModulatedFourierFeatures(tf.keras.layers.Layer):
    """k-modulated Fourier features: phi(xy; k) = [sin(k*B*xy), cos(k*B*xy)].

    Frequencies scale with the sample's own k, matching physics
    (wavelength = 2 pi / k). B has shape (m, 2), entries ~ N(0, scale^2).
    Call signature: layer([xy, k]) where xy is (N, 2) and k is (N, 1).
    """

    def __init__(self, num_features, scale=1.0, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.num_features = int(num_features)
        self.scale = float(scale)
        self.seed = int(seed)

    def build(self, input_shape):
        rng = np.random.default_rng(self.seed)
        B = rng.normal(loc=0.0, scale=self.scale,
                       size=(self.num_features, 2)).astype(np.float32)
        self.B = self.add_weight(
            name='B', shape=B.shape, trainable=False,
            initializer=tf.keras.initializers.Constant(B))
        super().build(input_shape)

    def call(self, inputs):
        xy, k = inputs
        proj = k * tf.matmul(xy, self.B, transpose_b=True)
        return tf.concat([tf.sin(proj), tf.cos(proj)], axis=-1)


class FiLMBlock(tf.keras.layers.Layer):
    """Dense -> activation -> FiLM affine modulation from params.

    h = Dense(width)(x)
    gamma, beta = Dense(2*width)(params_s) split in half
    out = gamma * act(h) + beta
    """

    def __init__(self, width, activation='tanh', **kwargs):
        super().__init__(**kwargs)
        self.width = int(width)
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        self.dense = tf.keras.layers.Dense(
            self.width, activation=None,
            kernel_initializer='glorot_normal')
        self.film = tf.keras.layers.Dense(
            2 * self.width, activation=None,
            kernel_initializer='glorot_normal',
            bias_initializer=tf.keras.initializers.Zeros())
        super().build(input_shape)

    def call(self, inputs):
        h, params_s = inputs
        h = self.activation(self.dense(h))
        gb = self.film(params_s)
        gamma, beta = tf.split(gb, 2, axis=-1)
        return (1.0 + gamma) * h + beta


def build_parametric_pinn(layers=(256, 256, 256, 256, 256),
                           activation='tanh',
                           spatial_range=((0.0, 1.0), (0.0, 1.0)),
                           param_range=((2.0 * np.pi, 6.0 * np.pi),
                                        (0.0, 1.0), (0.0, 1.0)),
                           fourier_m=128, fourier_scale=1.0,
                           fourier_seed=0):
    """Build a 5-input parametric PINN: (x, y, k, x_s, y_s) -> u.

    Uses k-modulated Fourier features on (x, y) so spatial frequency
    tracks each sample's k, and FiLM-style per-layer modulation from
    (k, x_s, y_s) so the parameters can't be ignored.

    Parameters
    ----------
    spatial_range : ((xlo, xhi), (ylo, yhi))
    param_range   : ((k_lo, k_hi), (xs_lo, xs_hi), (ys_lo, ys_hi))
    fourier_scale : float
        Std of B for the k-modulated Fourier layer. With the k multiplier
        carrying the magnitude, ~1 is a sensible default.
    """
    inp = tf.keras.Input(shape=(5,), dtype=tf.float32)

    xy = inp[:, 0:2]
    k_raw = inp[:, 2:3]
    params = inp[:, 2:5]

    sp_lo = tf.constant([r[0] for r in spatial_range], dtype=tf.float32)
    sp_hi = tf.constant([r[1] for r in spatial_range], dtype=tf.float32)
    xy_s = 2.0 * (xy - sp_lo) / (sp_hi - sp_lo) - 1.0

    p_lo = tf.constant([r[0] for r in param_range], dtype=tf.float32)
    p_hi = tf.constant([r[1] for r in param_range], dtype=tf.float32)
    p_s = 2.0 * (params - p_lo) / (p_hi - p_lo) - 1.0

    if fourier_m and fourier_m > 0:
        xy_feat = KModulatedFourierFeatures(
            num_features=fourier_m, scale=fourier_scale,
            seed=fourier_seed, name='kfourier_xy')([xy_s, k_raw])
    else:
        xy_feat = xy_s

    h = xy_feat
    for i, width in enumerate(layers):
        h = FiLMBlock(width, activation=activation,
                      name=f'film_{i}')([h, p_s])
    out = tf.keras.layers.Dense(1, activation=None,
                                kernel_initializer='glorot_normal')(h)
    return tf.keras.Model(inp, out)


def make_xy_callable(parametric_pinn, k, xs, ys):
    """Wrap a 5-input parametric PINN into a 2-input callable that looks
    like a scalar PINN. `(k, x_s, y_s)` are baked in. Useful for passing
    to solver/hybrid code that expects a `(N, 2) -> (N, 1)` model."""
    k_f = float(k); xs_f = float(xs); ys_f = float(ys)
    k_tf = tf.constant(k_f, dtype=tf.float32)
    xs_tf = tf.constant(xs_f, dtype=tf.float32)
    ys_tf = tf.constant(ys_f, dtype=tf.float32)

    def fn(xy, training=False):
        xy = tf.convert_to_tensor(xy, dtype=tf.float32)
        N = tf.shape(xy)[0]
        k_col = tf.fill([N, 1], k_tf)
        xs_col = tf.fill([N, 1], xs_tf)
        ys_col = tf.fill([N, 1], ys_tf)
        inp5 = tf.concat([xy, k_col, xs_col, ys_col], axis=-1)
        return parametric_pinn(inp5, training=training)
    return fn
