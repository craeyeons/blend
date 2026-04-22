"""
Thin MLP builder for scalar-output PINNs (Helmholtz).

Kept deliberately minimal and independent of `elasticity/lib/network.py`
which has elasticity-specific hard BC machinery.
"""

import tensorflow as tf


def build_pinn(num_inputs=2, layers=(64, 64, 64, 64), activation='tanh',
               input_range=((0.0, 1.0), (0.0, 1.0))):
    """Build a fully-connected MLP with input rescaling to [-1, 1].

    Parameters
    ----------
    num_inputs : int
        2 for non-parametric (x, y); > 2 for parametric PINNs
        (e.g. (x, y, x_s, y_s, k)).
    layers : tuple[int]
        Hidden layer widths.
    activation : str
        Keras activation name.
    input_range : sequence of (lo, hi) of length num_inputs
        Used to rescale inputs to [-1, 1] before the first dense layer.

    Returns
    -------
    keras.Model mapping (N, num_inputs) -> (N, 1) scalar output.
    """
    assert len(input_range) == num_inputs, \
        f"input_range has {len(input_range)} entries for num_inputs={num_inputs}"

    inp = tf.keras.Input(shape=(num_inputs,), dtype=tf.float32)

    los = tf.constant([r[0] for r in input_range], dtype=tf.float32)
    his = tf.constant([r[1] for r in input_range], dtype=tf.float32)
    h = 2.0 * (inp - los) / (his - los) - 1.0

    for width in layers:
        h = tf.keras.layers.Dense(
            width, activation=activation,
            kernel_initializer='glorot_normal')(h)
    out = tf.keras.layers.Dense(1, activation=None,
                                kernel_initializer='glorot_normal')(h)

    return tf.keras.Model(inp, out)
