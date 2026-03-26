"""
PINN architecture for 2D static linear elasticity.

Input: (x, y) coordinates
Output: (ux, uy) displacement components

Stresses are derived via automatic differentiation during residual computation.
"""

import tensorflow as tf


class Network:
    """
    Build a physics-informed neural network for 2D linear elasticity.
    """

    def __init__(self):
        self.activations = {
            'tanh': 'tanh',
            'swish': self.swish,
            'mish': self.mish,
        }

    def swish(self, x):
        return x * tf.math.sigmoid(x)

    def mish(self, x):
        return x * tf.math.tanh(tf.softplus(x))

    def build(self, num_inputs=2, layers=[64, 64, 64, 64],
              activation='tanh', num_outputs=2,
              input_range=None):
        """
        Build a PINN model for static linear elasticity.

        Parameters
        ----------
        num_inputs : int
            Number of input variables (default 2 for x, y).
        layers : list
            Hidden layer sizes.
        activation : str
            Activation function name.
        num_outputs : int
            Number of outputs (default 2 for ux, uy).
        input_range : list of (min, max) tuples, optional
            Per-input ranges for normalization to [-1, 1].
            E.g. [(0, 2), (0, 2)] for x in [0,2], y in [0,2].

        Returns
        -------
        tf.keras.Model
        """
        inputs = tf.keras.layers.Input(shape=(num_inputs,))
        x = inputs

        # Normalize inputs to [-1, 1] if ranges are provided
        if input_range is not None:
            lo = tf.constant([r[0] for r in input_range], dtype=tf.float32)
            hi = tf.constant([r[1] for r in input_range], dtype=tf.float32)
            x = 2.0 * (x - lo) / (hi - lo + 1e-10) - 1.0

        for units in layers:
            x = tf.keras.layers.Dense(
                units,
                activation=self.activations[activation],
                kernel_initializer='glorot_normal',
            )(x)
        outputs = tf.keras.layers.Dense(
            num_outputs, kernel_initializer='glorot_normal'
        )(x)
        return tf.keras.models.Model(inputs=inputs, outputs=outputs)
