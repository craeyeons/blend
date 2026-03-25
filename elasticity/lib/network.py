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
              activation='tanh', num_outputs=2):
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

        Returns
        -------
        tf.keras.Model
        """
        inputs = tf.keras.layers.Input(shape=(num_inputs,))
        x = inputs
        for units in layers:
            x = tf.keras.layers.Dense(
                units,
                activation=self.activations[activation],
                kernel_initializer='he_normal',
            )(x)
        outputs = tf.keras.layers.Dense(
            num_outputs, kernel_initializer='he_normal'
        )(x)
        return tf.keras.models.Model(inputs=inputs, outputs=outputs)
