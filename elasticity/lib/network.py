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
              input_range=None, hard_bc=None, hard_bc_params=None):
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
        hard_bc : str, optional
            Hard displacement BC mode baked into the network output.
            'l_bracket'       – bottom edge fixed: u *= (y - y_min)
            'plate_with_hole' – rollers: ux *= (x - x_min), uy *= (y - y_min)
            None              – no hard BC (soft penalty only)

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

        # Hard displacement BC: multiply output by distance-to-boundary
        # so the prescribed displacements are satisfied exactly.
        if hard_bc == 'l_bracket':
            # Bottom edge (y = y_min) fully fixed: u(x, y_min) = 0
            y_min = input_range[1][0] if input_range else 0.0
            dist = inputs[:, 1:2] - y_min          # (y - y_min), zero on bottom

            # Smooth void mask: ~1 in material, ~0 in void (upper-right cutout).
            # The void is where x > corner_x AND y > corner_y.
            # void_mask = 1 - H(x - cx) * H(y - cy)  with smooth Heaviside.
            params = hard_bc_params or {}
            cx = params.get('corner_x', 1.0)
            cy = params.get('corner_y', 1.0)
            k = 20.0  # sharpness of transition
            x_coord = inputs[:, 0:1]
            y_coord = inputs[:, 1:2]
            h_x = tf.sigmoid(k * (x_coord - cx))  # ~0 when x < cx, ~1 when x > cx
            h_y = tf.sigmoid(k * (y_coord - cy))  # ~0 when y < cy, ~1 when y > cy
            void_mask = 1.0 - h_x * h_y           # ~1 in material, ~0 in void

            outputs = outputs * dist * void_mask
        elif hard_bc == 'plate_with_hole':
            # Left edge roller: ux(x_min, y) = 0
            # Bottom edge roller: uy(x, y_min) = 0
            x_min = input_range[0][0] if input_range else -2.0
            y_min = input_range[1][0] if input_range else -2.0
            dist_x = inputs[:, 0:1] - x_min        # zero on left edge
            dist_y = inputs[:, 1:2] - y_min        # zero on bottom edge
            ux = outputs[:, 0:1] * dist_x
            uy = outputs[:, 1:2] * dist_y
            outputs = tf.concat([ux, uy], axis=1)

        return tf.keras.models.Model(inputs=inputs, outputs=outputs)
