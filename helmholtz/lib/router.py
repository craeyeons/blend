"""
CNN-based Router for hybrid PINN+FEM Helmholtz solving.

Decides per cell whether the PINN's prediction of `u` should be trusted
(accept -> pin as Dirichlet) or rejected (leave to FEM). The architecture
mirrors `elasticity/lib/router.py` but is adapted to the scalar Helmholtz
problem: 4 input channels instead of 9, PDE residual
    r = uxx + uyy + k^2 u + f
instead of the elastic equilibrium residuals.

Input (Ny, Nx, 4):
    0: layout mask (1 = solid, 0 = inside hole)
    1: source f(x, y), normalized by max|f|
    2: PINN prediction u(x, y), normalized by max|u|
    3: |PDE residual|, median-normalized

Output: (Ny, Nx, 1) raw logits — positive logit = "reject PINN (use FEM)",
negative logit = "accept PINN". A caller thresholds at 0 by default.

Loss (mirror of elasticity):
    L = 1/|M| * sum_M [ beta * softplus(s) + R_bar * softplus(-s) ]
        + lambda_tv * TV(s)

where `R_bar` is the median-normalized residual; loss is masked by the
layout (no gradient from cells inside the hole).
"""

import os
import numpy as np
import tensorflow as tf

_gpus = tf.config.list_physical_devices('GPU')
if _gpus:
    try:
        for _gpu in _gpus:
            tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass

from tensorflow import keras
from tensorflow.keras import layers


# ----------------------------------------------------------------------
# CNN architecture
# ----------------------------------------------------------------------

class RouterCNN(keras.Model):
    """U-Net router: (B, Ny, Nx, 4) -> (B, Ny, Nx, 1) raw logits."""

    def __init__(self, base_filters=32, name="helmholtz_router_cnn"):
        super().__init__(name=name)
        self.conv1 = layers.Conv2D(base_filters, 3, padding='same', activation='relu')
        self.conv1b = layers.Conv2D(base_filters, 3, padding='same', activation='relu')
        self.pool1 = layers.MaxPooling2D(2)

        self.conv2 = layers.Conv2D(base_filters * 2, 3, padding='same', activation='relu')
        self.conv2b = layers.Conv2D(base_filters * 2, 3, padding='same', activation='relu')
        self.pool2 = layers.MaxPooling2D(2)

        self.conv3 = layers.Conv2D(base_filters * 4, 3, padding='same', activation='relu')
        self.conv3b = layers.Conv2D(base_filters * 4, 3, padding='same', activation='relu')
        self.pool3 = layers.MaxPooling2D(2)

        self.conv4 = layers.Conv2D(base_filters * 8, 3, padding='same', activation='relu')
        self.conv4b = layers.Conv2D(base_filters * 8, 3, padding='same', activation='relu')

        self.up3 = layers.UpSampling2D(2)
        self.conv5 = layers.Conv2D(base_filters * 4, 3, padding='same', activation='relu')
        self.conv5b = layers.Conv2D(base_filters * 4, 3, padding='same', activation='relu')

        self.up2 = layers.UpSampling2D(2)
        self.conv6 = layers.Conv2D(base_filters * 2, 3, padding='same', activation='relu')
        self.conv6b = layers.Conv2D(base_filters * 2, 3, padding='same', activation='relu')

        self.up1 = layers.UpSampling2D(2)
        self.conv7 = layers.Conv2D(base_filters, 3, padding='same', activation='relu')
        self.conv7b = layers.Conv2D(base_filters, 3, padding='same', activation='relu')

        self.output_conv = layers.Conv2D(
            1, 1, padding='same', activation=None,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros',
        )

    def call(self, inputs, training=None):
        layout = inputs[..., 0:1]

        e1 = self.conv1b(self.conv1(inputs))
        p1 = self.pool1(e1)
        e2 = self.conv2b(self.conv2(p1))
        p2 = self.pool2(e2)
        e3 = self.conv3b(self.conv3(p2))
        p3 = self.pool3(e3)

        b = self.conv4b(self.conv4(p3))

        d3 = self._match_size(self.up3(b), e3)
        d3 = layers.Concatenate()([d3, e3])
        d3 = self.conv5b(self.conv5(d3))

        d2 = self._match_size(self.up2(d3), e2)
        d2 = layers.Concatenate()([d2, e2])
        d2 = self.conv6b(self.conv6(d2))

        d1 = self._match_size(self.up1(d2), e1)
        d1 = layers.Concatenate()([d1, e1])
        d1 = self.conv7b(self.conv7(d1))

        logits = self.output_conv(d1)
        return logits * layout

    def _match_size(self, x, target):
        th = tf.shape(target)[1]
        tw = tf.shape(target)[2]
        return tf.image.resize(x, [th, tw], method='bilinear')


# ----------------------------------------------------------------------
# Residual computer
# ----------------------------------------------------------------------

class HelmholtzResidualComputer:
    """Compute the Helmholtz PDE residual at grid cells using PINN autodiff.

        r(x, y) = d2u/dx2 + d2u/dy2 + k^2 u(x, y) + f(x, y)

    (Equivalent to -Delta u - k^2 u = f rearranged to zero.)
    """

    def __init__(self, pinn_model, k):
        self.pinn_model = pinn_model
        self.k = float(k)

    @tf.function
    def _compute_flat(self, xy_flat, f_flat):
        with tf.GradientTape() as t2:
            t2.watch(xy_flat)
            with tf.GradientTape() as t1:
                t1.watch(xy_flat)
                u = self.pinn_model(xy_flat, training=False)  # (N, 1)
            grads = t1.gradient(u, xy_flat)  # (N, 2)
        hess = t2.batch_jacobian(grads, xy_flat)  # (N, 2, 2)
        uxx = hess[:, 0, 0:1]
        uyy = hess[:, 1, 1:2]
        r = uxx + uyy + (self.k ** 2) * u + f_flat
        return tf.reshape(tf.abs(r), [-1])

    def compute_residual(self, X, Y, f_grid):
        """Return |residual| on the (Ny, Nx) grid."""
        shape = X.shape
        xy = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)
        f_flat = f_grid.ravel().astype(np.float32).reshape(-1, 1)
        xy_tf = tf.constant(xy)
        f_tf = tf.constant(f_flat)
        r_flat = self._compute_flat(xy_tf, f_tf).numpy()
        return r_flat.reshape(shape).astype(np.float32)


# ----------------------------------------------------------------------
# Router input builder
# ----------------------------------------------------------------------

def create_router_input(layout, f_source, pinn_u, residual):
    """Assemble (1, Ny, Nx, 4) tensor for the router.

    Normalization:
      - `f_source / max|f|`
      - `pinn_u / max|u|`
      - `residual / median(|residual|)` (on solid cells)
    """
    H, W = layout.shape
    f_max = np.max(np.abs(f_source)) + 1e-10
    u_max = np.max(np.abs(pinn_u)) + 1e-10
    solid = layout > 0
    if np.any(solid):
        res_vals = np.abs(residual[solid])
        res_median = np.median(res_vals) + 1e-10
    else:
        res_median = 1.0

    channels = np.stack([
        layout.astype(np.float32),
        (f_source / f_max).astype(np.float32),
        (pinn_u / u_max).astype(np.float32),
        (residual / res_median).astype(np.float32),
    ], axis=-1)
    return channels[np.newaxis, ...]


# ----------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------

class RouterTrainer:
    """Trains a RouterCNN against the median-normalized PDE residual.

    Loss (per solid pixel):
        beta * softplus(s) + R_bar * softplus(-s)
    plus total-variation regularization on s.
    """

    def __init__(self, router, pinn_model, k,
                 beta=0.1, lambda_tv=0.01, grad_clip_norm=1.0):
        self.router = router
        self.pinn_model = pinn_model
        self.k = float(k)
        self.beta = beta
        self.lambda_tv = lambda_tv
        self.grad_clip_norm = grad_clip_norm

        self.residual_computer = HelmholtzResidualComputer(pinn_model, k)
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        self.loss_history = []
        self.logistic_loss_history = []
        self.tv_loss_history = []

    def _tv(self, r):
        tv_h = tf.reduce_mean(tf.abs(r[:, :, 1:, :] - r[:, :, :-1, :]))
        tv_v = tf.reduce_mean(tf.abs(r[:, 1:, :, :] - r[:, :-1, :, :]))
        return tv_h + tv_v

    @tf.function
    def train_step(self, inputs, residual_field, layout_mask):
        """One gradient step.

        inputs : (1, Ny, Nx, 4) router input tensor
        residual_field : (Ny, Nx) median-normalized |residual|
        layout_mask : (Ny, Nx) float (1=solid, 0=hole)
        """
        with tf.GradientTape() as tape:
            s = self.router(inputs, training=True)
            s = s[0, :, :, 0]

            layout_f = tf.cast(layout_mask, tf.float32)
            num = tf.reduce_sum(layout_f) + 1e-10

            logistic = tf.reduce_sum(
                (self.beta * tf.math.softplus(s)
                 + residual_field * tf.math.softplus(-s)) * layout_f
            ) / num

            s_masked = s * layout_f
            s_4d = tf.reshape(s_masked, [1, tf.shape(s)[0], tf.shape(s)[1], 1])
            tv_loss = self.lambda_tv * self._tv(s_4d)

            total = logistic + tv_loss

        grads = tape.gradient(total, self.router.trainable_variables)
        if self.grad_clip_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, self.grad_clip_norm)
        self.optimizer.apply_gradients(
            zip(grads, self.router.trainable_variables))

        rejected = tf.reduce_sum(tf.cast(s > 0, tf.float32) * layout_f) / num
        return total, {
            'total_loss': total,
            'logistic_loss': logistic,
            'tv_loss': tv_loss,
            'reject_fraction': rejected,
        }

    def train(self, inputs, residual_field, layout_mask,
              epochs=2000, lr=1e-3, lr_min=1e-5, verbose=True):
        """Static training data: the router overfits to the current
        (PINN, geometry, k, source) tuple. Cosine-decay LR."""
        inputs_tf = tf.constant(inputs, dtype=tf.float32)
        res_tf = tf.constant(residual_field, dtype=tf.float32)
        layout_tf = tf.constant(layout_mask, dtype=tf.float32)

        self.optimizer.learning_rate.assign(lr)

        for epoch in range(epochs):
            progress = epoch / max(epochs - 1, 1)
            cur_lr = lr_min + 0.5 * (lr - lr_min) * (1 + np.cos(np.pi * progress))
            self.optimizer.learning_rate.assign(cur_lr)

            total, metrics = self.train_step(inputs_tf, res_tf, layout_tf)
            self.loss_history.append(float(metrics['total_loss']))
            self.logistic_loss_history.append(float(metrics['logistic_loss']))
            self.tv_loss_history.append(float(metrics['tv_loss']))

            if verbose and (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}  "
                      f"L={float(metrics['total_loss']):.4f}  "
                      f"logistic={float(metrics['logistic_loss']):.4f}  "
                      f"tv={float(metrics['tv_loss']):.4f}  "
                      f"reject%={float(metrics['reject_fraction'])*100:.1f}  "
                      f"lr={cur_lr:.2e}")

        return {
            'total_loss': self.loss_history,
            'logistic_loss': self.logistic_loss_history,
            'tv_loss': self.tv_loss_history,
        }

    def predict(self, inputs, threshold=0.0):
        inputs = tf.constant(inputs, dtype=tf.float32)
        r = self.router(inputs, training=False)[0, :, :, 0].numpy()
        mask = (r >= threshold).astype(np.int32)
        return r, mask


def median_normalize(residual, layout):
    """Return |residual| / median(|residual| on solid cells)."""
    solid = layout > 0
    vals = np.abs(residual[solid])
    med = float(np.median(vals)) + 1e-10
    return (np.abs(residual) / med).astype(np.float32)
