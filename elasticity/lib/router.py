"""
CNN-based Router for Hybrid PINN-FDM Elasticity Simulations.

Decides which regions of the domain should use PINN vs FDM solutions.
Direct analogue of the fluid router in ../lib/router.py.

Architecture:
    Input: Ny x Nx x 9 tensor
        - Channel 0: Layout mask (1=material, 0=void)
        - Channel 1: Displacement BC mask
        - Channel 2-3: Displacement BC values [ux, uy]
        - Channel 4: Traction BC mask
        - Channel 5-6: PINN predictions [ux, uy]
        - Channel 7: PINN von Mises stress
        - Channel 8: BC error field

    Output: Ny x Nx tensor of raw logits
        Positive = FDM, Negative = PINN

Loss Function:
    L = 1/N * sum[beta * softplus(s) + R_bar(x) * softplus(-s)] + lambda_tv * TV(s)

    Where R_bar is the median-normalized equilibrium residual from the PINN.
"""

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
import matplotlib.pyplot as plt
try:
    import scienceplots
    plt.style.use(['science', 'no-latex'])
except ImportError:
    pass
from matplotlib.colors import Normalize
import os


class RouterCNN(keras.Model):
    """
    U-Net CNN router for PINN-FDM domain segregation (elasticity).
    Outputs raw logits: positive = FDM, negative = PINN.
    """

    def __init__(self, base_filters=32, name="router_cnn"):
        super().__init__(name=name)

        # Encoder
        self.conv1 = layers.Conv2D(base_filters, 3, padding='same', activation='relu')
        self.conv1b = layers.Conv2D(base_filters, 3, padding='same', activation='relu')
        self.pool1 = layers.MaxPooling2D(2)

        self.conv2 = layers.Conv2D(base_filters * 2, 3, padding='same', activation='relu')
        self.conv2b = layers.Conv2D(base_filters * 2, 3, padding='same', activation='relu')
        self.pool2 = layers.MaxPooling2D(2)

        self.conv3 = layers.Conv2D(base_filters * 4, 3, padding='same', activation='relu')
        self.conv3b = layers.Conv2D(base_filters * 4, 3, padding='same', activation='relu')
        self.pool3 = layers.MaxPooling2D(2)

        # Bottleneck
        self.conv4 = layers.Conv2D(base_filters * 8, 3, padding='same', activation='relu')
        self.conv4b = layers.Conv2D(base_filters * 8, 3, padding='same', activation='relu')

        # Decoder
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
        target_h = tf.shape(target)[1]
        target_w = tf.shape(target)[2]
        return tf.image.resize(x, [target_h, target_w], method='bilinear')


class PINNResidualComputer:
    """
    Computes PINN equilibrium residuals for linear elasticity.

    Equilibrium:
        d(sigma_xx)/dx + d(sigma_xy)/dy = 0
        d(sigma_xy)/dx + d(sigma_yy)/dy = 0

    Constitutive (plane stress):
        sigma_xx = C11 * du/dx + C12 * dv/dy
        sigma_yy = C12 * du/dx + C11 * dv/dy
        sigma_xy = C66 * (du/dy + dv/dx)
    """

    def __init__(self, pinn_model, E=1.0, nu=0.3):
        self.pinn_model = pinn_model
        self.E = E
        self.nu = nu
        self.C11 = E / (1.0 - nu ** 2)
        self.C12 = nu * E / (1.0 - nu ** 2)
        self.C66 = E / (2.0 * (1.0 + nu))

    @tf.function
    def compute_residuals(self, x, y):
        """
        Compute equilibrium residuals at given coordinates.

        Returns
        -------
        eq_x_residual, eq_y_residual : tf.Tensor
            |d(sigma_xx)/dx + d(sigma_xy)/dy| and
            |d(sigma_xy)/dx + d(sigma_yy)/dy|
        """
        original_shape = tf.shape(x)
        x_flat = tf.reshape(x, [-1])
        y_flat = tf.reshape(y, [-1])
        xy = tf.stack([x_flat, y_flat], axis=-1)
        xy = tf.cast(xy, tf.float32)

        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(xy)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(xy)
                uvp = self.pinn_model(xy, training=False)
                ux = uvp[:, 0]
                uy = uvp[:, 1]

            grad_ux = tape1.gradient(ux, xy)  # (N, 2)
            grad_uy = tape1.gradient(uy, xy)

            dux_dx = grad_ux[:, 0]
            dux_dy = grad_ux[:, 1]
            duy_dx = grad_uy[:, 0]
            duy_dy = grad_uy[:, 1]

            # Stress components (plane stress)
            sxx = self.C11 * dux_dx + self.C12 * duy_dy
            syy = self.C12 * dux_dx + self.C11 * duy_dy
            sxy = self.C66 * (dux_dy + duy_dx)

        # Second derivatives for equilibrium
        grad_sxx = tape2.gradient(sxx, xy)
        grad_syy = tape2.gradient(syy, xy)
        grad_sxy = tape2.gradient(sxy, xy)

        dsxx_dx = grad_sxx[:, 0] if grad_sxx is not None else tf.zeros_like(ux)
        dsxy_dy = grad_sxy[:, 1] if grad_sxy is not None else tf.zeros_like(ux)
        dsxy_dx = grad_sxy[:, 0] if grad_sxy is not None else tf.zeros_like(ux)
        dsyy_dy = grad_syy[:, 1] if grad_syy is not None else tf.zeros_like(ux)

        del tape1, tape2

        # Equilibrium residuals
        eq_x = tf.abs(dsxx_dx + dsxy_dy)
        eq_y = tf.abs(dsxy_dx + dsyy_dy)

        eq_x = tf.reshape(eq_x, original_shape)
        eq_y = tf.reshape(eq_y, original_shape)

        return eq_x, eq_y

    def compute_total_residual(self, x, y, weights=None):
        """Weighted total equilibrium residual."""
        if weights is None:
            weights = {'eq_x': 1.0, 'eq_y': 1.0}
        eq_x, eq_y = self.compute_residuals(x, y)
        total = weights.get('eq_x', 1.0) * eq_x + weights.get('eq_y', 1.0) * eq_y
        return total


class RouterTrainer:
    """
    Training manager for the elasticity router.

    Same logistic loss as the fluid router:
        L = 1/N * sum[beta * softplus(s) + R_bar(x) * softplus(-s)] + lambda_tv * TV(s)
    """

    def __init__(self, router, pinn_model,
                 beta=0.1, lambda_tv=0.01,
                 grad_clip_norm=1.0,
                 residual_weights=None,
                 E=1.0, nu=0.3):
        self.router = router
        self.pinn_model = pinn_model
        self.beta = beta
        self.lambda_tv = lambda_tv
        self.grad_clip_norm = grad_clip_norm
        self.residual_weights = residual_weights or {'eq_x': 1.0, 'eq_y': 1.0}

        self.residual_computer = PINNResidualComputer(pinn_model, E, nu)
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)

        self.loss_history = []
        self.logistic_loss_history = []
        self.tv_loss_history = []

    def compute_total_variation(self, r):
        tv_h = tf.reduce_mean(tf.abs(r[:, :, 1:, :] - r[:, :, :-1, :]))
        tv_v = tf.reduce_mean(tf.abs(r[:, 1:, :, :] - r[:, :-1, :, :]))
        return tv_h + tv_v

    @tf.function
    def train_step(self, inputs, X, Y, layout_mask):
        bc_error = inputs[0, :, :, 8] if inputs.shape[-1] > 8 else tf.zeros_like(layout_mask)

        with tf.GradientTape() as tape:
            s = self.router(inputs, training=True)
            s = s[0, :, :, 0]

            layout_f = tf.cast(layout_mask, tf.float32)
            num_material = tf.reduce_sum(layout_f) + 1e-10

            pde_residual = self.residual_computer.compute_total_residual(
                X, Y, self.residual_weights
            )

            raw_residual = pde_residual + bc_error
            residual_flat = tf.reshape(raw_residual, [-1])
            residual_median = tf.sort(residual_flat)[tf.shape(residual_flat)[0] // 2]
            total_residual = raw_residual / (residual_median + 1e-10)

            logistic_loss = tf.reduce_sum(
                (self.beta * tf.math.softplus(s)
                 + total_residual * tf.math.softplus(-s)) * layout_f
            ) / num_material

            s_masked = s * layout_f
            s_4d = tf.reshape(s_masked, [1, tf.shape(s)[0], tf.shape(s)[1], 1])
            tv_loss = self.lambda_tv * self.compute_total_variation(s_4d)

            total_loss = logistic_loss + tv_loss

        gradients = tape.gradient(total_loss, self.router.trainable_variables)
        if self.grad_clip_norm is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip_norm)
        self.optimizer.apply_gradients(zip(gradients, self.router.trainable_variables))

        fdm_fraction = tf.reduce_sum(tf.sigmoid(s) * layout_f) / num_material

        return total_loss, {
            'total_loss': total_loss,
            'logistic_loss': logistic_loss,
            'tv_loss': tv_loss,
            'fdm_fraction': fdm_fraction,
        }

    def train(self, inputs, X, Y, layout_mask, epochs=100, verbose=True,
              lr=1e-3, lr_min=1e-5):
        inputs = tf.constant(inputs, dtype=tf.float32)
        X = tf.constant(X, dtype=tf.float32)
        Y = tf.constant(Y, dtype=tf.float32)
        layout_mask = tf.constant(layout_mask, dtype=tf.float32)

        self.optimizer.learning_rate.assign(lr)

        for epoch in range(epochs):
            progress = epoch / max(epochs - 1, 1)
            current_lr = lr_min + 0.5 * (lr - lr_min) * (1 + np.cos(np.pi * progress))
            self.optimizer.learning_rate.assign(current_lr)

            loss, metrics = self.train_step(inputs, X, Y, layout_mask)

            self.loss_history.append(float(metrics['total_loss']))
            self.logistic_loss_history.append(float(metrics['logistic_loss']))
            self.tv_loss_history.append(float(metrics['tv_loss']))

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {metrics['total_loss']:.4f}, "
                      f"Logistic: {metrics['logistic_loss']:.4f}, "
                      f"TV: {metrics['tv_loss']:.4f}, "
                      f"FDM%: {metrics['fdm_fraction']*100:.1f}%, "
                      f"lr: {current_lr:.2e}")

        return {
            'total_loss': self.loss_history,
            'logistic_loss': self.logistic_loss_history,
            'tv_loss': self.tv_loss_history,
        }

    def predict(self, inputs, threshold=0.0):
        inputs = tf.constant(inputs, dtype=tf.float32)
        r = self.router(inputs, training=False)
        r = r[0, :, :, 0].numpy()
        mask = (r >= threshold).astype(np.int32)
        return r, mask


def compute_bc_error_field(disp_bc_mask, bc_ux, bc_uy, pinn_ux, pinn_uy, layout):
    """Compute displacement BC error at boundary points."""
    bc_error = np.sqrt((pinn_ux - bc_ux) ** 2 + (pinn_uy - bc_uy) ** 2) * disp_bc_mask
    return (bc_error * layout).astype(np.float32)


def create_router_input(layout, disp_bc_mask, bc_ux, bc_uy, trac_bc_mask,
                         pinn_ux=None, pinn_uy=None, pinn_vm=None,
                         bc_error=None):
    """
    Create the 9-channel input tensor for the elasticity router.

    Channels:
        0: Layout mask (1=material, 0=void)
        1: Displacement BC mask
        2: BC ux values
        3: BC uy values
        4: Traction BC mask
        5: PINN ux prediction
        6: PINN uy prediction
        7: PINN von Mises stress
        8: BC error field
    """
    H, W = layout.shape
    if pinn_ux is None:
        pinn_ux = np.zeros((H, W), dtype=np.float32)
    if pinn_uy is None:
        pinn_uy = np.zeros((H, W), dtype=np.float32)
    if pinn_vm is None:
        pinn_vm = np.zeros((H, W), dtype=np.float32)
    if bc_error is None:
        bc_error = np.zeros((H, W), dtype=np.float32)

    inputs = np.stack([
        layout,
        disp_bc_mask,
        bc_ux,
        bc_uy,
        trac_bc_mask,
        pinn_ux,
        pinn_uy,
        pinn_vm,
        bc_error,
    ], axis=-1)

    return inputs[np.newaxis, ...]


def plot_router_output(r, X, Y, layout, title='Router Output',
                       save_path=None, show_hole=None):
    """Visualize router output for elasticity problems."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Continuous logits
    ax = axes[0]
    r_masked = np.ma.masked_where(layout == 0, r)
    valid = r[layout == 1]
    vmax = max(abs(np.nanmin(valid)), abs(np.nanmax(valid)), 1e-6)
    cf = ax.contourf(X, Y, r_masked, levels=50, cmap='RdBu_r',
                     norm=Normalize(vmin=-vmax, vmax=vmax))
    plt.colorbar(cf, ax=ax, label='Router Logit')
    if show_hole:
        cx, cy, radius = show_hole
        circle = plt.Circle((cx, cy), radius, color='gray', fill=True)
        ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Router Logits\n(neg=PINN, pos=FDM)')

    # 2. Binary mask
    ax = axes[1]
    mask = (r >= 0.0).astype(np.float32)
    mask_masked = np.ma.masked_where(layout == 0, mask)
    cf = ax.contourf(X, Y, mask_masked, levels=[-0.5, 0.5, 1.5],
                     colors=['blue', 'red'], alpha=0.7)
    cbar = plt.colorbar(cf, ax=ax, ticks=[0.0, 1.0])
    cbar.ax.set_yticklabels(['PINN', 'FDM'])
    if show_hole:
        circle = plt.Circle((cx, cy), radius, color='gray', fill=True)
        ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Binary Mask (threshold=0)')

    # 3. Domain segmentation
    ax = axes[2]
    combined = np.zeros_like(r)
    combined[layout == 0] = 0
    combined[(layout == 1) & (r < 0.0)] = 1
    combined[(layout == 1) & (r >= 0.0)] = 2
    cf = ax.contourf(X, Y, combined, levels=[-0.5, 0.5, 1.5, 2.5],
                     colors=['gray', 'blue', 'red'], alpha=0.7)
    cbar = plt.colorbar(cf, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Void', 'PINN', 'FDM'])
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Domain Segmentation')

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved router visualization to {save_path}")

    plt.show()
    return fig


def plot_training_history(history, save_path=None):
    """Plot training loss history."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['total_loss'], 'b-', linewidth=1.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title('Total Loss')
    axes[0].grid(True, alpha=0.3)

    logistic_key = 'logistic_loss' if 'logistic_loss' in history else 'total_loss'
    axes[1].plot(history[logistic_key], 'r-', linewidth=1.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Logistic Loss')
    axes[1].set_title('Logistic Loss')
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(history['tv_loss'], 'm-', linewidth=1.5)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('TV Loss')
    axes[2].set_title('Total Variation')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    plt.show()
    return fig
