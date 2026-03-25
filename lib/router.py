"""
CNN-based Router for Hybrid PINN-CFD Simulations.

This module implements a learnable router that decides which regions of a fluid
flow domain should use PINN vs CFD solutions. The router is trained to minimize
computational cost while maintaining solution accuracy.

Architecture:
    Input: 200×100×9 tensor
        - Channel 0: Layout mask (0=obstacle, 1=fluid)
        - Channel 1: Boundary condition mask
        - Channels 2-4: Boundary condition values [u, v, p]
        - Channels 5-7: PINN predictions [u, v, p] (allows CNN to learn error patterns)
        - Channel 8: Directionally-smeared BC error (propagated along PINN velocity)
    
    Output: 200×100 tensor with values in [0, 1]
        - 0: Use PINN solution
        - 1: Use CFD solution
        - Masked to 0 at obstacle locations

Design Philosophy:
    By providing PINN predictions as input, the CNN can learn where PINN is
    reliable vs unreliable WITHOUT hardcoded decay assumptions. The network
    learns spatial error patterns from the data itself.

Loss Function:
    L = β · Σ r(x_i) + Σ (1 - r(x_i)) · L_residual(PINN, x_i) + λ · TV(r)
    
    Where:
        - β: Cost coefficient for invoking CFD
        - r(x_i): Router output at spatial location i
        - L_residual: Physics residual of PINN solution
        - λ: Spatial smoothness regularization weight
        - TV(r): Total variation of router output
"""

import numpy as np
import tensorflow as tf

# Configure TensorFlow GPU memory growth to avoid cuDNN issues
_gpus = tf.config.list_physical_devices('GPU')
if _gpus:
    try:
        for _gpu in _gpus:
            tf.config.experimental.set_memory_growth(_gpu, True)
    except RuntimeError:
        pass  # Memory growth must be set before GPUs are initialized

from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'no-latex'])
from matplotlib.colors import Normalize
import os


class RouterCNN(keras.Model):
    """
    CNN-based router for PINN-CFD domain segregation.

    The architecture uses a U-Net-like structure to preserve spatial resolution
    while learning multi-scale features for optimal domain partitioning.

    Outputs raw logits in R (unbounded). Positive = CFD, negative = PINN.
    """

    def __init__(self, base_filters=32, temperature=1.0, name="router_cnn"):
        """
        Initialize the router CNN.

        Parameters:
        -----------
        base_filters : int
            Number of filters in the first convolutional layer.
            Subsequent layers double this number.
        temperature : float
            Unused, kept for backwards compatibility with saved configs.
        """
        super().__init__(name=name)
        
        # Encoder path
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
        
        # Decoder path
        self.up3 = layers.UpSampling2D(2)
        self.conv5 = layers.Conv2D(base_filters * 4, 3, padding='same', activation='relu')
        self.conv5b = layers.Conv2D(base_filters * 4, 3, padding='same', activation='relu')
        
        self.up2 = layers.UpSampling2D(2)
        self.conv6 = layers.Conv2D(base_filters * 2, 3, padding='same', activation='relu')
        self.conv6b = layers.Conv2D(base_filters * 2, 3, padding='same', activation='relu')
        
        self.up1 = layers.UpSampling2D(2)
        self.conv7 = layers.Conv2D(base_filters, 3, padding='same', activation='relu')
        self.conv7b = layers.Conv2D(base_filters, 3, padding='same', activation='relu')
        
        # Output layer - raw logits, no activation
        # Initialize bias to 0 so initial outputs are around 0
        self.output_conv = layers.Conv2D(
            1, 1, padding='same', activation=None,
            kernel_initializer='glorot_uniform',
            bias_initializer='zeros'
        )
    
    def call(self, inputs, training=None):
        """
        Forward pass through the router.
        
        Parameters:
        -----------
        inputs : tf.Tensor
            Input tensor of shape (batch, height, width, 5)
            Contains layout, BC mask, and BC values
        training : bool, optional
            Whether in training mode
            
        Returns:
        --------
        tf.Tensor
            Router output of shape (batch, height, width, 1)
            Raw logits in R, masked to 0 at obstacle locations.
            Positive = CFD, negative = PINN.
        """
        # Extract layout mask for final masking
        layout = inputs[..., 0:1]  # Shape: (batch, H, W, 1)

        # Encoder
        e1 = self.conv1b(self.conv1(inputs))
        p1 = self.pool1(e1)

        e2 = self.conv2b(self.conv2(p1))
        p2 = self.pool2(e2)

        e3 = self.conv3b(self.conv3(p2))
        p3 = self.pool3(e3)

        # Bottleneck
        b = self.conv4b(self.conv4(p3))

        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = self._match_size(d3, e3)
        d3 = layers.Concatenate()([d3, e3])
        d3 = self.conv5b(self.conv5(d3))

        d2 = self.up2(d3)
        d2 = self._match_size(d2, e2)
        d2 = layers.Concatenate()([d2, e2])
        d2 = self.conv6b(self.conv6(d2))

        d1 = self.up1(d2)
        d1 = self._match_size(d1, e1)
        d1 = layers.Concatenate()([d1, e1])
        d1 = self.conv7b(self.conv7(d1))

        # Output raw logits (no sigmoid)
        logits = self.output_conv(d1)

        # Mask output: 0 at obstacle locations
        output = logits * layout

        return output
    
    def _match_size(self, x, target):
        """Resize x to match target spatial dimensions using tf ops only."""
        # Always resize to target shape - works in graph mode
        target_h = tf.shape(target)[1]
        target_w = tf.shape(target)[2]
        x = tf.image.resize(x, [target_h, target_w], method='bilinear')
        return x


class PINNResidualComputer:
    """
    Computes PINN physics residuals for the Navier-Stokes equations.
    
    Used to evaluate how well the PINN solution satisfies the governing equations
    at each spatial point. Includes boundary condition error propagation.
    """
    
    def __init__(self, pinn_model, nu=0.01, rho=1.0,
                 x_domain=(0, 2), y_domain=(0, 1),
                 cylinder_center=(0.5, 0.5), cylinder_radius=0.1,
                 inlet_velocity=1.0):
        """
        Initialize the residual computer.
        
        Parameters:
        -----------
        pinn_model : tf.keras.Model
            Pre-trained PINN model that outputs (u, v, p) given (x, y)
        nu : float
            Kinematic viscosity
        rho : float
            Fluid density
        x_domain, y_domain : tuple
            Domain bounds
        cylinder_center : tuple
            Cylinder center coordinates
        cylinder_radius : float
            Cylinder radius
        inlet_velocity : float
            Inlet velocity magnitude
        """
        self.pinn_model = pinn_model
        self.nu = nu
        self.rho = rho
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.cylinder_center = cylinder_center
        self.cylinder_radius = cylinder_radius
        self.inlet_velocity = inlet_velocity
    
    @tf.function
    def compute_residuals(self, x, y):
        """
        Compute continuity and momentum residuals at given coordinates.
        
        Parameters:
        -----------
        x, y : tf.Tensor
            Coordinate tensors of shape (N,) or (H, W)
            
        Returns:
        --------
        continuity_residual : tf.Tensor
            |∂u/∂x + ∂v/∂y| at each point
        momentum_residual : tf.Tensor
            √(r_u² + r_v²) at each point
        """
        # Flatten coordinates if needed
        original_shape = tf.shape(x)
        x_flat = tf.reshape(x, [-1])
        y_flat = tf.reshape(y, [-1])
        
        # Stack coordinates for PINN input
        xy = tf.stack([x_flat, y_flat], axis=-1)
        xy = tf.cast(xy, tf.float32)
        
        # Compute derivatives using automatic differentiation
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(xy)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(xy)
                uvp = self.pinn_model(xy, training=False)
                u = uvp[:, 0]
                v = uvp[:, 1]
                p = uvp[:, 2]
            
            # First derivatives
            grad_u = tape1.gradient(u, xy)  # Shape: (N, 2)
            grad_v = tape1.gradient(v, xy)
            grad_p = tape1.gradient(p, xy)
            
            du_dx = grad_u[:, 0]
            du_dy = grad_u[:, 1]
            dv_dx = grad_v[:, 0]
            dv_dy = grad_v[:, 1]
            dp_dx = grad_p[:, 0]
            dp_dy = grad_p[:, 1]
        
        # Second derivatives
        grad_du_dx = tape2.gradient(du_dx, xy)
        grad_du_dy = tape2.gradient(du_dy, xy)
        grad_dv_dx = tape2.gradient(dv_dx, xy)
        grad_dv_dy = tape2.gradient(dv_dy, xy)
        
        d2u_dx2 = grad_du_dx[:, 0] if grad_du_dx is not None else tf.zeros_like(u)
        d2u_dy2 = grad_du_dy[:, 1] if grad_du_dy is not None else tf.zeros_like(u)
        d2v_dx2 = grad_dv_dx[:, 0] if grad_dv_dx is not None else tf.zeros_like(v)
        d2v_dy2 = grad_dv_dy[:, 1] if grad_dv_dy is not None else tf.zeros_like(v)
        
        del tape1, tape2
        
        # Continuity residual: |∂u/∂x + ∂v/∂y|
        continuity = tf.abs(du_dx + dv_dy)
        
        # Momentum residuals
        # r_u = ρ(u ∂u/∂x + v ∂u/∂y) + ∂p/∂x - ν(∂²u/∂x² + ∂²u/∂y²)
        # r_v = ρ(u ∂v/∂x + v ∂v/∂y) + ∂p/∂y - ν(∂²v/∂x² + ∂²v/∂y²)
        r_u = (self.rho * (u * du_dx + v * du_dy) + dp_dx 
               - self.nu * (d2u_dx2 + d2u_dy2))
        r_v = (self.rho * (u * dv_dx + v * dv_dy) + dp_dy 
               - self.nu * (d2v_dx2 + d2v_dy2))
        
        momentum = tf.sqrt(r_u**2 + r_v**2 + 1e-10)
        
        # Reshape to original spatial dimensions
        continuity = tf.reshape(continuity, original_shape)
        momentum = tf.reshape(momentum, original_shape)
        
        return continuity, momentum
    
    def compute_total_residual(self, x, y, weights={'continuity': 1.0, 'momentum': 1.0}):
        """
        Compute weighted total residual.
        
        Parameters:
        -----------
        x, y : tf.Tensor
            Coordinate tensors
        weights : dict
            Weights for continuity and momentum residuals
            
        Returns:
        --------
        total_residual : tf.Tensor
            Weighted sum of residuals at each point
        """
        continuity, momentum = self.compute_residuals(x, y)
        
        # Normalize residuals
        continuity_norm = continuity / (tf.reduce_mean(continuity) + 1e-10)
        momentum_norm = momentum / (tf.reduce_mean(momentum) + 1e-10)
        
        total = (weights['continuity'] * continuity_norm + 
                 weights['momentum'] * momentum_norm)
        
        return total
    
    def get_pinn_predictions(self, X, Y):
        """Get PINN u, v, p predictions for entire field."""
        xy = tf.stack([tf.reshape(X, [-1]), tf.reshape(Y, [-1])], axis=-1)
        xy = tf.cast(xy, tf.float32)
        uvp = self.pinn_model(xy, training=False)
        
        shape = tf.shape(X)
        u = tf.reshape(uvp[:, 0], shape)
        v = tf.reshape(uvp[:, 1], shape)
        p = tf.reshape(uvp[:, 2], shape)
        return u, v, p
    
    def compute_bc_error(self, X, Y, bc_mask, bc_u, bc_v):
        """
        Compute boundary condition error at BC locations.
        
        Parameters:
        -----------
        X, Y : tf.Tensor
            Coordinate grids (Ny, Nx)
        bc_mask : tf.Tensor
            Mask where BCs are specified (1 = BC, 0 = no BC)
        bc_u, bc_v : tf.Tensor
            Prescribed BC values for velocity
            
        Returns:
        --------
        bc_error : tf.Tensor
            BC error at each point (0 where no BC)
        """
        u, v, p = self.get_pinn_predictions(X, Y)
        
        # Error is difference between PINN prediction and prescribed BC
        u_error = tf.abs(u - bc_u) * bc_mask
        v_error = tf.abs(v - bc_v) * bc_mask
        
        bc_error = tf.sqrt(u_error**2 + v_error**2 + 1e-10)
        return bc_error
    
    def compute_upstream_propagated_error(self, X, Y, bc_mask, bc_u, bc_v):
        """
        Compute BC error without hardcoded decay propagation.
        
        NOTE: This function now returns just the local BC error. The CNN router
        receives PINN predictions as input channels, allowing it to learn 
        spatial error patterns WITHOUT hardcoded decay assumptions.
        
        Previously, this function manually propagated errors downstream with
        exp(-1.5 * x) decay. Now the CNN learns these patterns from data.
        
        Parameters:
        -----------
        X, Y : tf.Tensor
            Coordinate grids (Ny, Nx)
        bc_mask : tf.Tensor
            Mask where BCs are specified
        bc_u, bc_v : tf.Tensor
            Prescribed BC values
            
        Returns:
        --------
        bc_error : tf.Tensor
            Local BC error at each point (no artificial propagation)
        """
        # Just return local BC error - let CNN learn spatial patterns
        bc_error = self.compute_bc_error(X, Y, bc_mask, bc_u, bc_v)
        return bc_error
    
    def compute_total_residual_with_bc(self, X, Y, bc_mask, bc_u, bc_v,
                                       weights=None):
        """
        Compute total PDE residual from PINN predictions.

        Returns the weighted sum of continuity and momentum residuals,
        normalized by the domain median. This is robust to the
        heavy-tailed, right-skewed distribution typical of PDE residuals.
        β is interpretable as multiples of the median error.

        Parameters:
        -----------
        X, Y : tf.Tensor
            Coordinate grids
        bc_mask, bc_u, bc_v : tf.Tensor
            Unused, kept for API compatibility
        weights : dict
            Weights for: continuity, momentum

        Returns:
        --------
        total_residual : tf.Tensor
            PDE residual at each point, mean-normalized (R̄ ≈ 1)
        """
        if weights is None:
            weights = {
                'continuity': 1.0,
                'momentum': 1.0,
            }

        continuity, momentum = self.compute_residuals(X, Y)

        total = (weights.get('continuity', 1.0) * continuity +
                 weights.get('momentum', 1.0) * momentum)

        return total


class RouterTrainer:
    """
    Training manager for the CNN router.

    Implements the training loop with the logistic loss:
    L = 1/N * Σ [β · φ(s,0) + R(x) · φ(s,1)] + λ_tv · TV(s)

    Where:
    - s = r(x) is the raw router output (unbounded, in R)
    - φ(s,j) = log(1 + exp(-[2j-1]·s))  (logistic cost)
    - R(x) is the mean-normalized PDE residual at point x (R̄ = 1)
    - β is the cost of using CFD; router rejects PINN where R(x) > β
    """

    def __init__(self, router, pinn_model,
                 beta=0.1, lambda_tv=0.01,
                 lambda_entropy=0.1,
                 grad_clip_norm=1.0,
                 residual_weights=None,
                 nu=0.01, rho=1.0,
                 x_domain=(0, 2), y_domain=(0, 1),
                 cylinder_center=(0.5, 0.5), cylinder_radius=0.1,
                 inlet_velocity=1.0):
        """
        Initialize the trainer.

        Parameters:
        -----------
        router : RouterCNN
            The router model to train
        pinn_model : tf.keras.Model
            Pre-trained PINN model
        beta : float
            Cost coefficient for CFD usage (higher = less CFD)
        lambda_tv : float
            Weight for total variation regularization (spatial smoothness)
        lambda_entropy : float
            Unused, kept for API compatibility.
        grad_clip_norm : float
            Maximum gradient norm for clipping (stabilizes training).
            Set to None to disable. Recommended: 1.0-5.0.
        residual_weights : dict
            Weights for: continuity, momentum
        nu : float
            Kinematic viscosity
        rho : float
            Fluid density
        x_domain, y_domain : tuple
            Domain bounds
        cylinder_center : tuple
            Cylinder center
        cylinder_radius : float
            Cylinder radius
        inlet_velocity : float
            Inlet velocity
        """
        self.router = router
        self.pinn_model = pinn_model
        self.beta = beta
        self.lambda_tv = lambda_tv
        self.grad_clip_norm = grad_clip_norm
        
        # Default residual weights (PDE residuals only)
        self.residual_weights = residual_weights or {
            'continuity': 1.0,
            'momentum': 1.0,
        }
        
        # Initialize residual computer with domain info
        self.residual_computer = PINNResidualComputer(
            pinn_model, nu, rho,
            x_domain=x_domain,
            y_domain=y_domain,
            cylinder_center=cylinder_center,
            cylinder_radius=cylinder_radius,
            inlet_velocity=inlet_velocity
        )
        
        # Optimizer (learning rate set in train() via schedule)
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)
        
        # Metrics
        self.loss_history = []
        self.logistic_loss_history = []
        self.tv_loss_history = []

    def compute_total_variation(self, r):
        """
        Compute total variation of router output for spatial smoothness.
        
        TV(r) = Σ |r(x_i) - r(x_j)| for adjacent pixels
        
        Parameters:
        -----------
        r : tf.Tensor
            Router output of shape (batch, H, W, 1)
            
        Returns:
        --------
        tv : tf.Tensor
            Total variation loss (scalar)
        """
        # Horizontal variation
        tv_h = tf.reduce_mean(tf.abs(r[:, :, 1:, :] - r[:, :, :-1, :]))
        # Vertical variation
        tv_v = tf.reduce_mean(tf.abs(r[:, 1:, :, :] - r[:, :-1, :, :]))
        
        return tv_h + tv_v
    
    @tf.function
    def train_step(self, inputs, X, Y, layout_mask):
        """
        Perform one training step.

        Parameters:
        -----------
        inputs : tf.Tensor
            Router input of shape (batch, H, W, 8)
        X, Y : tf.Tensor
            Coordinate grids of shape (H, W)
        layout_mask : tf.Tensor
            Fluid domain mask of shape (H, W), 1=fluid, 0=obstacle

        Returns:
        --------
        loss : tf.Tensor
            Total loss value
        metrics : dict
            Dictionary of individual loss components
        """
        # Extract BC info from inputs
        bc_mask = inputs[0, :, :, 1]
        bc_u = inputs[0, :, :, 2]
        bc_v = inputs[0, :, :, 3]
        # Extract smeared BC error (channel 8) if present
        smeared_bc_err = inputs[0, :, :, 8] if inputs.shape[-1] > 8 else tf.zeros_like(bc_mask)

        with tf.GradientTape() as tape:
            # Forward pass: raw logits in R
            s = self.router(inputs, training=True)  # Shape: (batch, H, W, 1)
            s = s[0, :, :, 0]  # Remove batch and channel dims: (H, W)

            layout_f = tf.cast(layout_mask, tf.float32)
            num_fluid = tf.reduce_sum(layout_f) + 1e-10

            # Compute raw PINN PDE residuals (unnormalized)
            pde_residual = self.residual_computer.compute_total_residual_with_bc(
                X, Y, bc_mask, bc_u, bc_v, self.residual_weights
            )

            # Sum raw PDE residual and BC error, then median-normalize
            raw_residual = pde_residual + smeared_bc_err
            residual_flat = tf.reshape(raw_residual, [-1])
            residual_median = tf.sort(residual_flat)[tf.shape(residual_flat)[0] // 2]
            total_residual = raw_residual / (residual_median + 1e-10)

            # Logistic loss: 1/N * sum(beta * phi(s,0) + R(x) * phi(s,1))
            # Decision boundary: router assigns CFD where R(x) > beta
            logistic_loss = tf.reduce_sum(
                (self.beta * tf.math.softplus(s) +
                 total_residual * tf.math.softplus(-s)) * layout_f
            ) / num_fluid

            # Total variation regularization (spatial smoothness)
            s_masked = s * layout_f
            s_4d = tf.reshape(s_masked, [1, tf.shape(s)[0], tf.shape(s)[1], 1])
            tv_loss = self.lambda_tv * self.compute_total_variation(s_4d)

            # Total loss
            total_loss = logistic_loss + tv_loss

        # Compute gradients
        gradients = tape.gradient(total_loss, self.router.trainable_variables)

        # Gradient clipping for stability
        if self.grad_clip_norm is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip_norm)

        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.router.trainable_variables))

        # Compute CFD fraction for logging (sigmoid of s gives probability)
        cfd_fraction = tf.reduce_sum(tf.sigmoid(s) * layout_f) / num_fluid

        metrics = {
            'total_loss': total_loss,
            'logistic_loss': logistic_loss,
            'tv_loss': tv_loss,
            'cfd_fraction': cfd_fraction,
        }

        return total_loss, metrics
    
    def train(self, inputs, X, Y, layout_mask, epochs=100, verbose=True,
              lr=1e-3, lr_min=1e-5):
        """
        Train the router for multiple epochs with cosine LR schedule.

        Parameters:
        -----------
        inputs : tf.Tensor or np.ndarray
            Router input of shape (1, H, W, 8)
        X, Y : np.ndarray
            Coordinate grids of shape (H, W)
        layout_mask : np.ndarray
            Fluid domain mask (1=fluid, 0=obstacle)
        epochs : int
            Number of training epochs
        verbose : bool
            Whether to print progress
        lr : float
            Initial learning rate
        lr_min : float
            Minimum learning rate at end of cosine schedule

        Returns:
        --------
        history : dict
            Training history with loss components
        """
        # Convert to tensors
        inputs = tf.constant(inputs, dtype=tf.float32)
        X = tf.constant(X, dtype=tf.float32)
        Y = tf.constant(Y, dtype=tf.float32)
        layout_mask = tf.constant(layout_mask, dtype=tf.float32)

        # Set initial learning rate
        self.optimizer.learning_rate.assign(lr)

        for epoch in range(epochs):
            # Cosine annealing: lr decays from lr to lr_min
            progress = epoch / max(epochs - 1, 1)
            current_lr = lr_min + 0.5 * (lr - lr_min) * (1 + np.cos(np.pi * progress))
            self.optimizer.learning_rate.assign(current_lr)

            loss, metrics = self.train_step(inputs, X, Y, layout_mask)

            # Record history
            self.loss_history.append(float(metrics['total_loss']))
            self.logistic_loss_history.append(float(metrics['logistic_loss']))
            self.tv_loss_history.append(float(metrics['tv_loss']))

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {metrics['total_loss']:.4f}, "
                      f"Logistic: {metrics['logistic_loss']:.4f}, "
                      f"TV: {metrics['tv_loss']:.4f}, "
                      f"CFD%: {metrics['cfd_fraction']*100:.1f}%, "
                      f"lr: {current_lr:.2e}")

        history = {
            'total_loss': self.loss_history,
            'logistic_loss': self.logistic_loss_history,
            'tv_loss': self.tv_loss_history,
        }

        return history
    
    def predict(self, inputs, threshold=0.0):
        """
        Get router prediction and binary mask.

        Parameters:
        -----------
        inputs : tf.Tensor or np.ndarray
            Router input of shape (1, H, W, 8)
        threshold : float
            Threshold for binary mask (default: 0.0, positive=CFD)

        Returns:
        --------
        r : np.ndarray
            Raw router logits (H, W), positive=CFD, negative=PINN
        mask : np.ndarray
            Binary mask (H, W), 1=CFD, 0=PINN
        """
        inputs = tf.constant(inputs, dtype=tf.float32)
        r = self.router(inputs, training=False)
        r = r[0, :, :, 0].numpy()

        mask = (r >= threshold).astype(np.int32)

        return r, mask


def compute_bc_error_field(bc_mask, bc_u, bc_v, pinn_u, pinn_v, layout):
    """
    Compute local BC error at boundary points.

    This measures how well the PINN satisfies the prescribed boundary
    conditions. Nonzero only at boundary locations.

    Parameters:
    -----------
    bc_mask : np.ndarray (H, W)
        1 where BCs are prescribed, 0 elsewhere
    bc_u, bc_v : np.ndarray (H, W)
        Prescribed BC velocity values
    pinn_u, pinn_v : np.ndarray (H, W)
        PINN predicted velocities
    layout : np.ndarray (H, W)
        Fluid mask (1=fluid, 0=obstacle)

    Returns:
    --------
    bc_error : np.ndarray (H, W)
        Local BC error field (nonzero only at boundary points)
    """
    bc_error = np.sqrt((pinn_u - bc_u)**2 + (pinn_v - bc_v)**2) * bc_mask
    return (bc_error * layout).astype(np.float32)


# Keep old name as alias for backwards compatibility
compute_smeared_bc_error = compute_bc_error_field


def solve_error_transport(pinn_u, pinn_v, bc_error, layout, nu,
                          x_domain, y_domain, n_iters=100):
    """
    Solve steady-state error transport equation:
        u·∇e = ν∇²e
    with Dirichlet BC: e = bc_error at boundary points.

    Propagates BC errors through the domain using the PINN velocity field
    and physical diffusion. The decay is determined by the physics
    (Re, velocity field), not prescribed.

    Parameters
    ----------
    pinn_u, pinn_v : np.ndarray (Ny, Nx)
        PINN velocity field (advection).
    bc_error : np.ndarray (Ny, Nx)
        Local BC error (Dirichlet values at boundary points).
    layout : np.ndarray (Ny, Nx)
        Fluid mask (1=fluid, 0=solid).
    nu : float
        Kinematic viscosity.
    x_domain, y_domain : tuple (min, max)
        Physical domain extents.
    n_iters : int
        Jacobi iterations (default 100).

    Returns
    -------
    e : np.ndarray (Ny, Nx)
        Transported error field (float32).
    """
    Ny, Nx = layout.shape
    dx = (x_domain[1] - x_domain[0]) / max(Nx - 1, 1)
    dy = (y_domain[1] - y_domain[0]) / max(Ny - 1, 1)

    # Upwind decomposition
    u_pos = np.maximum(pinn_u, 0.0)
    u_neg = np.minimum(pinn_u, 0.0)
    v_pos = np.maximum(pinn_v, 0.0)
    v_neg = np.minimum(pinn_v, 0.0)

    # Discretisation coefficients (all non-negative → unconditionally stable)
    a_W = u_pos / dx + nu / dx**2          # west  (j-1)
    a_E = -u_neg / dx + nu / dx**2         # east  (j+1)
    a_S = v_pos / dy + nu / dy**2          # south (i-1)
    a_N = -v_neg / dy + nu / dy**2         # north (i+1)
    a_P = a_W + a_E + a_S + a_N            # centre

    bc_src = bc_error > 0                   # source-point mask
    e = bc_error.copy().astype(np.float64)

    for _ in range(n_iters):
        # Neighbour values (zero-padded at domain edges)
        e_W = np.zeros_like(e);  e_W[:, 1:]  = e[:, :-1]
        e_E = np.zeros_like(e);  e_E[:, :-1] = e[:, 1:]
        e_S = np.zeros_like(e);  e_S[1:, :]  = e[:-1, :]
        e_N = np.zeros_like(e);  e_N[:-1, :] = e[1:, :]

        e_new = (a_W * e_W + a_E * e_E + a_S * e_S + a_N * e_N) / (a_P + 1e-10)

        # Dirichlet at BC-error points, zero in obstacles
        e_new = np.where(bc_src, bc_error, e_new)
        e_new = np.maximum(e_new * layout, 0.0)
        e = e_new

    return e.astype(np.float32)


def create_router_input(layout, bc_mask, bc_values_u, bc_values_v, bc_values_p,
                        pinn_u=None, pinn_v=None, pinn_p=None,
                        smeared_bc_error=None):
    """
    Create the 9-channel input tensor for the router.

    By including PINN predictions and smeared BC error, the CNN can learn
    spatial error patterns and understand downstream error propagation.

    Parameters:
    -----------
    layout : np.ndarray
        Layout mask of shape (H, W), 0=obstacle, 1=fluid
    bc_mask : np.ndarray
        Boundary condition mask of shape (H, W)
    bc_values_u, bc_values_v, bc_values_p : np.ndarray
        Boundary condition values of shape (H, W)
    pinn_u, pinn_v, pinn_p : np.ndarray, optional
        PINN predictions of shape (H, W). If None, zeros are used.
    smeared_bc_error : np.ndarray, optional
        Directionally-smeared BC error of shape (H, W). If None, zeros are used.

    Returns:
    --------
    inputs : np.ndarray
        Stacked input of shape (1, H, W, 9)
        Channels: [layout, bc_mask, bc_u, bc_v, bc_p, pinn_u, pinn_v, pinn_p, smeared_bc_error]
    """
    H, W = layout.shape

    # Default to zeros if PINN predictions not provided
    if pinn_u is None:
        pinn_u = np.zeros((H, W), dtype=np.float32)
    if pinn_v is None:
        pinn_v = np.zeros((H, W), dtype=np.float32)
    if pinn_p is None:
        pinn_p = np.zeros((H, W), dtype=np.float32)
    if smeared_bc_error is None:
        smeared_bc_error = np.zeros((H, W), dtype=np.float32)

    inputs = np.stack([
        layout,           # Ch 0: Layout mask
        bc_mask,          # Ch 1: BC mask
        bc_values_u,      # Ch 2: BC u
        bc_values_v,      # Ch 3: BC v
        bc_values_p,      # Ch 4: BC p
        pinn_u,           # Ch 5: PINN u prediction
        pinn_v,           # Ch 6: PINN v prediction
        pinn_p,           # Ch 7: PINN p prediction
        smeared_bc_error, # Ch 8: Directionally-smeared BC error
    ], axis=-1)

    return inputs[np.newaxis, ...]  # Add batch dimension


def create_cylinder_setup(Nx=200, Ny=100, x_domain=(0, 2), y_domain=(0, 1),
                          cylinder_center=(0.5, 0.5), cylinder_radius=0.1,
                          inlet_velocity=1.0):
    """
    Create the standard cylinder flow setup for router training.
    
    Parameters:
    -----------
    Nx, Ny : int
        Grid dimensions
    x_domain, y_domain : tuple
        Domain bounds
    cylinder_center : tuple
        Cylinder center coordinates
    cylinder_radius : float
        Cylinder radius
    inlet_velocity : float
        Inlet velocity magnitude
        
    Returns:
    --------
    X, Y : np.ndarray
        Coordinate grids of shape (Ny, Nx)
    layout : np.ndarray
        Layout mask (1=fluid, 0=obstacle)
    bc_mask : np.ndarray
        Boundary condition mask
    bc_u, bc_v, bc_p : np.ndarray
        Boundary condition values
    """
    x_min, x_max = x_domain
    y_min, y_max = y_domain
    Cx, Cy = cylinder_center
    
    # Create coordinate grids
    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    X, Y = np.meshgrid(x, y)  # Shape: (Ny, Nx)
    
    # Layout: 1 = fluid, 0 = obstacle (cylinder)
    dist_sq = (X - Cx)**2 + (Y - Cy)**2
    layout = (dist_sq > cylinder_radius**2).astype(np.float32)
    
    # Boundary condition mask and values
    bc_mask = np.zeros((Ny, Nx), dtype=np.float32)
    bc_u = np.zeros((Ny, Nx), dtype=np.float32)
    bc_v = np.zeros((Ny, Nx), dtype=np.float32)
    bc_p = np.zeros((Ny, Nx), dtype=np.float32)
    
    # Inlet (left boundary, x = x_min): parabolic profile
    Ly = y_max - y_min
    y_inlet = y
    u_inlet = 4 * inlet_velocity * (y_inlet - y_min) * (y_max - y_inlet) / (Ly**2)
    bc_mask[:, 0] = 1.0
    bc_u[:, 0] = u_inlet
    bc_v[:, 0] = 0.0
    
    # Top wall (y = y_max): no-slip
    bc_mask[-1, :] = 1.0
    bc_u[-1, :] = 0.0
    bc_v[-1, :] = 0.0
    
    # Bottom wall (y = y_min): no-slip
    bc_mask[0, :] = 1.0
    bc_u[0, :] = 0.0
    bc_v[0, :] = 0.0
    
    # Cylinder surface: no-slip
    # Find points adjacent to cylinder
    cylinder_mask = (dist_sq <= cylinder_radius**2)
    from scipy import ndimage
    dilated = ndimage.binary_dilation(cylinder_mask)
    cylinder_boundary = dilated & ~cylinder_mask
    bc_mask[cylinder_boundary] = 1.0
    bc_u[cylinder_boundary] = 0.0
    bc_v[cylinder_boundary] = 0.0
    
    # Outlet: we don't enforce BC here (zero gradient)
    # But mark it for reference
    # bc_mask[:, -1] = 1.0  # Uncomment if needed
    
    return X, Y, layout, bc_mask, bc_u, bc_v, bc_p


def plot_router_output(r, X, Y, layout, title='Router Output', 
                       save_path=None, show_circle=None):
    """
    Visualize the router output.
    
    Parameters:
    -----------
    r : np.ndarray
        Router output of shape (Ny, Nx)
    X, Y : np.ndarray
        Coordinate grids
    layout : np.ndarray
        Layout mask
    title : str
        Plot title
    save_path : str, optional
        Path to save figure
    show_circle : tuple, optional
        (cx, cy, radius) for cylinder visualization
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Continuous router output (raw logits)
    ax = axes[0]
    r_masked = np.ma.masked_where(layout == 0, r)
    vmax = max(abs(np.nanmin(r[layout == 1])), abs(np.nanmax(r[layout == 1])), 1e-6)
    cf = ax.contourf(X, Y, r_masked, levels=50, cmap='RdBu_r',
                     norm=Normalize(vmin=-vmax, vmax=vmax))
    plt.colorbar(cf, ax=ax, label='Router Logit')
    if show_circle:
        cx, cy, radius = show_circle
        circle = plt.Circle((cx, cy), radius, color='gray', fill=True)
        ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Router Logits\n(neg=PINN, pos=CFD)')

    # 2. Binary mask (threshold = 0)
    ax = axes[1]
    mask = (r >= 0.0).astype(np.float32)
    mask_masked = np.ma.masked_where(layout == 0, mask)
    cf = ax.contourf(X, Y, mask_masked, levels=[-0.5, 0.5, 1.5],
                     colors=['blue', 'red'], alpha=0.7)
    cbar = plt.colorbar(cf, ax=ax, ticks=[0.0, 1.0])
    cbar.ax.set_yticklabels(['PINN', 'CFD'])
    if show_circle:
        circle = plt.Circle((cx, cy), radius, color='gray', fill=True)
        ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Binary Mask (threshold=0)')

    # 3. Layout with regions
    ax = axes[2]
    combined = np.zeros_like(r)
    combined[layout == 0] = 0  # Obstacle
    combined[(layout == 1) & (r < 0.0)] = 1  # PINN region
    combined[(layout == 1) & (r >= 0.0)] = 2  # CFD region
    cf = ax.contourf(X, Y, combined, levels=[-0.5, 0.5, 1.5, 2.5],
                     colors=['gray', 'blue', 'red'], alpha=0.7)
    cbar = plt.colorbar(cf, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(['Obstacle', 'PINN', 'CFD'])
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


def plot_coverage_evolution(r, layout, save_path=None, title='Coverage Evolution (0%-100%)'):
    """
    Plot router threshold evolution at fixed CFD coverage deciles.

    Parameters:
    -----------
    r : np.ndarray
        Router output logits of shape (H, W)
    layout : np.ndarray
        Layout mask (1=fluid, 0=obstacle)
    save_path : str, optional
        Path to save figure
    title : str
        Figure title

    Returns:
    --------
    metrics : dict
        Dictionary with target_coverage, threshold, actual_coverage arrays
    fig : matplotlib.figure.Figure
        Generated figure
    """
    fluid_mask = layout > 0
    fluid_logits = r[fluid_mask]

    if fluid_logits.size == 0:
        target_cov = np.linspace(0.0, 1.0, 11)
        thresholds = np.zeros_like(target_cov)
        actual_cov = np.zeros_like(target_cov)
    else:
        sorted_logits = np.sort(fluid_logits)[::-1]
        n_fluid = len(sorted_logits)

        target_cov = np.linspace(0.0, 1.0, 11)
        thresholds = []
        actual_cov = []

        for cov in target_cov:
            if cov <= 0.0:
                threshold = sorted_logits[0] + 1e-6
            elif cov >= 1.0:
                threshold = sorted_logits[-1] - 1e-6
            else:
                n_cfd = int(cov * n_fluid)
                n_cfd = max(1, min(n_cfd, n_fluid))
                threshold = sorted_logits[n_cfd - 1]

            achieved = np.mean(fluid_logits >= threshold)
            thresholds.append(float(threshold))
            actual_cov.append(float(achieved))

        thresholds = np.array(thresholds)
        actual_cov = np.array(actual_cov)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    ax = axes[0]
    ax.plot(target_cov * 100.0, thresholds, 'o-', linewidth=2, markersize=5)
    ax.set_xlabel('Target CFD Coverage (%)')
    ax.set_ylabel('Router Threshold')
    ax.set_title('Threshold @ Coverage Deciles')
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(target_cov * 100.0, actual_cov * 100.0, 'o-', linewidth=2, markersize=5,
            label='Achieved')
    ax.plot([0, 100], [0, 100], '--', linewidth=1.5, label='Ideal')
    ax.set_xlabel('Target CFD Coverage (%)')
    ax.set_ylabel('Achieved CFD Coverage (%)')
    ax.set_title('Coverage Tracking')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.suptitle(title, fontsize=12)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved coverage evolution to {save_path}")

    plt.show()

    metrics = {
        'target_coverage': target_cov,
        'threshold': thresholds,
        'actual_coverage': actual_cov,
    }
    return metrics, fig


def plot_training_history(history, save_path=None):
    """
    Plot training loss history.
    
    Parameters:
    -----------
    history : dict
        Training history with loss components
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Total loss
    ax = axes[0]
    ax.plot(history['total_loss'], 'b-', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.grid(True, alpha=0.3)

    # Logistic loss
    ax = axes[1]
    logistic_key = 'logistic_loss' if 'logistic_loss' in history else 'cfd_cost'
    ax.plot(history[logistic_key], 'r-', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Logistic Loss')
    ax.set_title('Logistic Loss')
    ax.grid(True, alpha=0.3)

    # TV loss
    ax = axes[2]
    ax.plot(history['tv_loss'], 'm-', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('TV Loss')
    ax.set_title('Total Variation Regularization')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved training history to {save_path}")
    
    plt.show()
    return fig


def create_cavity_setup(N=100, x_domain=(0, 1), y_domain=(0, 1), lid_velocity=1.0):
    """
    Create the standard lid-driven cavity flow setup for router training.
    
    Parameters:
    -----------
    N : int
        Grid size (N x N)
    x_domain, y_domain : tuple
        Domain bounds
    lid_velocity : float
        Lid velocity (top wall)
        
    Returns:
    --------
    X, Y : np.ndarray
        Coordinate grids of shape (N, N)
    layout : np.ndarray
        Layout mask (all 1s for cavity - no obstacles)
    bc_mask : np.ndarray
        Boundary condition mask
    bc_u, bc_v, bc_p : np.ndarray
        Boundary condition values
    """
    x_min, x_max = x_domain
    y_min, y_max = y_domain
    
    # Create coordinate grids
    x = np.linspace(x_min, x_max, N)
    y = np.linspace(y_min, y_max, N)
    X, Y = np.meshgrid(x, y)  # Shape: (N, N)
    
    # Layout: all fluid (no obstacles in cavity)
    layout = np.ones((N, N), dtype=np.float32)
    
    # Boundary condition mask and values
    bc_mask = np.zeros((N, N), dtype=np.float32)
    bc_u = np.zeros((N, N), dtype=np.float32)
    bc_v = np.zeros((N, N), dtype=np.float32)
    bc_p = np.zeros((N, N), dtype=np.float32)
    
    # Top lid (y = y_max): u = lid_velocity, v = 0
    bc_mask[-1, :] = 1.0
    bc_u[-1, :] = lid_velocity
    bc_v[-1, :] = 0.0
    
    # Bottom wall (y = y_min): no-slip
    bc_mask[0, :] = 1.0
    bc_u[0, :] = 0.0
    bc_v[0, :] = 0.0
    
    # Left wall (x = x_min): no-slip
    bc_mask[:, 0] = 1.0
    bc_u[:, 0] = 0.0
    bc_v[:, 0] = 0.0
    
    # Right wall (x = x_max): no-slip
    bc_mask[:, -1] = 1.0
    bc_u[:, -1] = 0.0
    bc_v[:, -1] = 0.0
    
    return X, Y, layout, bc_mask, bc_u, bc_v, bc_p
