"""
CNN-based Router for Hybrid PINN-CFD Simulations.

This module implements a learnable router that decides which regions of a fluid
flow domain should use PINN vs CFD solutions. The router is trained to minimize
computational cost while maintaining solution accuracy.

Architecture:
    Input: 200×100×5 tensor
        - Channel 1: Layout mask (0=obstacle, 1=fluid)
        - Channel 2: Boundary condition mask
        - Channels 3-5: Boundary condition values [u, v, p]
    
    Output: 200×100 tensor with values in [0, 1]
        - 0: Use PINN solution
        - 1: Use CFD solution
        - Masked to 0 at obstacle locations

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
from matplotlib.colors import Normalize
import os


class RouterCNN(keras.Model):
    """
    CNN-based router for PINN-CFD domain segregation.
    
    The architecture uses a U-Net-like structure to preserve spatial resolution
    while learning multi-scale features for optimal domain partitioning.
    
    Uses temperature-scaled sigmoid to control sharpness of decisions:
    - Low temperature (e.g., 0.5): softer decisions, outputs closer to 0.5
    - High temperature (e.g., 2.0): sharper decisions, outputs closer to 0 or 1
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
            Temperature for sigmoid scaling. Lower = softer outputs.
            Start with 0.5-1.0 for stable training, increase for sharper masks.
        """
        super().__init__(name=name)
        
        self.temperature = temperature
        
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
        
        # Output layer - no activation, we apply temperature-scaled sigmoid in call()
        # Initialize bias to 0 so initial outputs are around 0.5
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
            Values in [0, 1], masked by layout
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
        # Handle size mismatch due to pooling
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
        
        # Output logits (no activation yet)
        logits = self.output_conv(d1)
        
        # Temperature-scaled sigmoid: sigmoid(logits / temperature)
        # Lower temperature = softer outputs (closer to 0.5)
        # Higher temperature = sharper outputs (closer to 0 or 1)
        output = tf.sigmoid(logits / self.temperature)
        
        # Mask output: 0 at obstacle locations
        output = output * layout
        
        return output
    
    def set_temperature(self, temperature):
        """Update temperature for inference (e.g., anneal during training)."""
        self.temperature = temperature
    
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
        Compute error that propagates from upstream BC violations.
        
        Key insight: If the PINN gets the inlet wrong, EVERYTHING downstream
        is wrong, even if it locally satisfies PDEs. This function assigns
        high error to all points downstream of BC violations.
        
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
        propagated_error : tf.Tensor
            Error field that includes downstream propagation
        """
        x_min = self.x_domain[0]
        Cx = self.cylinder_center[0]
        
        # Get BC error at boundary locations
        bc_error = self.compute_bc_error(X, Y, bc_mask, bc_u, bc_v)
        
        # For inlet specifically: compute mean inlet error
        # Inlet is first column (x = x_min)
        inlet_error = tf.reduce_mean(bc_error[:, 0])
        
        # Propagate inlet error downstream with decay
        # Points further from inlet still affected but less
        x_dist_from_inlet = X - x_min
        x_max_dist = self.x_domain[1] - x_min
        
        # Decay factor: 1 at inlet, moderate decay
        # At outlet: exp(-0.5) ≈ 0.61, so ~61% of inlet error reaches outlet
        decay = tf.exp(-1.5 * x_dist_from_inlet / x_max_dist)
        
        # Inlet error propagates to all downstream points
        inlet_propagated = inlet_error * decay
        
        # Also propagate from cylinder surface errors
        # Points in wake region are affected by cylinder BC errors
        dist_from_cyl = tf.sqrt((X - Cx)**2 + (Y - self.cylinder_center[1])**2)
        cyl_mask = tf.cast(dist_from_cyl <= self.cylinder_radius * 1.5, tf.float32)
        cyl_bc_error = tf.reduce_sum(bc_error * cyl_mask) / (tf.reduce_sum(cyl_mask) + 1e-10)
        
        # Wake region: downstream of cylinder
        in_wake = tf.cast(X > Cx, tf.float32)
        # Moderate decay in wake: exp(-0.5) ≈ 0.61 at outlet
        wake_decay = tf.exp(-1.5 * (X - Cx) / (self.x_domain[1] - Cx + 1e-10))
        cyl_propagated = cyl_bc_error * in_wake * wake_decay
        
        # Combine: local BC error + propagated errors
        propagated_error = bc_error + inlet_propagated + cyl_propagated
        
        return propagated_error
    
    def compute_total_residual_with_bc(self, X, Y, bc_mask, bc_u, bc_v,
                                       weights=None):
        """
        Compute total residual including BC error propagation.
        
        Residuals are clipped at 95th percentile then normalized,
        producing values in [0, 1] range with outliers capped at 1.
        
        Parameters:
        -----------
        X, Y : tf.Tensor
            Coordinate grids
        bc_mask : tf.Tensor
            Boundary condition mask
        bc_u, bc_v : tf.Tensor
            Prescribed BC values
        weights : dict
            Weights for: continuity, momentum, bc_local, bc_propagated
            
        Returns:
        --------
        total_residual : tf.Tensor
            Comprehensive residual at each point, normalized to [0, 1]
        """
        if weights is None:
            weights = {
                'continuity': 1.0,
                'momentum': 1.0,
                'bc_local': 2.0,       # Local BC error (high weight)
                'bc_propagated': 1.5   # Propagated error from upstream
            }
        
        # Helper function: normalize by percentile then clip at max_scale
        def clip_and_normalize(x, percentile=0.95, max_scale=1.5):
            """Normalize by percentile value, then clip at max_scale."""
            x_flat = tf.reshape(x, [-1])
            k = tf.cast(tf.cast(tf.size(x_flat), tf.float32) * percentile, tf.int32)
            k = tf.maximum(k, 1)
            top_values, _ = tf.nn.top_k(x_flat, k)
            p_val = top_values[-1] + 1e-10
            # Normalize by percentile, then clip at max_scale
            x_norm = x / p_val
            return tf.minimum(x_norm, max_scale)
        
        # Standard PDE residuals
        continuity, momentum = self.compute_residuals(X, Y)
        
        # Clip and normalize each component to [0, 1]
        continuity_norm = clip_and_normalize(continuity)
        momentum_norm = clip_and_normalize(momentum)
        
        pde_residual = (weights.get('continuity', 1.0) * continuity_norm + 
                        weights.get('momentum', 1.0) * momentum_norm)
        
        # BC error with propagation
        if weights.get('bc_local', 0) > 0 or weights.get('bc_propagated', 0) > 0:
            bc_error = self.compute_bc_error(X, Y, bc_mask, bc_u, bc_v)
            bc_error_norm = clip_and_normalize(bc_error)
            
            propagated = self.compute_upstream_propagated_error(X, Y, bc_mask, bc_u, bc_v)
            propagated_norm = clip_and_normalize(propagated)
            
            pde_residual = (pde_residual + 
                           weights.get('bc_local', 0) * bc_error_norm +
                           weights.get('bc_propagated', 0) * propagated_norm)
        
        # Final normalization: divide by total weight sum
        # This makes the weighted average in [0, 1]
        total_weight = (weights.get('continuity', 1.0) + weights.get('momentum', 1.0) +
                       weights.get('bc_local', 0) + weights.get('bc_propagated', 0))
        pde_residual = pde_residual / (total_weight + 1e-10)
        
        return pde_residual


class RouterTrainer:
    """
    Training manager for the CNN router.
    
    Implements the training loop with the loss function:
    L = β · Σ r(x_i) + Σ (1 - r(x_i)) · L_residual + λ_tv · TV(r) 
        - λ_entropy · H(r) + λ_variance · Var(r)
    
    Where:
    - H(r) is binary entropy to encourage diverse (not all 0 or 1) outputs
    - Var(r) is variance penalty to encourage spread in router outputs
    
    Includes BC error propagation to detect upstream errors.
    """
    
    def __init__(self, router, pinn_model, 
                 beta=0.1, lambda_tv=0.01,
                 lambda_entropy=0.1, lambda_variance=0.05,
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
            Weight for entropy regularization (encourages non-extreme outputs).
            Higher = more intermediate values. Try 0.05-0.2.
        lambda_variance : float
            Weight for variance regularization (encourages output spread).
            Higher = more diverse outputs. Try 0.02-0.1.
        grad_clip_norm : float
            Maximum gradient norm for clipping (stabilizes training).
            Set to None to disable. Recommended: 1.0-5.0.
        residual_weights : dict
            Weights for: continuity, momentum, bc_local, bc_propagated
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
        self.lambda_entropy = lambda_entropy
        self.lambda_variance = lambda_variance
        self.grad_clip_norm = grad_clip_norm
        
        # Default residual weights with BC propagation
        self.residual_weights = residual_weights or {
            'continuity': 1.0,
            'momentum': 1.0,
            'bc_local': 2.0,        # Direct BC error (high weight!)
            'bc_propagated': 1.5    # Propagated error from upstream
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
        
        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-4)
        
        # Metrics
        self.loss_history = []
        self.cfd_cost_history = []
        self.residual_loss_history = []
        self.tv_loss_history = []
        self.entropy_history = []
        self.variance_history = []
    
    def compute_binary_entropy(self, r, layout_mask):
        """
        Compute binary entropy of router output.
        
        H(r) = -Σ [r log(r) + (1-r) log(1-r)] / N
        
        Maximum entropy (0.693) at r=0.5, minimum (0) at r=0 or r=1.
        We MAXIMIZE entropy to encourage non-extreme outputs.
        
        Parameters:
        -----------
        r : tf.Tensor
            Router output of shape (H, W), values in [0, 1]
        layout_mask : tf.Tensor
            Fluid domain mask
            
        Returns:
        --------
        entropy : tf.Tensor
            Mean binary entropy (scalar)
        """
        eps = 1e-7  # Prevent log(0)
        r_clipped = tf.clip_by_value(r, eps, 1.0 - eps)
        
        # Binary entropy: -[r*log(r) + (1-r)*log(1-r)]
        entropy_per_point = -(r_clipped * tf.math.log(r_clipped) + 
                              (1.0 - r_clipped) * tf.math.log(1.0 - r_clipped))
        
        # Average over fluid points only
        num_fluid = tf.reduce_sum(layout_mask) + eps
        mean_entropy = tf.reduce_sum(entropy_per_point * layout_mask) / num_fluid
        
        return mean_entropy
    
    def compute_output_variance(self, r, layout_mask):
        """
        Compute variance of router output over the domain.
        
        Low variance = all outputs similar (bad: either all CFD or all PINN)
        High variance = diverse outputs (good: mixed regions)
        
        We MAXIMIZE variance to encourage diverse outputs.
        
        Parameters:
        -----------
        r : tf.Tensor
            Router output of shape (H, W)
        layout_mask : tf.Tensor
            Fluid domain mask
            
        Returns:
        --------
        variance : tf.Tensor
            Variance of router output (scalar)
        """
        eps = 1e-7
        num_fluid = tf.reduce_sum(layout_mask) + eps
        
        # Mean over fluid points
        mean_r = tf.reduce_sum(r * layout_mask) / num_fluid
        
        # Variance over fluid points
        variance = tf.reduce_sum(((r - mean_r) ** 2) * layout_mask) / num_fluid
        
        return variance
    
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
            Router input of shape (batch, H, W, 5)
            Channel 0: layout, Channel 1: bc_mask, Channels 2-4: bc_u, bc_v, bc_p
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
        
        with tf.GradientTape() as tape:
            # Forward pass
            r = self.router(inputs, training=True)  # Shape: (batch, H, W, 1)
            r = r[0, :, :, 0]  # Remove batch and channel dims: (H, W)
            
            # Apply layout mask
            r_masked = r * tf.cast(layout_mask, tf.float32)
            num_fluid = tf.reduce_sum(tf.cast(layout_mask, tf.float32)) + 1e-10
            
            # 1. CFD cost: β · mean(r) → in [0, β]
            # This is the fraction of domain assigned to CFD, scaled by β
            cfd_fraction = tf.reduce_sum(r_masked) / num_fluid  # in [0, 1]
            cfd_cost = self.beta * cfd_fraction
            
            # 2. PINN residual weighted by (1 - r)
            # Compute PINN residuals including BC error propagation
            # Residuals are normalized to [0, 1] by clipping at 95th percentile
            total_residual = self.residual_computer.compute_total_residual_with_bc(
                X, Y, bc_mask, bc_u, bc_v, self.residual_weights
            )
            
            # Normalize total_residual by 95th percentile, then clip at 1.5
            residual_flat = tf.reshape(total_residual, [-1])
            k = tf.cast(tf.cast(tf.size(residual_flat), tf.float32) * 0.95, tf.int32)
            k = tf.maximum(k, 1)
            top_values, _ = tf.nn.top_k(residual_flat, k)
            residual_p95 = top_values[-1] + 1e-10
            # Normalize by p95, then clip at 1.5x
            total_residual_norm = total_residual / residual_p95
            total_residual_norm = tf.minimum(total_residual_norm, 1.5)
            
            # Weight by (1 - r): points assigned to PINN should have low residual
            pinn_weight = (1.0 - r_masked) * tf.cast(layout_mask, tf.float32)
            pinn_fraction = tf.reduce_sum(pinn_weight) / num_fluid  # in [0, 1]
            
            # FIXED: Use mean over all fluid points, not weighted mean
            # This gives proper gradients even when r→0 or r→1
            # residual_loss = mean((1-r) * residual) → in [0, 1]
            residual_loss = tf.reduce_sum(pinn_weight * total_residual_norm) / num_fluid
            
            # 3. Total variation regularization (spatial smoothness)
            r_4d = tf.reshape(r_masked, [1, tf.shape(r_masked)[0], tf.shape(r_masked)[1], 1])
            tv_loss = self.lambda_tv * self.compute_total_variation(r_4d)
            
            # 4. Entropy regularization (encourage non-extreme outputs)
            # We SUBTRACT entropy because we want to MAXIMIZE it
            entropy = self.compute_binary_entropy(r_masked, tf.cast(layout_mask, tf.float32))
            entropy_loss = -self.lambda_entropy * entropy  # Negative = maximize entropy
            
            # 5. Variance regularization (encourage diverse outputs)
            # We SUBTRACT variance because we want to MAXIMIZE it
            variance = self.compute_output_variance(r_masked, tf.cast(layout_mask, tf.float32))
            variance_loss = -self.lambda_variance * variance  # Negative = maximize variance
            
            # Total loss
            total_loss = cfd_cost + residual_loss + tv_loss + entropy_loss + variance_loss
        
        # Compute gradients
        gradients = tape.gradient(total_loss, self.router.trainable_variables)
        
        # Gradient clipping for stability
        if self.grad_clip_norm is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip_norm)
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.router.trainable_variables))
        
        metrics = {
            'total_loss': total_loss,
            'cfd_cost': cfd_cost,
            'residual_loss': residual_loss,
            'tv_loss': tv_loss,
            'entropy': entropy,
            'variance': variance,
            'cfd_fraction': cfd_fraction,
            'pinn_fraction': pinn_fraction
        }
        
        return total_loss, metrics
    
    def train(self, inputs, X, Y, layout_mask, epochs=100, verbose=True):
        """
        Train the router for multiple epochs.
        
        Parameters:
        -----------
        inputs : tf.Tensor or np.ndarray
            Router input of shape (1, H, W, 5)
        X, Y : np.ndarray
            Coordinate grids of shape (H, W)
        layout_mask : np.ndarray
            Fluid domain mask (1=fluid, 0=obstacle)
        epochs : int
            Number of training epochs
        verbose : bool
            Whether to print progress
            
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
        
        for epoch in range(epochs):
            loss, metrics = self.train_step(inputs, X, Y, layout_mask)
            
            # Record history
            self.loss_history.append(float(metrics['total_loss']))
            self.cfd_cost_history.append(float(metrics['cfd_cost']))
            self.residual_loss_history.append(float(metrics['residual_loss']))
            self.tv_loss_history.append(float(metrics['tv_loss']))
            self.entropy_history.append(float(metrics['entropy']))
            self.variance_history.append(float(metrics['variance']))
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {metrics['total_loss']:.4f}, "
                      f"CFD: {metrics['cfd_cost']:.4f}, "
                      f"Res: {metrics['residual_loss']:.4f}, "
                      f"Ent: {metrics['entropy']:.3f}, "
                      f"Var: {metrics['variance']:.3f}, "
                      f"CFD%: {metrics['cfd_fraction']*100:.1f}%")
        
        history = {
            'total_loss': self.loss_history,
            'cfd_cost': self.cfd_cost_history,
            'residual_loss': self.residual_loss_history,
            'tv_loss': self.tv_loss_history,
            'entropy': self.entropy_history,
            'variance': self.variance_history
        }
        
        return history
    
    def predict(self, inputs, threshold=0.5):
        """
        Get router prediction and binary mask.
        
        Parameters:
        -----------
        inputs : tf.Tensor or np.ndarray
            Router input of shape (1, H, W, 5)
        threshold : float
            Threshold for binary mask (default: 0.5)
            
        Returns:
        --------
        r : np.ndarray
            Continuous router output (H, W)
        mask : np.ndarray
            Binary mask (H, W), 1=CFD, 0=PINN
        """
        inputs = tf.constant(inputs, dtype=tf.float32)
        r = self.router(inputs, training=False)
        r = r[0, :, :, 0].numpy()
        
        mask = (r >= threshold).astype(np.int32)
        
        return r, mask


def create_router_input(layout, bc_mask, bc_values_u, bc_values_v, bc_values_p):
    """
    Create the 5-channel input tensor for the router.
    
    Parameters:
    -----------
    layout : np.ndarray
        Layout mask of shape (H, W), 0=obstacle, 1=fluid
    bc_mask : np.ndarray
        Boundary condition mask of shape (H, W)
    bc_values_u, bc_values_v, bc_values_p : np.ndarray
        Boundary condition values of shape (H, W)
        
    Returns:
    --------
    inputs : np.ndarray
        Stacked input of shape (1, H, W, 5)
    """
    inputs = np.stack([
        layout,
        bc_mask,
        bc_values_u,
        bc_values_v,
        bc_values_p
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
    
    # 1. Continuous router output
    ax = axes[0]
    r_masked = np.ma.masked_where(layout == 0, r)
    cf = ax.contourf(X, Y, r_masked, levels=50, cmap='RdBu_r', 
                     norm=Normalize(vmin=0, vmax=1))
    plt.colorbar(cf, ax=ax, label='Router Score')
    if show_circle:
        cx, cy, radius = show_circle
        circle = plt.Circle((cx, cy), radius, color='gray', fill=True)
        ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Continuous Router Output\n(0=PINN, 1=CFD)')
    
    # 2. Binary mask (threshold = 0.5)
    ax = axes[1]
    mask = (r >= 0.5).astype(np.float32)
    mask_masked = np.ma.masked_where(layout == 0, mask)
    cf = ax.contourf(X, Y, mask_masked, levels=[0, 0.5, 1], 
                     colors=['blue', 'red'], alpha=0.7)
    cbar = plt.colorbar(cf, ax=ax, ticks=[0.25, 0.75])
    cbar.ax.set_yticklabels(['PINN', 'CFD'])
    if show_circle:
        circle = plt.Circle((cx, cy), radius, color='gray', fill=True)
        ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Binary Mask (threshold=0.5)')
    
    # 3. Layout with regions
    ax = axes[2]
    combined = np.zeros_like(r)
    combined[layout == 0] = 0  # Obstacle
    combined[(layout == 1) & (r < 0.5)] = 1  # PINN region
    combined[(layout == 1) & (r >= 0.5)] = 2  # CFD region
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
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Total loss
    ax = axes[0, 0]
    ax.plot(history['total_loss'], 'b-', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss')
    ax.grid(True, alpha=0.3)
    
    # CFD cost
    ax = axes[0, 1]
    ax.plot(history['cfd_cost'], 'r-', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('CFD Cost')
    ax.set_title('CFD Cost (β · Σr)')
    ax.grid(True, alpha=0.3)
    
    # Residual loss
    ax = axes[1, 0]
    ax.plot(history['residual_loss'], 'g-', linewidth=1.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Residual Loss')
    ax.set_title('PINN Residual Loss')
    ax.grid(True, alpha=0.3)
    
    # TV loss
    ax = axes[1, 1]
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
