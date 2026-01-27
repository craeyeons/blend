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
    """
    
    def __init__(self, base_filters=32, name="router_cnn"):
        """
        Initialize the router CNN.
        
        Parameters:
        -----------
        base_filters : int
            Number of filters in the first convolutional layer.
            Subsequent layers double this number.
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
        
        # Output layer (sigmoid for [0, 1] range)
        self.output_conv = layers.Conv2D(1, 1, padding='same', activation='sigmoid')
    
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
        
        # Output
        output = self.output_conv(d1)
        
        # Mask output: 0 at obstacle locations
        output = output * layout
        
        return output
    
    def _match_size(self, x, target):
        """Resize x to match target spatial dimensions."""
        target_shape = tf.shape(target)
        x_shape = tf.shape(x)
        
        # Crop or pad to match
        if x_shape[1] != target_shape[1] or x_shape[2] != target_shape[2]:
            x = tf.image.resize(x, [target_shape[1], target_shape[2]], method='bilinear')
        return x


class PINNResidualComputer:
    """
    Computes PINN physics residuals for the Navier-Stokes equations.
    
    Used to evaluate how well the PINN solution satisfies the governing equations
    at each spatial point.
    """
    
    def __init__(self, pinn_model, nu=0.01, rho=1.0):
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
        """
        self.pinn_model = pinn_model
        self.nu = nu
        self.rho = rho
    
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


class RouterTrainer:
    """
    Training manager for the CNN router.
    
    Implements the training loop with the loss function:
    L = β · Σ r(x_i) + Σ (1 - r(x_i)) · L_residual + λ · TV(r)
    """
    
    def __init__(self, router, pinn_model, 
                 beta=0.1, lambda_tv=0.01,
                 residual_weights={'continuity': 1.0, 'momentum': 1.0},
                 nu=0.01, rho=1.0):
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
            Weight for total variation regularization
        residual_weights : dict
            Weights for different residual components
        nu : float
            Kinematic viscosity
        rho : float
            Fluid density
        """
        self.router = router
        self.pinn_model = pinn_model
        self.beta = beta
        self.lambda_tv = lambda_tv
        self.residual_weights = residual_weights
        
        # Initialize residual computer
        self.residual_computer = PINNResidualComputer(pinn_model, nu, rho)
        
        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-4)
        
        # Metrics
        self.loss_history = []
        self.cfd_cost_history = []
        self.residual_loss_history = []
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
            Router input of shape (batch, H, W, 5)
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
        with tf.GradientTape() as tape:
            # Forward pass
            r = self.router(inputs, training=True)  # Shape: (batch, H, W, 1)
            r = r[0, :, :, 0]  # Remove batch and channel dims: (H, W)
            
            # Apply layout mask
            r_masked = r * tf.cast(layout_mask, tf.float32)
            
            # 1. CFD cost: β · Σ r(x_i)
            cfd_cost = self.beta * tf.reduce_sum(r_masked)
            
            # 2. PINN residual weighted by (1 - r)
            # Compute PINN residuals at all points
            total_residual = self.residual_computer.compute_total_residual(
                X, Y, self.residual_weights
            )
            # Weight by (1 - r): points assigned to PINN should have low residual
            pinn_weight = (1.0 - r_masked) * tf.cast(layout_mask, tf.float32)
            residual_loss = tf.reduce_sum(pinn_weight * total_residual)
            
            # 3. Total variation regularization
            r_4d = tf.reshape(r_masked, [1, tf.shape(r_masked)[0], tf.shape(r_masked)[1], 1])
            tv_loss = self.lambda_tv * self.compute_total_variation(r_4d)
            
            # Total loss
            total_loss = cfd_cost + residual_loss + tv_loss
        
        # Compute gradients and update
        gradients = tape.gradient(total_loss, self.router.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.router.trainable_variables))
        
        metrics = {
            'total_loss': total_loss,
            'cfd_cost': cfd_cost,
            'residual_loss': residual_loss,
            'tv_loss': tv_loss,
            'cfd_fraction': tf.reduce_mean(r_masked)
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
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {metrics['total_loss']:.4f}, "
                      f"CFD: {metrics['cfd_cost']:.4f}, "
                      f"Residual: {metrics['residual_loss']:.4f}, "
                      f"TV: {metrics['tv_loss']:.4f}, "
                      f"CFD%: {metrics['cfd_fraction']*100:.1f}%")
        
        history = {
            'total_loss': self.loss_history,
            'cfd_cost': self.cfd_cost_history,
            'residual_loss': self.residual_loss_history,
            'tv_loss': self.tv_loss_history
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
