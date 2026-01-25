"""
Learned Router for Hybrid PINN-CFD Navier-Stokes Flow.

This module implements a physics-based router network that learns to decide
whether to accept PINN predictions or reject them (triggering CFD recomputation)
at each spatial location. The router is trained using physics-based losses 
(no ground truth CFD data required).

Key Features:
- CNN-based router for structured grids
- Physics-based loss function (Navier-Stokes residuals)
- Soft-hard decoupling strategy for differentiable training
- Spatial smoothness regularization with flow-aligned weighting
- No forced boundary/grid CFD regions - purely learned from physics

References:
    The soft-hard decoupling strategy uses soft probabilities for gradient flow
    while using hard thresholds for CFD execution.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import jax
import jax.numpy as jnp
from jax import jit, lax
from scipy import ndimage


class RouterNetwork(keras.Model):
    """
    CNN-based router network for structured grids.
    
    Input: Multi-channel feature map (Ny, Nx, num_features)
    Output: Soft probability map r(x,y) ∈ [0,1] where:
        - r ≈ 0: Accept PINN prediction
        - r ≈ 1: Reject PINN, use CFD instead
    """
    
    def __init__(self, num_input_features=12, num_filters=32, kernel_size=3, num_layers=4):
        """
        Initialize router network.
        
        Parameters:
        -----------
        num_input_features : int
            Number of input feature channels
        num_filters : int
            Number of filters in conv layers
        kernel_size : int
            Kernel size for conv layers
        num_layers : int
            Number of convolutional layers
        """
        super().__init__()
        
        self.num_input_features = num_input_features
        
        # Encoder: progressively extract features
        self.conv_layers = []
        self.bn_layers = []
        
        for i in range(num_layers):
            in_filters = num_input_features if i == 0 else num_filters
            self.conv_layers.append(
                layers.Conv2D(
                    filters=num_filters,
                    kernel_size=kernel_size,
                    padding='same',
                    activation=None,
                    kernel_initializer='he_normal'
                )
            )
            self.bn_layers.append(layers.BatchNormalization())
        
        # Output layer: single channel probability
        self.output_conv = layers.Conv2D(
            filters=1,
            kernel_size=1,
            padding='same',
            activation='sigmoid',
            kernel_initializer='glorot_normal'
        )
        
    def call(self, x, training=False):
        """
        Forward pass.
        
        Parameters:
        -----------
        x : Tensor of shape (batch, Ny, Nx, num_features)
            Input feature map
        training : bool
            Whether in training mode
            
        Returns:
        --------
        r_soft : Tensor of shape (batch, Ny, Nx, 1)
            Soft probabilities for CFD rejection
        """
        h = x
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            h = conv(h)
            h = bn(h, training=training)
            h = tf.nn.leaky_relu(h, alpha=0.1)
        
        # Output probability
        r_soft = self.output_conv(h)
        
        return r_soft


class RouterFeatureExtractor:
    """
    Extract input features for the router network from PINN predictions.
    
    Features include:
    - Distance to inlet: d_inlet(x)
    - Distance to cylinder: d_cyl(x)
    - Flow-aligned coordinates: (ξ, η)
    - Local PINN predictions: (u, v, p)
    - Velocity gradients: (∂u/∂x, ∂u/∂y, ∂v/∂x, ∂v/∂y)
    - Second derivatives: (∂²u/∂x², ∂²u/∂y², ∂²v/∂x², ∂²v/∂y²)
    - Boundary condition violation
    - Local complexity: strain rate, vorticity magnitude
    """
    
    def __init__(self, x_domain, y_domain, cylinder_center, cylinder_radius,
                 inlet_velocity=1.0, dx=None, dy=None):
        """
        Initialize feature extractor.
        
        Parameters:
        -----------
        x_domain : tuple
            (x_min, x_max) domain bounds
        y_domain : tuple
            (y_min, y_max) domain bounds
        cylinder_center : tuple
            (Cx, Cy) cylinder center
        cylinder_radius : float
            Cylinder radius
        inlet_velocity : float
            Inlet velocity for BC violation computation
        dx, dy : float
            Grid spacing
        """
        self.x_min, self.x_max = x_domain
        self.y_min, self.y_max = y_domain
        self.Cx, self.Cy = cylinder_center
        self.radius = cylinder_radius
        self.u_inlet = inlet_velocity
        self.dx = dx
        self.dy = dy
        
        self.Lx = self.x_max - self.x_min
        self.Ly = self.y_max - self.y_min
        
    def compute_distance_features(self, X, Y):
        """
        Compute distance-based features.
        
        Returns:
        --------
        d_inlet : ndarray
            Distance to inlet (normalized)
        d_cyl : ndarray
            Distance to cylinder surface (normalized)
        """
        # Distance to inlet (left boundary)
        d_inlet = (X - self.x_min) / self.Lx
        
        # Distance to cylinder surface
        dist_to_center = np.sqrt((X - self.Cx)**2 + (Y - self.Cy)**2)
        d_cyl = (dist_to_center - self.radius) / self.Lx
        d_cyl = np.maximum(d_cyl, 0)  # Inside cylinder has d_cyl = 0
        
        return d_inlet, d_cyl
    
    def compute_flow_aligned_coords(self, X, Y, u_mean=None):
        """
        Compute flow-aligned coordinates (ξ, η).
        
        ξ: streamwise direction (aligned with mean flow)
        η: transverse direction (perpendicular to mean flow)
        
        For cylinder flow, mean flow is approximately horizontal (x-direction).
        """
        if u_mean is None:
            # Assume horizontal mean flow
            u_mean_x, u_mean_y = 1.0, 0.0
        else:
            u_mean_x = np.mean(u_mean)
            u_mean_y = 0.0  # For cylinder flow, v_mean ≈ 0
        
        # Normalize mean flow direction
        norm = np.sqrt(u_mean_x**2 + u_mean_y**2) + 1e-10
        ex = u_mean_x / norm
        ey = u_mean_y / norm
        
        # Flow-aligned coordinates (shift origin to cylinder)
        x_rel = X - self.Cx
        y_rel = Y - self.Cy
        
        # ξ = projection onto mean flow direction
        xi = (x_rel * ex + y_rel * ey) / self.Lx
        
        # η = projection onto transverse direction
        eta = (-x_rel * ey + y_rel * ex) / self.Ly
        
        return xi, eta
    
    def compute_velocity_gradients(self, u, v, dx, dy):
        """
        Compute first and second order velocity gradients.
        
        Returns dict with:
        - du_dx, du_dy, dv_dx, dv_dy (first derivatives)
        - d2u_dx2, d2u_dy2, d2v_dx2, d2v_dy2 (second derivatives)
        """
        grads = {}
        
        # First derivatives (central differences)
        grads['du_dx'] = np.zeros_like(u)
        grads['du_dy'] = np.zeros_like(u)
        grads['dv_dx'] = np.zeros_like(v)
        grads['dv_dy'] = np.zeros_like(v)
        
        grads['du_dx'][1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
        grads['du_dy'][1:-1, 1:-1] = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)
        grads['dv_dx'][1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)
        grads['dv_dy'][1:-1, 1:-1] = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
        
        # Boundary gradients (one-sided differences)
        grads['du_dx'][:, 0] = (u[:, 1] - u[:, 0]) / dx
        grads['du_dx'][:, -1] = (u[:, -1] - u[:, -2]) / dx
        grads['dv_dx'][:, 0] = (v[:, 1] - v[:, 0]) / dx
        grads['dv_dx'][:, -1] = (v[:, -1] - v[:, -2]) / dx
        
        grads['du_dy'][0, :] = (u[1, :] - u[0, :]) / dy
        grads['du_dy'][-1, :] = (u[-1, :] - u[-2, :]) / dy
        grads['dv_dy'][0, :] = (v[1, :] - v[0, :]) / dy
        grads['dv_dy'][-1, :] = (v[-1, :] - v[-2, :]) / dy
        
        # Second derivatives (central differences)
        grads['d2u_dx2'] = np.zeros_like(u)
        grads['d2u_dy2'] = np.zeros_like(u)
        grads['d2v_dx2'] = np.zeros_like(v)
        grads['d2v_dy2'] = np.zeros_like(v)
        
        grads['d2u_dx2'][1:-1, 1:-1] = (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dx**2
        grads['d2u_dy2'][1:-1, 1:-1] = (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy**2
        grads['d2v_dx2'][1:-1, 1:-1] = (v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2]) / dx**2
        grads['d2v_dy2'][1:-1, 1:-1] = (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1]) / dy**2
        
        return grads
    
    def compute_bc_violation(self, u, v, X, Y):
        """
        Compute boundary condition violation at boundaries.
        
        Returns:
        --------
        bc_violation : ndarray
            ||u_PINN - u_BC||² at boundary points, 0 in interior
        """
        Ny, Nx = u.shape
        bc_violation = np.zeros_like(u)
        
        # Inlet BC violation: parabolic profile
        y_inlet = Y[:, 0]
        u_inlet_bc = 4 * self.u_inlet * (y_inlet - self.y_min) * (self.y_max - y_inlet) / (self.Ly**2)
        bc_violation[:, 0] = (u[:, 0] - u_inlet_bc)**2 + v[:, 0]**2
        
        # Wall BC violation: no-slip
        bc_violation[0, :] = u[0, :]**2 + v[0, :]**2  # Bottom
        bc_violation[-1, :] = u[-1, :]**2 + v[-1, :]**2  # Top
        
        # Cylinder BC: handled separately (mask)
        
        return bc_violation
    
    def compute_complexity_metrics(self, u, v, dx, dy):
        """
        Compute local complexity metrics: strain rate and vorticity.
        """
        # Strain rate magnitude
        du_dx = np.zeros_like(u)
        du_dy = np.zeros_like(u)
        dv_dx = np.zeros_like(v)
        dv_dy = np.zeros_like(v)
        
        du_dx[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
        du_dy[1:-1, 1:-1] = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)
        dv_dx[1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)
        dv_dy[1:-1, 1:-1] = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
        
        S_xx = du_dx
        S_yy = dv_dy
        S_xy = 0.5 * (du_dy + dv_dx)
        
        strain_rate = np.sqrt(2.0 * (S_xx**2 + 2*S_xy**2 + S_yy**2))
        
        # Vorticity magnitude
        vorticity = np.abs(dv_dx - du_dy)
        
        return strain_rate, vorticity
    
    def extract_features(self, X, Y, u, v, p, cylinder_mask=None):
        """
        Extract all features for the router network.
        
        Parameters:
        -----------
        X, Y : ndarray of shape (Ny, Nx)
            Coordinate grids
        u, v, p : ndarray of shape (Ny, Nx)
            PINN velocity and pressure predictions
        cylinder_mask : ndarray of shape (Ny, Nx), optional
            Mask indicating cylinder interior (1 = inside)
            
        Returns:
        --------
        features : ndarray of shape (1, Ny, Nx, num_features)
            Feature tensor ready for router network
        """
        dx = self.dx if self.dx else (X[0, 1] - X[0, 0])
        dy = self.dy if self.dy else (Y[1, 0] - Y[0, 0])
        
        Ny, Nx = X.shape
        
        # Distance features (2 channels)
        d_inlet, d_cyl = self.compute_distance_features(X, Y)
        
        # Flow-aligned coordinates (2 channels)
        xi, eta = self.compute_flow_aligned_coords(X, Y, u)
        
        # Normalize PINN predictions (3 channels)
        u_max = np.abs(u).max() + 1e-10
        v_max = np.abs(v).max() + 1e-10
        p_max = np.abs(p).max() + 1e-10
        
        u_norm = u / u_max
        v_norm = v / v_max
        p_norm = p / p_max
        
        # Velocity gradients (4 first-order + 4 second-order = 8 channels, reduced to 4)
        grads = self.compute_velocity_gradients(u, v, dx, dy)
        
        # Combine gradients into key features
        grad_u_mag = np.sqrt(grads['du_dx']**2 + grads['du_dy']**2)
        grad_v_mag = np.sqrt(grads['dv_dx']**2 + grads['dv_dy']**2)
        laplacian_u = grads['d2u_dx2'] + grads['d2u_dy2']
        laplacian_v = grads['d2v_dx2'] + grads['d2v_dy2']
        
        # Normalize gradients
        grad_scale = max(grad_u_mag.max(), grad_v_mag.max()) + 1e-10
        lap_scale = max(np.abs(laplacian_u).max(), np.abs(laplacian_v).max()) + 1e-10
        
        grad_u_norm = grad_u_mag / grad_scale
        grad_v_norm = grad_v_mag / grad_scale
        lap_u_norm = laplacian_u / lap_scale
        lap_v_norm = laplacian_v / lap_scale
        
        # BC violation (1 channel)
        bc_violation = self.compute_bc_violation(u, v, X, Y)
        bc_scale = bc_violation.max() + 1e-10
        bc_norm = bc_violation / bc_scale
        
        # Complexity metrics (2 channels)
        strain_rate, vorticity = self.compute_complexity_metrics(u, v, dx, dy)
        strain_scale = strain_rate.max() + 1e-10
        vort_scale = vorticity.max() + 1e-10
        strain_norm = strain_rate / strain_scale
        vort_norm = vorticity / vort_scale
        
        # Stack all features: total 14 channels
        features = np.stack([
            d_inlet,        # 0: distance to inlet
            d_cyl,          # 1: distance to cylinder
            xi,             # 2: streamwise coordinate
            eta,            # 3: transverse coordinate
            u_norm,         # 4: normalized u velocity
            v_norm,         # 5: normalized v velocity
            p_norm,         # 6: normalized pressure
            grad_u_norm,    # 7: |∇u|
            grad_v_norm,    # 8: |∇v|
            lap_u_norm,     # 9: ∇²u (normalized)
            lap_v_norm,     # 10: ∇²v (normalized)
            bc_norm,        # 11: BC violation
            strain_norm,    # 12: strain rate
            vort_norm       # 13: vorticity
        ], axis=-1)
        
        # Add batch dimension
        features = features[np.newaxis, ...]
        
        return features.astype(np.float32)


class PhysicsLoss:
    """
    Physics-based loss functions for router training.
    
    Computes:
    - L_physics: Navier-Stokes residual in hybrid solution
    - L_BC: Boundary condition violation
    - L_spatial: Spatial smoothness with flow-aligned weighting
    - L_cost: Computational cost (fraction using CFD)
    """
    
    def __init__(self, nu, rho=1.0, dx=None, dy=None,
                 lambda_BC=10.0, lambda_spatial=0.1, lambda_cost=0.01,
                 x_domain=None, y_domain=None, inlet_velocity=1.0):
        """
        Initialize physics loss.
        
        Parameters:
        -----------
        nu : float
            Kinematic viscosity
        rho : float
            Fluid density
        dx, dy : float
            Grid spacing
        lambda_BC : float
            Weight for BC loss
        lambda_spatial : float
            Weight for spatial smoothness loss
        lambda_cost : float
            Weight for cost penalty
        """
        self.nu = nu
        self.rho = rho
        self.dx = dx
        self.dy = dy
        
        self.lambda_BC = lambda_BC
        self.lambda_spatial = lambda_spatial
        self.lambda_cost = lambda_cost
        
        self.x_min, self.x_max = x_domain if x_domain else (0, 2)
        self.y_min, self.y_max = y_domain if y_domain else (0, 1)
        self.u_inlet = inlet_velocity
        self.Ly = self.y_max - self.y_min
        
    @tf.function
    def navier_stokes_residual(self, u, v, p, x, y):
        """
        Compute Navier-Stokes residual using automatic differentiation.
        
        N[u,v,p] = [
            ρ(u∂u/∂x + v∂u/∂y) + ∂p/∂x - μ(∂²u/∂x² + ∂²u/∂y²),  # x-momentum
            ρ(u∂v/∂x + v∂v/∂y) + ∂p/∂y - μ(∂²v/∂x² + ∂²v/∂y²),  # y-momentum
            ∂u/∂x + ∂v/∂y                                          # continuity
        ]
        
        Parameters:
        -----------
        u, v, p : Tensor of shape (batch, Ny, Nx, 1)
            Hybrid velocity and pressure fields
        x, y : Tensor of shape (batch, Ny, Nx, 1)
            Coordinate grids
            
        Returns:
        --------
        residual : Tensor (scalar)
            Mean squared NS residual
        """
        mu = self.nu * self.rho
        
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, y])
            
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x, y])
                
                # First derivatives
                u_x = tape1.gradient(u, x)
                u_y = tape1.gradient(u, y)
                v_x = tape1.gradient(v, x)
                v_y = tape1.gradient(v, y)
                p_x = tape1.gradient(p, x)
                p_y = tape1.gradient(p, y)
            
            # Second derivatives
            u_xx = tape2.gradient(u_x, x)
            u_yy = tape2.gradient(u_y, y)
            v_xx = tape2.gradient(v_x, x)
            v_yy = tape2.gradient(v_y, y)
        
        del tape1, tape2
        
        # Handle None gradients
        if u_xx is None: u_xx = tf.zeros_like(u)
        if u_yy is None: u_yy = tf.zeros_like(u)
        if v_xx is None: v_xx = tf.zeros_like(v)
        if v_yy is None: v_yy = tf.zeros_like(v)
        
        # Momentum equations
        R_u = self.rho * (u * u_x + v * u_y) + p_x - mu * (u_xx + u_yy)
        R_v = self.rho * (u * v_x + v * v_y) + p_y - mu * (v_xx + v_yy)
        
        # Continuity
        R_c = u_x + v_y
        
        # Mean squared residual
        residual = tf.reduce_mean(R_u**2 + R_v**2 + R_c**2)
        
        return residual
    
    def compute_ns_residual_fd(self, u, v, p, dx, dy):
        """
        Compute Navier-Stokes residual using finite differences.
        
        More stable for grid-based data than automatic differentiation.
        """
        mu = self.nu * self.rho
        
        # First derivatives (central differences)
        u_x = np.zeros_like(u)
        u_y = np.zeros_like(u)
        v_x = np.zeros_like(v)
        v_y = np.zeros_like(v)
        p_x = np.zeros_like(p)
        p_y = np.zeros_like(p)
        
        u_x[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
        u_y[1:-1, 1:-1] = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)
        v_x[1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)
        v_y[1:-1, 1:-1] = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
        p_x[1:-1, 1:-1] = (p[1:-1, 2:] - p[1:-1, :-2]) / (2 * dx)
        p_y[1:-1, 1:-1] = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2 * dy)
        
        # Second derivatives
        u_xx = np.zeros_like(u)
        u_yy = np.zeros_like(u)
        v_xx = np.zeros_like(v)
        v_yy = np.zeros_like(v)
        
        u_xx[1:-1, 1:-1] = (u[1:-1, 2:] - 2*u[1:-1, 1:-1] + u[1:-1, :-2]) / dx**2
        u_yy[1:-1, 1:-1] = (u[2:, 1:-1] - 2*u[1:-1, 1:-1] + u[:-2, 1:-1]) / dy**2
        v_xx[1:-1, 1:-1] = (v[1:-1, 2:] - 2*v[1:-1, 1:-1] + v[1:-1, :-2]) / dx**2
        v_yy[1:-1, 1:-1] = (v[2:, 1:-1] - 2*v[1:-1, 1:-1] + v[:-2, 1:-1]) / dy**2
        
        # Momentum residuals
        R_u = self.rho * (u * u_x + v * u_y) + p_x - mu * (u_xx + u_yy)
        R_v = self.rho * (u * v_x + v * v_y) + p_y - mu * (v_xx + v_yy)
        
        # Continuity residual
        R_c = u_x + v_y
        
        # Mean squared residual (interior points only)
        residual = np.mean(R_u[1:-1, 1:-1]**2 + R_v[1:-1, 1:-1]**2 + R_c[1:-1, 1:-1]**2)
        
        return residual, (R_u, R_v, R_c)
    
    def boundary_loss(self, u, v, Y):
        """
        Compute boundary condition violation.
        
        - Inlet: parabolic profile
        - Walls: no-slip
        """
        Ny = u.shape[0]
        
        # Inlet BC
        y_inlet = Y[:, 0]
        u_inlet_bc = 4 * self.u_inlet * (y_inlet - self.y_min) * (self.y_max - y_inlet) / (self.Ly**2)
        inlet_loss = np.mean((u[:, 0] - u_inlet_bc)**2 + v[:, 0]**2)
        
        # Wall BCs (top and bottom)
        wall_loss = np.mean(u[0, :]**2 + v[0, :]**2)  # Bottom
        wall_loss += np.mean(u[-1, :]**2 + v[-1, :]**2)  # Top
        
        return inlet_loss + wall_loss
    
    def spatial_smoothness_loss(self, r_soft, X, Y, u_mean=None, alpha=2.0, h=None):
        """
        Compute spatial smoothness loss with flow-aligned weighting.
        
        L_spatial = Σ_{(i,j) ∈ edges} w_ij · |r(x_i) - r(x_j)|
        
        w_ij = exp(-||x_i - x_j||²/h²) · (1 + α · flow_alignment(i,j))
        
        Parameters:
        -----------
        r_soft : ndarray of shape (Ny, Nx)
            Soft probability mask
        X, Y : ndarray of shape (Ny, Nx)
            Coordinate grids
        u_mean : ndarray, optional
            Mean flow field for flow alignment
        alpha : float
            Flow alignment weight
        h : float
            Spatial scale (default: grid spacing)
        """
        if h is None:
            h = np.sqrt((X[0, 1] - X[0, 0])**2 + (Y[1, 0] - Y[0, 0])**2)
        
        Ny, Nx = r_soft.shape
        
        # Compute total variation in x and y directions
        # Horizontal edges (x-direction, along flow)
        r_diff_x = np.abs(r_soft[:, 1:] - r_soft[:, :-1])
        
        # Vertical edges (y-direction, perpendicular to flow)
        r_diff_y = np.abs(r_soft[1:, :] - r_soft[:-1, :])
        
        # For flow around cylinder, mean flow is predominantly in x-direction
        # Stronger regularization along flow (x) to prevent fragmented masks
        w_x = 1.0 + alpha  # Flow-aligned weighting
        w_y = 1.0           # Transverse weighting
        
        spatial_loss = w_x * np.mean(r_diff_x) + w_y * np.mean(r_diff_y)
        
        return spatial_loss
    
    def cost_penalty(self, r_soft):
        """
        Compute computational cost penalty.
        
        L_cost = mean(r_soft) = fraction of domain using CFD
        """
        return np.mean(r_soft)
    
    def total_loss(self, u_hybrid, v_hybrid, p_hybrid, r_soft, X, Y, dx, dy,
                   cylinder_mask=None):
        """
        Compute total physics-based loss.
        
        L_total = L_physics + λ_BC · L_BC + λ_spatial · L_spatial + λ_cost · L_cost
        """
        # Physics residual
        L_physics, residuals = self.compute_ns_residual_fd(u_hybrid, v_hybrid, p_hybrid, dx, dy)
        
        # Boundary loss
        L_BC = self.boundary_loss(u_hybrid, v_hybrid, Y)
        
        # Spatial smoothness
        L_spatial = self.spatial_smoothness_loss(r_soft, X, Y)
        
        # Cost penalty
        L_cost = self.cost_penalty(r_soft)
        
        # Total loss
        total = L_physics + self.lambda_BC * L_BC + self.lambda_spatial * L_spatial + self.lambda_cost * L_cost
        
        losses = {
            'total': total,
            'physics': L_physics,
            'bc': L_BC,
            'spatial': L_spatial,
            'cost': L_cost,
            'cfd_fraction': np.mean(r_soft > 0.5)
        }
        
        return total, losses


class HybridRouterTrainer:
    """
    Trainer for the hybrid PINN-CFD router.
    
    Implements the soft-hard decoupling strategy:
    - Hard thresholding for CFD execution (no gradients)
    - Soft blending for hybrid solution (differentiable)
    """
    
    def __init__(self, router, pinn_network, cfd_solver,
                 feature_extractor, physics_loss,
                 learning_rate=1e-3, threshold=0.5):
        """
        Initialize trainer.
        
        Parameters:
        -----------
        router : RouterNetwork
            Router network to train
        pinn_network : tf.keras.Model
            Pre-trained PINN network (frozen)
        cfd_solver : callable
            CFD solver function
        feature_extractor : RouterFeatureExtractor
            Feature extraction for router input
        physics_loss : PhysicsLoss
            Physics-based loss functions
        learning_rate : float
            Learning rate for optimizer
        threshold : float
            Threshold for binary CFD/PINN decision
        """
        self.router = router
        self.pinn = pinn_network
        self.cfd_solver = cfd_solver
        self.feature_extractor = feature_extractor
        self.physics_loss = physics_loss
        self.threshold = threshold
        
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.history = {
            'total_loss': [],
            'physics_loss': [],
            'bc_loss': [],
            'spatial_loss': [],
            'cost_loss': [],
            'cfd_fraction': [],
            'inlet_rejection_rate': []
        }
    
    def get_pinn_prediction(self, xy):
        """Get PINN prediction (frozen, no gradients)."""
        uvp = self.pinn.predict(xy, batch_size=len(xy), verbose=0)
        return uvp[..., 0], uvp[..., 1], uvp[..., 2]
    
    def compute_hybrid_solution(self, u_pinn, v_pinn, p_pinn,
                                 u_cfd, v_cfd, p_cfd, r_soft):
        """
        Compute hybrid solution using SOFT weights.
        
        u_hybrid = (1 - r_soft) * u_PINN + r_soft * u_CFD
        
        This maintains differentiability for backpropagation.
        """
        u_hybrid = (1 - r_soft) * u_pinn + r_soft * u_cfd
        v_hybrid = (1 - r_soft) * v_pinn + r_soft * v_cfd
        p_hybrid = (1 - r_soft) * p_pinn + r_soft * p_cfd
        
        return u_hybrid, v_hybrid, p_hybrid
    
    @tf.function
    def train_step(self, features, u_pinn, v_pinn, p_pinn,
                   u_cfd, v_cfd, p_cfd, X, Y, dx, dy):
        """
        Single training step with gradient computation.
        
        The key insight: use r_soft for blending (differentiable) 
        but r_binary for CFD region selection (non-differentiable).
        """
        with tf.GradientTape() as tape:
            # Router forward pass
            r_soft = self.router(features, training=True)
            r_soft = tf.squeeze(r_soft, axis=[0, -1])  # (Ny, Nx)
            
            # Compute hybrid solution using SOFT weights
            u_hybrid = (1.0 - r_soft) * u_pinn + r_soft * u_cfd
            v_hybrid = (1.0 - r_soft) * v_pinn + r_soft * v_cfd
            p_hybrid = (1.0 - r_soft) * p_pinn + r_soft * p_cfd
            
            # Compute losses
            # Note: Using TensorFlow operations for gradient flow
            
            # Physics residual (simplified for TF graph)
            # Using finite difference approximation
            mu = self.physics_loss.nu * self.physics_loss.rho
            
            # Central differences for derivatives
            u_x = (u_hybrid[:, 2:] - u_hybrid[:, :-2]) / (2 * dx)
            u_y = (u_hybrid[2:, :] - u_hybrid[:-2, :]) / (2 * dy)
            v_x = (v_hybrid[:, 2:] - v_hybrid[:, :-2]) / (2 * dx)
            v_y = (v_hybrid[2:, :] - v_hybrid[:-2, :]) / (2 * dy)
            p_x = (p_hybrid[:, 2:] - p_hybrid[:, :-2]) / (2 * dx)
            p_y = (p_hybrid[2:, :] - p_hybrid[:-2, :]) / (2 * dy)
            
            u_xx = (u_hybrid[:, 2:] - 2*u_hybrid[:, 1:-1] + u_hybrid[:, :-2]) / (dx**2)
            u_yy = (u_hybrid[2:, :] - 2*u_hybrid[1:-1, :] + u_hybrid[:-2, :]) / (dy**2)
            v_xx = (v_hybrid[:, 2:] - 2*v_hybrid[:, 1:-1] + v_hybrid[:, :-2]) / (dx**2)
            v_yy = (v_hybrid[2:, :] - 2*v_hybrid[1:-1, :] + v_hybrid[:-2, :]) / (dy**2)
            
            # Interior slice indices
            u_int = u_hybrid[1:-1, 1:-1]
            v_int = v_hybrid[1:-1, 1:-1]
            
            u_x_int = u_x[1:-1, :]
            u_y_int = u_y[:, 1:-1]
            v_x_int = v_x[1:-1, :]
            v_y_int = v_y[:, 1:-1]
            p_x_int = p_x[1:-1, :]
            p_y_int = p_y[:, 1:-1]
            
            u_xx_int = u_xx[1:-1, :]
            u_yy_int = u_yy[:, 1:-1]
            v_xx_int = v_xx[1:-1, :]
            v_yy_int = v_yy[:, 1:-1]
            
            # Momentum residuals
            R_u = self.physics_loss.rho * (u_int * u_x_int + v_int * u_y_int) + p_x_int - mu * (u_xx_int + u_yy_int)
            R_v = self.physics_loss.rho * (u_int * v_x_int + v_int * v_y_int) + p_y_int - mu * (v_xx_int + v_yy_int)
            R_c = u_x_int + v_y_int
            
            L_physics = tf.reduce_mean(R_u**2 + R_v**2 + R_c**2)
            
            # Boundary loss
            Ny = tf.shape(u_hybrid)[0]
            y_vals = tf.linspace(self.physics_loss.y_min, self.physics_loss.y_max, Ny)
            u_inlet_bc = 4 * self.physics_loss.u_inlet * (y_vals - self.physics_loss.y_min) * \
                         (self.physics_loss.y_max - y_vals) / (self.physics_loss.Ly**2)
            
            inlet_loss = tf.reduce_mean((u_hybrid[:, 0] - u_inlet_bc)**2 + v_hybrid[:, 0]**2)
            wall_loss = tf.reduce_mean(u_hybrid[0, :]**2 + v_hybrid[0, :]**2)
            wall_loss += tf.reduce_mean(u_hybrid[-1, :]**2 + v_hybrid[-1, :]**2)
            L_BC = inlet_loss + wall_loss
            
            # Spatial smoothness (total variation)
            r_diff_x = tf.abs(r_soft[:, 1:] - r_soft[:, :-1])
            r_diff_y = tf.abs(r_soft[1:, :] - r_soft[:-1, :])
            L_spatial = 3.0 * tf.reduce_mean(r_diff_x) + tf.reduce_mean(r_diff_y)
            
            # Cost penalty
            L_cost = tf.reduce_mean(r_soft)
            
            # Total loss
            total_loss = (L_physics + 
                         self.physics_loss.lambda_BC * L_BC + 
                         self.physics_loss.lambda_spatial * L_spatial + 
                         self.physics_loss.lambda_cost * L_cost)
        
        # Compute gradients
        gradients = tape.gradient(total_loss, self.router.trainable_variables)
        
        # Clip gradients
        gradients = [tf.clip_by_norm(g, 1.0) if g is not None else g for g in gradients]
        
        # Apply gradients
        self.optimizer.apply_gradients(zip(gradients, self.router.trainable_variables))
        
        return {
            'total': total_loss,
            'physics': L_physics,
            'bc': L_BC,
            'spatial': L_spatial,
            'cost': L_cost
        }
    
    def train(self, X, Y, xy, num_epochs=100, verbose=True, cylinder_mask=None):
        """
        Train the router network.
        
        Parameters:
        -----------
        X, Y : ndarray of shape (Ny, Nx)
            Coordinate grids
        xy : ndarray of shape (N, 2)
            Flattened coordinates for PINN queries
        num_epochs : int
            Number of training epochs
        verbose : bool
            Print progress
        cylinder_mask : ndarray, optional
            Mask for cylinder interior
            
        Returns:
        --------
        history : dict
            Training history
        """
        Ny, Nx = X.shape
        dx = X[0, 1] - X[0, 0]
        dy = Y[1, 0] - Y[0, 0]
        
        # Get PINN prediction (fixed throughout training)
        print("Getting PINN predictions...")
        u_pinn, v_pinn, p_pinn = self.get_pinn_prediction(xy)
        u_pinn = u_pinn.reshape(Ny, Nx)
        v_pinn = v_pinn.reshape(Ny, Nx)
        p_pinn = p_pinn.reshape(Ny, Nx)
        
        # Apply cylinder mask to PINN
        if cylinder_mask is not None:
            u_pinn = np.where(cylinder_mask == 1, 0.0, u_pinn)
            v_pinn = np.where(cylinder_mask == 1, 0.0, v_pinn)
        
        # Convert to TensorFlow tensors
        u_pinn_tf = tf.constant(u_pinn, dtype=tf.float32)
        v_pinn_tf = tf.constant(v_pinn, dtype=tf.float32)
        p_pinn_tf = tf.constant(p_pinn, dtype=tf.float32)
        X_tf = tf.constant(X, dtype=tf.float32)
        Y_tf = tf.constant(Y, dtype=tf.float32)
        
        print(f"\nStarting router training for {num_epochs} epochs...")
        print(f"Grid size: {Ny}x{Nx}")
        print("-" * 60)
        
        for epoch in range(num_epochs):
            # Extract features from current PINN prediction
            features = self.feature_extractor.extract_features(
                X, Y, u_pinn, v_pinn, p_pinn, cylinder_mask
            )
            features_tf = tf.constant(features, dtype=tf.float32)
            
            # Get current router prediction (for CFD region determination)
            with tf.GradientTape() as tape:
                r_soft_pred = self.router(features_tf, training=False)
            r_soft_np = r_soft_pred.numpy().squeeze()
            
            # Determine CFD region using HARD threshold (detached from gradients)
            r_binary = (r_soft_np > self.threshold).astype(np.float32)
            cfd_mask = r_binary == 1
            
            # Run CFD solver on CFD region
            if cfd_mask.sum() > 0:
                u_cfd, v_cfd, p_cfd = self.cfd_solver(
                    cfd_mask=cfd_mask,
                    u_pinn=u_pinn,
                    v_pinn=v_pinn,
                    p_pinn=p_pinn,
                    cylinder_mask=cylinder_mask
                )
            else:
                # No CFD region - use PINN everywhere
                u_cfd = u_pinn.copy()
                v_cfd = v_pinn.copy()
                p_cfd = p_pinn.copy()
            
            # Convert CFD solution to TensorFlow
            u_cfd_tf = tf.constant(u_cfd, dtype=tf.float32)
            v_cfd_tf = tf.constant(v_cfd, dtype=tf.float32)
            p_cfd_tf = tf.constant(p_cfd, dtype=tf.float32)
            
            # Training step
            losses = self.train_step(
                features_tf, u_pinn_tf, v_pinn_tf, p_pinn_tf,
                u_cfd_tf, v_cfd_tf, p_cfd_tf,
                X_tf, Y_tf, dx, dy
            )
            
            # Record history
            self.history['total_loss'].append(float(losses['total']))
            self.history['physics_loss'].append(float(losses['physics']))
            self.history['bc_loss'].append(float(losses['bc']))
            self.history['spatial_loss'].append(float(losses['spatial']))
            self.history['cost_loss'].append(float(losses['cost']))
            self.history['cfd_fraction'].append(float(np.mean(r_binary)))
            
            # Inlet rejection rate (first 10% of x-domain should have high r)
            inlet_cols = int(Nx * 0.1)
            inlet_rejection = np.mean(r_soft_np[:, :inlet_cols])
            self.history['inlet_rejection_rate'].append(float(inlet_rejection))
            
            if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
                print(f"Epoch {epoch:4d} | "
                      f"Loss: {losses['total']:.4e} | "
                      f"Physics: {losses['physics']:.4e} | "
                      f"BC: {losses['bc']:.4e} | "
                      f"CFD%: {100*np.mean(r_binary):.1f}% | "
                      f"Inlet reject: {100*inlet_rejection:.1f}%")
        
        print("-" * 60)
        print("Training complete!")
        
        return self.history
    
    def get_final_mask(self, X, Y, u_pinn, v_pinn, p_pinn, cylinder_mask=None):
        """
        Get the final binary mask from the trained router.
        """
        features = self.feature_extractor.extract_features(
            X, Y, u_pinn, v_pinn, p_pinn, cylinder_mask
        )
        
        r_soft = self.router(features, training=False)
        r_soft = r_soft.numpy().squeeze()
        
        r_binary = (r_soft > self.threshold).astype(np.int32)
        
        return r_binary, r_soft


def create_router_cfd_solver(simulation_class, Re, N, max_iter=50000, tol=1e-5,
                             x_domain=(0, 2), y_domain=(0, 1),
                             cylinder_center=(0.5, 0.5), cylinder_radius=0.1,
                             inlet_velocity=1.0):
    """
    Create a CFD solver function for the router trainer.
    
    This wraps the hybrid CFD solver to solve only in the specified region
    using PINN values as boundary conditions at the interface.
    """
    from lib.cylinder_flow import CylinderFlowHybridSimulation
    
    def cfd_solver(cfd_mask, u_pinn, v_pinn, p_pinn, cylinder_mask=None):
        """
        Solve CFD in specified region using PINN as boundary conditions.
        """
        # If no CFD region, return PINN
        if cfd_mask.sum() == 0:
            return u_pinn, v_pinn, p_pinn
        
        # Create a placeholder network that returns the PINN values
        class PINNProxy:
            def __init__(self, u, v, p, shape):
                self.u = u.flatten()
                self.v = v.flatten()
                self.p = p.flatten()
                self.shape = shape
            
            def predict(self, xy, batch_size=None):
                return np.stack([self.u, self.v, self.p], axis=-1)
        
        proxy = PINNProxy(u_pinn, v_pinn, p_pinn, u_pinn.shape)
        
        # Create hybrid simulation with the mask
        sim = CylinderFlowHybridSimulation(
            network=proxy,
            uv_func=lambda net, xy: (net.u, net.v),
            mask=cfd_mask.astype(np.int32),
            Re=Re,
            N=N,
            max_iter=max_iter,
            tol=tol,
            x_domain=x_domain,
            y_domain=y_domain,
            cylinder_center=cylinder_center,
            cylinder_radius=cylinder_radius,
            inlet_velocity=inlet_velocity
        )
        
        # Override PINN values to use actual PINN predictions
        sim.u_pinn = u_pinn
        sim.v_pinn = v_pinn
        sim.p_pinn = p_pinn
        sim.u_pinn_jax = jnp.array(u_pinn)
        sim.v_pinn_jax = jnp.array(v_pinn)
        sim.p_pinn_jax = jnp.array(p_pinn)
        
        # Solve
        u, v, p = sim.solve()
        
        return u, v, p
    
    return cfd_solver
