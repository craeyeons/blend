"""
Learned Router for Hybrid PINN-CFD Domain Segregation.

This module implements a neural network-based router that learns to predict
optimal PINN/CFD region assignments based on:
- Local flow complexity metrics
- Boundary condition violations
- Spatial coordinates and distances
- Flow-aligned features

The router produces smooth, connected masks via custom loss functions that
encourage spatial coherence, upstream consistency, and region connectivity.

Architecture:
- CNN: For structured grids (default)
- GNN: For unstructured meshes (future extension)
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from scipy import ndimage
from scipy.ndimage import label as connected_components
from typing import Dict, Tuple, Optional, Callable
import warnings


class RouterFeatureExtractor:
    """
    Extracts input features for the router network from flow fields.
    
    Features computed per grid point:
    - Local complexity metric D(x_i): strain rate, vorticity, PDE residuals
    - Distance to inlet d_inlet(x_i)
    - Distance to cylinder d_cyl(x_i)
    - Flow-aligned coordinates (ξ, η)
    - Boundary condition violation D_BC(x_i)
    - Local flow quantities: (u, v, p, ||∇u||, ||∇v||)
    """
    
    def __init__(self, 
                 x_domain: Tuple[float, float] = (0, 2),
                 y_domain: Tuple[float, float] = (0, 1),
                 cylinder_center: Tuple[float, float] = (0.5, 0.5),
                 cylinder_radius: float = 0.1,
                 inlet_velocity: float = 1.0,
                 nu: float = 0.01):
        """
        Initialize feature extractor.
        
        Parameters:
        -----------
        x_domain, y_domain : tuple
            Domain bounds
        cylinder_center : tuple
            Cylinder center (Cx, Cy)
        cylinder_radius : float
            Cylinder radius
        inlet_velocity : float
            Maximum inlet velocity
        nu : float
            Kinematic viscosity
        """
        self.x_ini, self.x_f = x_domain
        self.y_ini, self.y_f = y_domain
        self.Cx, self.Cy = cylinder_center
        self.radius = cylinder_radius
        self.u_inlet = inlet_velocity
        self.nu = nu
        self.Lx = self.x_f - self.x_ini
        self.Ly = self.y_f - self.y_ini
    
    def inlet_profile(self, y: np.ndarray) -> np.ndarray:
        """Parabolic inlet velocity profile."""
        return 4 * self.u_inlet * (y - self.y_ini) * (self.y_f - y) / (self.Ly ** 2)
    
    def compute_distance_to_inlet(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute distance from each point to inlet (x = x_ini)."""
        return X - self.x_ini
    
    def compute_distance_to_cylinder(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """Compute distance from each point to cylinder surface."""
        dist_to_center = np.sqrt((X - self.Cx)**2 + (Y - self.Cy)**2)
        return np.maximum(dist_to_center - self.radius, 0.0)
    
    def compute_flow_aligned_coords(self, X: np.ndarray, Y: np.ndarray,
                                     u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute flow-aligned coordinates (ξ, η).
        
        ξ: streamwise coordinate (along mean flow direction)
        η: cross-stream coordinate (perpendicular to mean flow)
        
        For cylinder flow, mean flow is primarily in x-direction.
        """
        # Compute mean flow direction
        u_mean = np.mean(u[u > 0]) if np.any(u > 0) else 1.0
        v_mean = np.mean(v)
        flow_mag = np.sqrt(u_mean**2 + v_mean**2) + 1e-10
        
        # Unit vectors
        e_xi = np.array([u_mean, v_mean]) / flow_mag  # Streamwise
        e_eta = np.array([-v_mean, u_mean]) / flow_mag  # Cross-stream
        
        # Project coordinates
        xi = (X - self.x_ini) * e_xi[0] + (Y - self.Cy) * e_xi[1]
        eta = (X - self.x_ini) * e_eta[0] + (Y - self.Cy) * e_eta[1]
        
        return xi, eta
    
    def compute_boundary_violation(self, u: np.ndarray, v: np.ndarray,
                                    X: np.ndarray, Y: np.ndarray,
                                    u_pinn: np.ndarray, v_pinn: np.ndarray) -> np.ndarray:
        """
        Compute boundary condition violation D_BC(x_i) = ||u_PINN - u_BC||².
        
        Only non-zero at boundary points (inlet, walls, cylinder).
        """
        Ny, Nx = u.shape
        D_BC = np.zeros_like(u)
        
        # Inlet boundary (x = x_ini): should match parabolic profile
        y_inlet = Y[:, 0]
        u_inlet_target = self.inlet_profile(y_inlet)
        D_BC[:, 0] = (u_pinn[:, 0] - u_inlet_target)**2 + v_pinn[:, 0]**2
        
        # Top and bottom walls (y = y_ini, y = y_f): no-slip
        D_BC[0, :] = u_pinn[0, :]**2 + v_pinn[0, :]**2
        D_BC[-1, :] = u_pinn[-1, :]**2 + v_pinn[-1, :]**2
        
        # Cylinder surface: no-slip (points near cylinder)
        dist_to_cyl = self.compute_distance_to_cylinder(X, Y)
        dx = self.Lx / (Nx - 1)
        near_cylinder = dist_to_cyl < 2 * dx
        D_BC[near_cylinder] = u_pinn[near_cylinder]**2 + v_pinn[near_cylinder]**2
        
        return D_BC
    
    def compute_velocity_gradients(self, u: np.ndarray, v: np.ndarray,
                                    dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute ||∇u|| and ||∇v||."""
        # Gradient magnitude of u
        du_dx = np.zeros_like(u)
        du_dy = np.zeros_like(u)
        du_dx[1:-1, 1:-1] = (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)
        du_dy[1:-1, 1:-1] = (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy)
        grad_u_mag = np.sqrt(du_dx**2 + du_dy**2)
        
        # Gradient magnitude of v
        dv_dx = np.zeros_like(v)
        dv_dy = np.zeros_like(v)
        dv_dx[1:-1, 1:-1] = (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)
        dv_dy[1:-1, 1:-1] = (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
        grad_v_mag = np.sqrt(dv_dx**2 + dv_dy**2)
        
        return grad_u_mag, grad_v_mag
    
    def compute_complexity_score(self, u: np.ndarray, v: np.ndarray, p: np.ndarray,
                                  dx: float, dy: float) -> np.ndarray:
        """
        Compute local complexity metric D(x_i).
        
        Combines strain rate, vorticity, and PDE residuals.
        """
        from lib.complexity_scoring import ComplexityScorer
        
        scorer = ComplexityScorer(
            weights={'strain': 1.0, 'vorticity': 1.0, 'momentum': 2.0, 'continuity': 2.0},
            normalization='mean'
        )
        return scorer.compute_complexity_score(u, v, p, self.nu, dx, dy)
    
    def extract_features(self, u_pinn: np.ndarray, v_pinn: np.ndarray, p_pinn: np.ndarray,
                         X: np.ndarray, Y: np.ndarray,
                         dx: float, dy: float,
                         u_cfd: Optional[np.ndarray] = None,
                         v_cfd: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Extract all features for router input.
        
        Parameters:
        -----------
        u_pinn, v_pinn, p_pinn : ndarray of shape (Ny, Nx)
            PINN predicted fields
        X, Y : ndarray of shape (Ny, Nx)
            Coordinate grids
        dx, dy : float
            Grid spacing
        u_cfd, v_cfd : ndarray, optional
            CFD solution (for computing error-based features during training)
        
        Returns:
        --------
        features : ndarray of shape (Ny, Nx, num_features)
            Feature tensor for router input
        """
        Ny, Nx = u_pinn.shape
        
        # 1. Distance features
        d_inlet = self.compute_distance_to_inlet(X, Y)
        d_cyl = self.compute_distance_to_cylinder(X, Y)
        
        # 2. Flow-aligned coordinates
        xi, eta = self.compute_flow_aligned_coords(X, Y, u_pinn, v_pinn)
        
        # 3. Boundary condition violation
        D_BC = self.compute_boundary_violation(u_pinn, v_pinn, X, Y, u_pinn, v_pinn)
        
        # 4. Local flow quantities
        grad_u, grad_v = self.compute_velocity_gradients(u_pinn, v_pinn, dx, dy)
        
        # 5. Complexity score
        D = self.compute_complexity_score(u_pinn, v_pinn, p_pinn, dx, dy)
        
        # 6. Normalized coordinates
        x_norm = (X - self.x_ini) / self.Lx
        y_norm = (Y - self.y_ini) / self.Ly
        
        # Normalize features
        d_inlet_norm = d_inlet / self.Lx
        d_cyl_norm = d_cyl / self.radius
        xi_norm = xi / self.Lx
        eta_norm = eta / self.Ly
        
        # Normalize flow quantities by reference scales
        u_ref = self.u_inlet
        grad_ref = u_ref / self.radius
        D_ref = np.mean(D) + 1e-10
        D_BC_ref = np.max(D_BC) + 1e-10
        
        u_norm = u_pinn / u_ref
        v_norm = v_pinn / u_ref
        p_norm = p_pinn / (u_ref ** 2)  # Dynamic pressure scaling
        grad_u_norm = grad_u / grad_ref
        grad_v_norm = grad_v / grad_ref
        D_norm = D / D_ref
        D_BC_norm = D_BC / D_BC_ref
        
        # Stack features: (Ny, Nx, num_features)
        features = np.stack([
            x_norm,        # 0: Normalized x coordinate
            y_norm,        # 1: Normalized y coordinate
            d_inlet_norm,  # 2: Distance to inlet (normalized)
            d_cyl_norm,    # 3: Distance to cylinder (normalized)
            xi_norm,       # 4: Streamwise coordinate
            eta_norm,      # 5: Cross-stream coordinate
            u_norm,        # 6: u velocity
            v_norm,        # 7: v velocity
            p_norm,        # 8: pressure
            grad_u_norm,   # 9: ||∇u||
            grad_v_norm,   # 10: ||∇v||
            D_norm,        # 11: Complexity score
            D_BC_norm,     # 12: Boundary violation
        ], axis=-1)
        
        return features.astype(np.float32)


class RouterCNN(Model):
    """
    Convolutional Neural Network router for structured grids.
    
    Takes spatial features as input and outputs binary decision
    r(x_i) ∈ {0,1}: 0 = accept PINN, 1 = reject (use CFD).
    
    Architecture uses dilated convolutions to capture multi-scale spatial context.
    """
    
    def __init__(self, num_features: int = 13, 
                 filters: list = [32, 64, 64, 32],
                 kernel_size: int = 3,
                 use_batch_norm: bool = True,
                 dropout_rate: float = 0.1,
                 **kwargs):
        """
        Initialize router CNN.
        
        Parameters:
        -----------
        num_features : int
            Number of input feature channels
        filters : list
            Number of filters in each conv layer
        kernel_size : int
            Convolution kernel size
        use_batch_norm : bool
            Whether to use batch normalization
        dropout_rate : float
            Dropout rate for regularization
        """
        super().__init__(**kwargs)
        
        self.num_features = num_features
        self.use_batch_norm = use_batch_norm
        
        # Multi-scale feature extraction with dilated convolutions
        self.conv_layers = []
        self.bn_layers = []
        
        dilation_rates = [1, 2, 4, 1]  # Multi-scale receptive field
        
        for i, (f, d) in enumerate(zip(filters, dilation_rates)):
            conv = layers.Conv2D(
                f, kernel_size, 
                padding='same',
                dilation_rate=d,
                activation=None,
                kernel_initializer='he_normal',
                name=f'conv_{i}'
            )
            self.conv_layers.append(conv)
            
            if use_batch_norm:
                bn = layers.BatchNormalization(name=f'bn_{i}')
                self.bn_layers.append(bn)
        
        self.activation = layers.Activation('relu')
        self.dropout = layers.Dropout(dropout_rate)
        
        # Output layer: probability of rejecting PINN (using CFD)
        self.output_conv = layers.Conv2D(
            1, 1, 
            padding='same',
            activation='sigmoid',
            kernel_initializer='glorot_uniform',
            name='output'
        )
    
    def call(self, inputs, training=False):
        """
        Forward pass.
        
        Parameters:
        -----------
        inputs : tensor of shape (batch, Ny, Nx, num_features)
            Input features
        training : bool
            Whether in training mode
        
        Returns:
        --------
        r : tensor of shape (batch, Ny, Nx, 1)
            Rejection probability (0 = PINN, 1 = CFD)
        """
        x = inputs
        
        for i, conv in enumerate(self.conv_layers):
            x = conv(x)
            if self.use_batch_norm:
                x = self.bn_layers[i](x, training=training)
            x = self.activation(x)
            x = self.dropout(x, training=training)
        
        # Output
        r = self.output_conv(x)
        
        return r
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_features': self.num_features,
            'filters': [c.filters for c in self.conv_layers],
            'use_batch_norm': self.use_batch_norm,
        })
        return config


class RouterLoss:
    """
    Custom loss function for router training.
    
    L_total = L_base + λ_spatial * L_spatial + λ_upstream * L_upstream + λ_connect * L_connect
    
    Components:
    - L_base: Local complexity-based loss
    - L_spatial: Spatial regularization for smoothness
    - L_upstream: Upstream penalty for boundary condition awareness
    - L_connect: Connectivity penalty to avoid fragmentation
    """
    
    def __init__(self,
                 threshold: float = 1.0,
                 lambda_spatial: float = 0.1,
                 lambda_upstream: float = 1.0,
                 lambda_connect: float = 0.5,
                 h_scale: float = 1.0,
                 alpha_flow: float = 2.0,
                 delta_influence: float = 0.2,
                 xi_threshold: float = 0.3):
        """
        Initialize loss function.
        
        Parameters:
        -----------
        threshold : float
            Target complexity for accepted PINN regions (τ)
        lambda_spatial : float
            Weight for spatial regularization
        lambda_upstream : float
            Weight for upstream penalty
        lambda_connect : float
            Weight for connectivity penalty
        h_scale : float
            Spatial scale for edge weights (in grid units)
        alpha_flow : float
            Flow-directional coupling strength
        delta_influence : float
            Influence length for upstream penalty (fraction of domain)
        xi_threshold : float
            Streamwise threshold for upstream region (fraction of domain)
        """
        self.threshold = threshold
        self.lambda_spatial = lambda_spatial
        self.lambda_upstream = lambda_upstream
        self.lambda_connect = lambda_connect
        self.h_scale = h_scale
        self.alpha_flow = alpha_flow
        self.delta_influence = delta_influence
        self.xi_threshold = xi_threshold
    
    def base_loss(self, r: tf.Tensor, D: tf.Tensor) -> tf.Tensor:
        """
        Base loss: penalize complexity in accepted regions, penalize excessive rejection.
        
        L_base = (1/N) Σ_i [(1 - r(x_i)) · D(x_i) + r(x_i) · τ]
        
        Parameters:
        -----------
        r : tensor of shape (batch, Ny, Nx, 1)
            Rejection probability
        D : tensor of shape (batch, Ny, Nx, 1)
            Local complexity metric
        
        Returns:
        --------
        loss : scalar tensor
        """
        # When r ≈ 0 (accept PINN): penalize high complexity
        # When r ≈ 1 (reject/CFD): fixed cost τ
        accept_cost = (1.0 - r) * D
        reject_cost = r * self.threshold
        
        return tf.reduce_mean(accept_cost + reject_cost)
    
    def spatial_loss(self, r: tf.Tensor, 
                     X: tf.Tensor, Y: tf.Tensor,
                     u_mean: tf.Tensor) -> tf.Tensor:
        """
        Spatial regularization: encourage smooth masks with flow-directional coupling.
        
        L_spatial = Σ_{(i,j) ∈ edges} w_ij · |r(x_i) - r(x_j)|
        
        w_ij = exp(-||x_i - x_j||²/h²) · (1 + α · flow_alignment(i,j))
        """
        # Compute differences to neighbors (4-connectivity)
        # Right neighbor
        r_right = tf.roll(r, shift=-1, axis=2)
        diff_right = tf.abs(r[:, :, :-1, :] - r_right[:, :, :-1, :])
        
        # Bottom neighbor
        r_bottom = tf.roll(r, shift=-1, axis=1)
        diff_bottom = tf.abs(r[:, :-1, :, :] - r_bottom[:, :-1, :, :])
        
        # For uniform grid, w_ij ≈ 1 + α * flow_alignment
        # Flow alignment: positive for neighbors in flow direction
        # Assuming mean flow is in +x direction for cylinder flow
        w_right = 1.0 + self.alpha_flow * tf.ones_like(diff_right)
        w_bottom = 1.0  # Cross-stream direction
        
        loss_right = tf.reduce_mean(w_right * diff_right)
        loss_bottom = tf.reduce_mean(w_bottom * diff_bottom)
        
        return loss_right + loss_bottom
    
    def upstream_loss(self, r: tf.Tensor, 
                      D_BC: tf.Tensor, 
                      D: tf.Tensor,
                      d_inlet: tf.Tensor,
                      xi: tf.Tensor) -> tf.Tensor:
        """
        Upstream penalty: strongly discourage accepting PINN near inlet where BC violations are high.
        
        L_upstream = Σ_{i ∈ upstream_region} r(x_i) · [D_BC(x_i) + D(x_i)] · exp(-d_i/δ)
        
        Note: We want to REJECT (r=1) where BC violations are high.
        But original formulation penalizes rejection. We invert to penalize acceptance:
        
        L_upstream = Σ_{i ∈ upstream} (1 - r(x_i)) · [D_BC(x_i) + D(x_i)] · exp(-d_i/δ)
        """
        # Upstream region mask
        upstream_mask = tf.cast(xi < self.xi_threshold, tf.float32)
        
        # Exponential decay from inlet
        influence = tf.exp(-d_inlet / self.delta_influence)
        
        # Penalize accepting PINN (low r) where BC violation is high
        accept_penalty = (1.0 - r) * (D_BC + D) * influence * upstream_mask[..., tf.newaxis]
        
        return tf.reduce_mean(accept_penalty)
    
    def connectivity_loss(self, r: tf.Tensor) -> tf.Tensor:
        """
        Connectivity penalty: encourage accepted (PINN) region to be connected.
        
        Uses a differentiable approximation based on spatial gradients.
        Penalizes "islands" of accepted regions by encouraging smoothness.
        
        For actual connected component analysis, use post-processing.
        """
        # Compute second-order differences (Laplacian-like)
        # High values indicate isolated points
        r_pad = tf.pad(r, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
        
        laplacian = (
            r_pad[:, 2:, 1:-1, :] + r_pad[:, :-2, 1:-1, :] +
            r_pad[:, 1:-1, 2:, :] + r_pad[:, 1:-1, :-2, :] -
            4 * r
        )
        
        # Penalize high absolute Laplacian (indicates isolated regions)
        return tf.reduce_mean(tf.abs(laplacian))
    
    def supervised_loss(self, r: tf.Tensor, 
                        labels: tf.Tensor,
                        weights: Optional[tf.Tensor] = None) -> tf.Tensor:
        """
        Supervised loss: binary cross-entropy with optional sample weights.
        
        Parameters:
        -----------
        r : tensor of shape (batch, Ny, Nx, 1)
            Predicted rejection probability
        labels : tensor of shape (batch, Ny, Nx, 1)
            Ground truth labels (1 = should reject PINN)
        weights : tensor, optional
            Per-sample weights
        """
        bce = tf.keras.losses.binary_crossentropy(labels, r)
        
        if weights is not None:
            bce = bce * weights
        
        return tf.reduce_mean(bce)
    
    def __call__(self, r: tf.Tensor, 
                 D: tf.Tensor, 
                 D_BC: tf.Tensor,
                 d_inlet: tf.Tensor, 
                 xi: tf.Tensor,
                 X: tf.Tensor, 
                 Y: tf.Tensor,
                 u_mean: tf.Tensor,
                 labels: Optional[tf.Tensor] = None,
                 weights: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:
        """
        Compute total loss with all components.
        
        Returns:
        --------
        dict with 'total', 'base', 'spatial', 'upstream', 'connect' losses
        """
        losses = {}
        
        # Base loss
        losses['base'] = self.base_loss(r, D)
        
        # Spatial regularization
        losses['spatial'] = self.spatial_loss(r, X, Y, u_mean)
        
        # Upstream penalty
        losses['upstream'] = self.upstream_loss(r, D_BC, D, d_inlet, xi)
        
        # Connectivity penalty
        losses['connect'] = self.connectivity_loss(r)
        
        # Total unsupervised loss
        total = (losses['base'] + 
                 self.lambda_spatial * losses['spatial'] +
                 self.lambda_upstream * losses['upstream'] +
                 self.lambda_connect * losses['connect'])
        
        # Add supervised loss if labels provided
        if labels is not None:
            losses['supervised'] = self.supervised_loss(r, labels, weights)
            total = total + losses['supervised']
        
        losses['total'] = total
        
        return losses


class RouterTrainer:
    """
    Training pipeline for the learned router.
    
    Supports both:
    1. Supervised training: using ground truth labels from PINN-CFD error comparison
    2. Unsupervised training: using physics-based loss functions only
    """
    
    def __init__(self,
                 router: RouterCNN,
                 feature_extractor: RouterFeatureExtractor,
                 loss_fn: RouterLoss,
                 learning_rate: float = 1e-3,
                 clip_norm: float = 1.0):
        """
        Initialize trainer.
        
        Parameters:
        -----------
        router : RouterCNN
            Router network
        feature_extractor : RouterFeatureExtractor
            Feature extraction module
        loss_fn : RouterLoss
            Loss function
        learning_rate : float
            Learning rate
        clip_norm : float
            Gradient clipping norm
        """
        self.router = router
        self.feature_extractor = feature_extractor
        self.loss_fn = loss_fn
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)
        self.clip_norm = clip_norm
        
        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.val_loss = tf.keras.metrics.Mean(name='val_loss')
    
    def generate_labels_from_error(self, 
                                    u_pinn: np.ndarray, v_pinn: np.ndarray,
                                    u_cfd: np.ndarray, v_cfd: np.ndarray,
                                    error_threshold: float = 0.1,
                                    smooth_labels: bool = True) -> np.ndarray:
        """
        Generate ground truth labels from PINN-CFD error comparison.
        
        Labels:
        - 1 (reject PINN): where |u_PINN - u_CFD| > threshold
        - 0 (accept PINN): where PINN is accurate
        
        Parameters:
        -----------
        u_pinn, v_pinn : ndarray
            PINN velocity predictions
        u_cfd, v_cfd : ndarray
            CFD ground truth
        error_threshold : float
            Relative error threshold for rejection
        smooth_labels : bool
            Whether to dilate rejection region for smoother boundaries
        
        Returns:
        --------
        labels : ndarray of shape (Ny, Nx)
            Binary labels
        """
        # Compute relative error
        u_ref = np.max(np.abs(u_cfd)) + 1e-10
        error_u = np.abs(u_pinn - u_cfd) / u_ref
        error_v = np.abs(v_pinn - v_cfd) / u_ref
        error = np.maximum(error_u, error_v)
        
        # Create binary labels
        labels = (error > error_threshold).astype(np.float32)
        
        # Optionally smooth labels (dilate rejection region)
        if smooth_labels:
            labels = ndimage.binary_dilation(labels > 0.5, iterations=2).astype(np.float32)
        
        return labels
    
    def compute_sample_weights(self, 
                                labels: np.ndarray,
                                d_inlet: np.ndarray,
                                D_BC: np.ndarray) -> np.ndarray:
        """
        Compute sample weights emphasizing boundary and upstream regions.
        
        Parameters:
        -----------
        labels : ndarray
            Binary labels
        d_inlet : ndarray
            Distance to inlet
        D_BC : ndarray
            Boundary condition violation
        
        Returns:
        --------
        weights : ndarray
            Per-sample weights
        """
        # Base weight
        weights = np.ones_like(labels)
        
        # Higher weight near inlet (upstream)
        inlet_weight = np.exp(-d_inlet / (0.2 * np.max(d_inlet)))
        weights = weights + 2.0 * inlet_weight
        
        # Higher weight where BC violation is high
        D_BC_norm = D_BC / (np.max(D_BC) + 1e-10)
        weights = weights + 2.0 * D_BC_norm
        
        # Normalize
        weights = weights / np.mean(weights)
        
        return weights.astype(np.float32)
    
    @tf.function
    def train_step(self, features: tf.Tensor, 
                   D: tf.Tensor, D_BC: tf.Tensor,
                   d_inlet: tf.Tensor, xi: tf.Tensor,
                   X: tf.Tensor, Y: tf.Tensor,
                   labels: Optional[tf.Tensor] = None,
                   weights: Optional[tf.Tensor] = None) -> Dict[str, tf.Tensor]:
        """
        Single training step.
        """
        with tf.GradientTape() as tape:
            # Forward pass
            r = self.router(features, training=True)
            
            # Compute loss
            u_mean = tf.constant(1.0, dtype=tf.float32)  # Assuming +x flow
            losses = self.loss_fn(r, D, D_BC, d_inlet, xi, X, Y, u_mean, labels, weights)
        
        # Backward pass
        gradients = tape.gradient(losses['total'], self.router.trainable_variables)
        
        # Clip gradients
        gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.router.trainable_variables))
        
        return losses
    
    def prepare_training_data(self, 
                               u_pinn: np.ndarray, v_pinn: np.ndarray, p_pinn: np.ndarray,
                               X: np.ndarray, Y: np.ndarray,
                               dx: float, dy: float,
                               u_cfd: Optional[np.ndarray] = None,
                               v_cfd: Optional[np.ndarray] = None,
                               error_threshold: float = 0.1) -> Dict[str, tf.Tensor]:
        """
        Prepare training data from flow fields.
        
        Returns:
        --------
        data : dict with tensors for training
        """
        # Extract features
        features = self.feature_extractor.extract_features(
            u_pinn, v_pinn, p_pinn, X, Y, dx, dy
        )
        
        # Compute auxiliary quantities
        d_inlet = self.feature_extractor.compute_distance_to_inlet(X, Y)
        D = self.feature_extractor.compute_complexity_score(u_pinn, v_pinn, p_pinn, dx, dy)
        D_BC = self.feature_extractor.compute_boundary_violation(
            u_pinn, v_pinn, X, Y, u_pinn, v_pinn
        )
        xi, _ = self.feature_extractor.compute_flow_aligned_coords(X, Y, u_pinn, v_pinn)
        
        # Normalize
        Lx = self.feature_extractor.Lx
        d_inlet_norm = d_inlet / Lx
        xi_norm = xi / Lx
        D_norm = D / (np.mean(D) + 1e-10)
        D_BC_norm = D_BC / (np.max(D_BC) + 1e-10)
        
        data = {
            'features': tf.constant(features[np.newaxis, ...], dtype=tf.float32),
            'd_inlet': tf.constant(d_inlet_norm[np.newaxis, ..., np.newaxis], dtype=tf.float32),
            'xi': tf.constant(xi_norm[np.newaxis, ...], dtype=tf.float32),
            'D': tf.constant(D_norm[np.newaxis, ..., np.newaxis], dtype=tf.float32),
            'D_BC': tf.constant(D_BC_norm[np.newaxis, ..., np.newaxis], dtype=tf.float32),
            'X': tf.constant(X[np.newaxis, ...], dtype=tf.float32),
            'Y': tf.constant(Y[np.newaxis, ...], dtype=tf.float32),
        }
        
        # Generate labels if CFD solution is available
        if u_cfd is not None and v_cfd is not None:
            labels = self.generate_labels_from_error(
                u_pinn, v_pinn, u_cfd, v_cfd, error_threshold
            )
            weights = self.compute_sample_weights(labels, d_inlet_norm, D_BC_norm)
            
            data['labels'] = tf.constant(labels[np.newaxis, ..., np.newaxis], dtype=tf.float32)
            data['weights'] = tf.constant(weights[np.newaxis, ...], dtype=tf.float32)
        
        return data
    
    def train(self, 
              training_data: list,
              epochs: int = 100,
              verbose: bool = True) -> Dict[str, list]:
        """
        Train the router on multiple flow cases.
        
        Parameters:
        -----------
        training_data : list
            List of dictionaries with training data for each case
        epochs : int
            Number of training epochs
        verbose : bool
            Whether to print progress
        
        Returns:
        --------
        history : dict with loss histories
        """
        history = {'total': [], 'base': [], 'spatial': [], 'upstream': [], 'connect': []}
        
        for epoch in range(epochs):
            epoch_losses = {k: [] for k in history.keys()}
            
            for data in training_data:
                losses = self.train_step(
                    data['features'],
                    data['D'],
                    data['D_BC'],
                    data['d_inlet'],
                    data['xi'],
                    data['X'],
                    data['Y'],
                    data.get('labels'),
                    data.get('weights')
                )
                
                for k in epoch_losses:
                    if k in losses:
                        epoch_losses[k].append(float(losses[k]))
            
            # Record mean losses
            for k in history:
                if epoch_losses[k]:
                    history[k].append(np.mean(epoch_losses[k]))
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs} - "
                      f"Loss: {history['total'][-1]:.4f} "
                      f"(base: {history['base'][-1]:.4f}, "
                      f"spatial: {history['spatial'][-1]:.4f}, "
                      f"upstream: {history['upstream'][-1]:.4f})")
        
        return history


class LearnedRouter:
    """
    High-level interface for using the learned router.
    
    Combines feature extraction, prediction, and mask post-processing.
    """
    
    def __init__(self,
                 router: RouterCNN,
                 feature_extractor: RouterFeatureExtractor,
                 threshold: float = 0.5,
                 min_cfd_region_size: int = 50,
                 smooth_iterations: int = 2):
        """
        Initialize learned router.
        
        Parameters:
        -----------
        router : RouterCNN
            Trained router network
        feature_extractor : RouterFeatureExtractor
            Feature extraction module
        threshold : float
            Probability threshold for rejection (default: 0.5)
        min_cfd_region_size : int
            Minimum CFD region size (smaller regions merged to PINN)
        smooth_iterations : int
            Number of morphological smoothing iterations
        """
        self.router = router
        self.feature_extractor = feature_extractor
        self.threshold = threshold
        self.min_cfd_region_size = min_cfd_region_size
        self.smooth_iterations = smooth_iterations
    
    def predict_mask(self, 
                     u_pinn: np.ndarray, v_pinn: np.ndarray, p_pinn: np.ndarray,
                     X: np.ndarray, Y: np.ndarray,
                     dx: float, dy: float,
                     cylinder_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict binary mask for PINN/CFD regions.
        
        Parameters:
        -----------
        u_pinn, v_pinn, p_pinn : ndarray
            PINN predicted fields
        X, Y : ndarray
            Coordinate grids
        dx, dy : float
            Grid spacing
        cylinder_mask : ndarray, optional
            Cylinder interior mask (these points always use CFD)
        
        Returns:
        --------
        mask : ndarray of shape (Ny, Nx)
            Binary mask (1 = CFD, 0 = PINN)
        r_prob : ndarray
            Raw rejection probabilities
        """
        # Extract features
        features = self.feature_extractor.extract_features(
            u_pinn, v_pinn, p_pinn, X, Y, dx, dy
        )
        
        # Predict
        features_tf = tf.constant(features[np.newaxis, ...], dtype=tf.float32)
        r_prob = self.router(features_tf, training=False).numpy().squeeze()
        
        # Threshold to binary
        mask = (r_prob > self.threshold).astype(np.int32)
        
        # Post-processing: smooth mask
        if self.smooth_iterations > 0:
            # Opening then closing for smooth boundaries
            mask = ndimage.binary_opening(mask, iterations=self.smooth_iterations)
            mask = ndimage.binary_closing(mask, iterations=self.smooth_iterations)
            mask = mask.astype(np.int32)
        
        # Remove small CFD regions
        if self.min_cfd_region_size > 0:
            labeled, num_features = connected_components(mask)
            for i in range(1, num_features + 1):
                region = labeled == i
                if np.sum(region) < self.min_cfd_region_size:
                    mask[region] = 0  # Convert to PINN
        
        # Force domain boundaries to CFD
        mask[0, :] = 1   # Bottom
        mask[-1, :] = 1  # Top
        mask[:, 0] = 1   # Inlet
        mask[:, -1] = 1  # Outlet
        
        # Force cylinder region to CFD
        if cylinder_mask is not None:
            # Dilate cylinder for boundary layer
            dilated_cyl = ndimage.binary_dilation(
                cylinder_mask > 0, iterations=2
            ).astype(np.int32)
            mask = np.maximum(mask, dilated_cyl)
        
        return mask, r_prob
    
    def save(self, path: str):
        """Save router weights."""
        self.router.save_weights(path)
    
    def load(self, path: str):
        """Load router weights."""
        self.router.load_weights(path)


def create_router(x_domain: Tuple[float, float] = (0, 2),
                  y_domain: Tuple[float, float] = (0, 1),
                  cylinder_center: Tuple[float, float] = (0.5, 0.5),
                  cylinder_radius: float = 0.1,
                  inlet_velocity: float = 1.0,
                  nu: float = 0.01,
                  **router_kwargs) -> Tuple[LearnedRouter, RouterTrainer]:
    """
    Factory function to create a complete router system.
    
    Returns:
    --------
    router : LearnedRouter
        Ready-to-use router
    trainer : RouterTrainer
        Trainer for the router
    """
    # Create components
    feature_extractor = RouterFeatureExtractor(
        x_domain=x_domain,
        y_domain=y_domain,
        cylinder_center=cylinder_center,
        cylinder_radius=cylinder_radius,
        inlet_velocity=inlet_velocity,
        nu=nu
    )
    
    router_cnn = RouterCNN(
        num_features=13,
        filters=router_kwargs.get('filters', [32, 64, 64, 32]),
        kernel_size=router_kwargs.get('kernel_size', 3),
        use_batch_norm=router_kwargs.get('use_batch_norm', True),
        dropout_rate=router_kwargs.get('dropout_rate', 0.1)
    )
    
    loss_fn = RouterLoss(
        threshold=router_kwargs.get('threshold', 1.0),
        lambda_spatial=router_kwargs.get('lambda_spatial', 0.1),
        lambda_upstream=router_kwargs.get('lambda_upstream', 1.0),
        lambda_connect=router_kwargs.get('lambda_connect', 0.5),
        h_scale=router_kwargs.get('h_scale', 1.0),
        alpha_flow=router_kwargs.get('alpha_flow', 2.0),
        delta_influence=router_kwargs.get('delta_influence', 0.2),
        xi_threshold=router_kwargs.get('xi_threshold', 0.3)
    )
    
    trainer = RouterTrainer(
        router=router_cnn,
        feature_extractor=feature_extractor,
        loss_fn=loss_fn,
        learning_rate=router_kwargs.get('learning_rate', 1e-3)
    )
    
    learned_router = LearnedRouter(
        router=router_cnn,
        feature_extractor=feature_extractor,
        threshold=router_kwargs.get('inference_threshold', 0.5),
        min_cfd_region_size=router_kwargs.get('min_cfd_region_size', 50),
        smooth_iterations=router_kwargs.get('smooth_iterations', 2)
    )
    
    return learned_router, trainer
