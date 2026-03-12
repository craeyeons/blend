"""
Train the CNN-based router for hybrid PINN-CFD simulations for CAVITY FLOW.

This script trains a router to decide which regions should use PINN vs CFD
for the lid-driven cavity flow problem.

The cavity PINN outputs (psi, p) where psi is the stream function.
Velocities are computed as: u = ∂ψ/∂y, v = -∂ψ/∂x

Usage:
    python train_router_cavity.py --model-path ./models/pinn_cavity_flow.h5 --epochs 200
    python train_router_cavity.py --beta 0.05 --lambda-tv 0.02 --epochs 500
"""

import argparse
import os
import numpy as np
import tensorflow as tf
from datetime import datetime

# Configure TensorFlow GPU memory growth to avoid cuDNN issues
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU memory growth setting failed: {e}")

# Import router components
from lib.router import (
    RouterCNN,
    create_router_input,
    create_cavity_setup,
    plot_router_output,
    plot_training_history
)
from lib.network import Network as CavityNetwork


class CavityPINNResidualComputer:
    """
    Computes PINN physics residuals for the Navier-Stokes equations.
    Specialized for cavity flow with stream function output (psi, p).
    """
    
    def __init__(self, pinn_model, nu=0.01, rho=1.0, x_domain=(0, 1), y_domain=(0, 1)):
        """
        Initialize the residual computer.
        
        Parameters:
        -----------
        pinn_model : tf.keras.Model
            Pre-trained PINN model that outputs (psi, p) given (x, y)
        nu : float
            Kinematic viscosity
        rho : float
            Fluid density
        x_domain, y_domain : tuple
            Domain bounds
        """
        self.pinn_model = pinn_model
        self.nu = nu
        self.rho = rho
        self.x_domain = x_domain
        self.y_domain = y_domain
    
    @tf.function
    def compute_velocities_from_psi(self, xy):
        """
        Compute velocities from stream function.
        u = ∂ψ/∂y, v = -∂ψ/∂x
        """
        with tf.GradientTape() as tape:
            tape.watch(xy)
            psi_p = self.pinn_model(xy, training=False)
            psi = psi_p[:, 0]
        
        grad_psi = tape.gradient(psi, xy)
        u = grad_psi[:, 1]   # ∂ψ/∂y
        v = -grad_psi[:, 0]  # -∂ψ/∂x
        p = psi_p[:, 1]
        
        return u, v, p
    
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
                # Get velocities from stream function
                psi_p = self.pinn_model(xy, training=False)
                psi = psi_p[:, 0]
                p = psi_p[:, 1]
            
            # Compute velocities: u = ∂ψ/∂y, v = -∂ψ/∂x
            grad_psi = tape1.gradient(psi, xy)
            u = grad_psi[:, 1]   # ∂ψ/∂y
            v = -grad_psi[:, 0]  # -∂ψ/∂x
            
            # Compute velocity gradients
            with tf.GradientTape(persistent=True) as tape_inner:
                tape_inner.watch(xy)
                psi_p_inner = self.pinn_model(xy, training=False)
                psi_inner = psi_p_inner[:, 0]
            grad_psi_inner = tape_inner.gradient(psi_inner, xy)
            u_inner = grad_psi_inner[:, 1]
            v_inner = -grad_psi_inner[:, 0]
            del tape_inner
            
            grad_u = tape1.gradient(u_inner, xy)
            grad_v = tape1.gradient(v_inner, xy)
            grad_p = tape1.gradient(p, xy)
            
            if grad_u is not None:
                du_dx = grad_u[:, 0]
                du_dy = grad_u[:, 1]
            else:
                du_dx = tf.zeros_like(u)
                du_dy = tf.zeros_like(u)
            
            if grad_v is not None:
                dv_dx = grad_v[:, 0]
                dv_dy = grad_v[:, 1]
            else:
                dv_dx = tf.zeros_like(v)
                dv_dy = tf.zeros_like(v)
            
            if grad_p is not None:
                dp_dx = grad_p[:, 0]
                dp_dy = grad_p[:, 1]
            else:
                dp_dx = tf.zeros_like(p)
                dp_dy = tf.zeros_like(p)
        
        # Second derivatives for viscous term
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
        # For stream function, this should be ~0 by construction
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
        """Compute weighted total residual."""
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
        
        u, v, p = self.compute_velocities_from_psi(xy)
        
        shape = tf.shape(X)
        u = tf.reshape(u, shape)
        v = tf.reshape(v, shape)
        p = tf.reshape(p, shape)
        return u, v, p
    
    def compute_bc_error(self, X, Y, bc_mask, bc_u, bc_v):
        """Compute boundary condition error at BC locations."""
        u, v, p = self.get_pinn_predictions(X, Y)
        
        u_error = tf.abs(u - bc_u) * bc_mask
        v_error = tf.abs(v - bc_v) * bc_mask
        
        bc_error = tf.sqrt(u_error**2 + v_error**2 + 1e-10)
        return bc_error
    
    def compute_total_residual_with_bc(self, X, Y, bc_mask, bc_u, bc_v, weights=None):
        """Compute total residual including BC error."""
        if weights is None:
            weights = {
                'continuity': 1.0,
                'momentum': 1.0,
                'bc_local': 2.0,
                'bc_propagated': 1.5
            }
        
        def clip_and_normalize(x, percentile=0.95, max_scale=1.5):
            x_flat = tf.reshape(x, [-1])
            num_elements = tf.shape(x_flat)[0]
            k = tf.cast(tf.cast(num_elements, tf.float32) * percentile, tf.int32)
            k = tf.maximum(k, 1)
            top_k_vals, _ = tf.math.top_k(x_flat, k=k)
            p95_val = top_k_vals[-1]
            x_clipped = tf.minimum(x, p95_val * max_scale)
            x_norm = x_clipped / (p95_val + 1e-10)
            return x_norm
        
        X_tf = tf.cast(X, tf.float32)
        Y_tf = tf.cast(Y, tf.float32)
        bc_mask_tf = tf.cast(bc_mask, tf.float32)
        bc_u_tf = tf.cast(bc_u, tf.float32)
        bc_v_tf = tf.cast(bc_v, tf.float32)
        
        continuity, momentum = self.compute_residuals(X_tf, Y_tf)
        bc_error = self.compute_bc_error(X_tf, Y_tf, bc_mask_tf, bc_u_tf, bc_v_tf)
        
        continuity_norm = clip_and_normalize(continuity)
        momentum_norm = clip_and_normalize(momentum)
        bc_error_norm = clip_and_normalize(bc_error)
        
        total = (weights['continuity'] * continuity_norm + 
                 weights['momentum'] * momentum_norm +
                 weights['bc_local'] * bc_error_norm)
        
        return total


class CavityRouterTrainer:
    """
    Trainer for cavity flow router.
    """
    
    def __init__(self, router, pinn_model, beta=0.1, lambda_tv=0.01,
                 lambda_entropy=0.1, lambda_variance=0.05,
                 grad_clip_norm=None, residual_weights=None,
                 nu=0.01, rho=1.0, x_domain=(0, 1), y_domain=(0, 1)):
        self.router = router
        self.pinn_model = pinn_model
        self.beta = beta
        self.lambda_tv = lambda_tv
        self.lambda_entropy = lambda_entropy
        self.lambda_variance = lambda_variance
        self.grad_clip_norm = grad_clip_norm
        
        if residual_weights is None:
            residual_weights = {
                'continuity': 1.0,
                'momentum': 1.0,
                'bc_local': 2.0,
                'bc_propagated': 1.5
            }
        self.residual_weights = residual_weights
        
        self.residual_computer = CavityPINNResidualComputer(
            pinn_model=pinn_model,
            nu=nu, rho=rho,
            x_domain=x_domain,
            y_domain=y_domain
        )
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
    
    def compute_loss(self, inputs, X, Y, layout_mask, bc_mask, bc_u, bc_v):
        """Compute total loss for router training."""
        r = self.router(inputs, training=True)
        r = tf.squeeze(r, axis=0)
        r = tf.squeeze(r, axis=-1)
        
        layout_mask_tf = tf.cast(layout_mask, tf.float32)
        bc_mask_tf = tf.cast(bc_mask, tf.float32)
        bc_u_tf = tf.cast(bc_u, tf.float32)
        bc_v_tf = tf.cast(bc_v, tf.float32)
        X_tf = tf.cast(X, tf.float32)
        Y_tf = tf.cast(Y, tf.float32)
        
        # Compute physics residual
        residual = self.residual_computer.compute_total_residual_with_bc(
            X_tf, Y_tf, bc_mask_tf, bc_u_tf, bc_v_tf,
            weights=self.residual_weights
        )
        
        # Mask to fluid region
        r_fluid = r * layout_mask_tf
        residual_fluid = residual * layout_mask_tf
        n_fluid = tf.reduce_sum(layout_mask_tf) + 1e-10
        
        # CFD cost: β * mean(r)
        cfd_cost = self.beta * tf.reduce_sum(r_fluid) / n_fluid
        
        # Residual loss: mean((1-r) * R) where R is physics residual
        residual_loss = tf.reduce_sum((1.0 - r_fluid) * residual_fluid) / n_fluid
        
        # Total variation regularization
        r_2d = r * layout_mask_tf
        tv_h = tf.reduce_sum(tf.abs(r_2d[:, 1:] - r_2d[:, :-1]))
        tv_v = tf.reduce_sum(tf.abs(r_2d[1:, :] - r_2d[:-1, :]))
        tv_loss = self.lambda_tv * (tv_h + tv_v) / n_fluid
        
        # Entropy regularization: encourage non-extreme values
        r_clipped = tf.clip_by_value(r_fluid / (layout_mask_tf + 1e-10), 1e-7, 1 - 1e-7)
        entropy = -tf.reduce_mean(r_clipped * tf.math.log(r_clipped) + 
                                  (1 - r_clipped) * tf.math.log(1 - r_clipped))
        entropy_term = -self.lambda_entropy * entropy
        
        # Variance regularization
        mean_r = tf.reduce_sum(r_fluid) / n_fluid
        variance = tf.reduce_sum(layout_mask_tf * (r_fluid / (layout_mask_tf + 1e-10) - mean_r)**2) / n_fluid
        variance_term = -self.lambda_variance * variance
        
        total_loss = cfd_cost + residual_loss + tv_loss + entropy_term + variance_term
        
        return total_loss, cfd_cost, residual_loss, tv_loss, entropy_term, variance_term
    
    @tf.function
    def train_step(self, inputs, X, Y, layout_mask, bc_mask, bc_u, bc_v):
        """Single training step."""
        with tf.GradientTape() as tape:
            total_loss, cfd_cost, residual_loss, tv_loss, entropy_term, variance_term = \
                self.compute_loss(inputs, X, Y, layout_mask, bc_mask, bc_u, bc_v)
        
        gradients = tape.gradient(total_loss, self.router.trainable_variables)
        
        if self.grad_clip_norm is not None:
            gradients, _ = tf.clip_by_global_norm(gradients, self.grad_clip_norm)
        
        self.optimizer.apply_gradients(zip(gradients, self.router.trainable_variables))
        
        return total_loss, cfd_cost, residual_loss, tv_loss, entropy_term, variance_term
    
    def train(self, inputs, X, Y, layout_mask, bc_mask, bc_u, bc_v, epochs=200, verbose=True):
        """Train the router."""
        history = {
            'total_loss': [], 'cfd_cost': [], 'residual_loss': [],
            'tv_loss': [], 'entropy_term': [], 'variance_term': []
        }
        
        X_tf = tf.constant(X, dtype=tf.float32)
        Y_tf = tf.constant(Y, dtype=tf.float32)
        layout_tf = tf.constant(layout_mask, dtype=tf.float32)
        bc_mask_tf = tf.constant(bc_mask, dtype=tf.float32)
        bc_u_tf = tf.constant(bc_u, dtype=tf.float32)
        bc_v_tf = tf.constant(bc_v, dtype=tf.float32)
        inputs_tf = tf.constant(inputs, dtype=tf.float32)
        
        for epoch in range(epochs):
            total_loss, cfd_cost, residual_loss, tv_loss, entropy_term, variance_term = \
                self.train_step(inputs_tf, X_tf, Y_tf, layout_tf, bc_mask_tf, bc_u_tf, bc_v_tf)
            
            history['total_loss'].append(float(total_loss))
            history['cfd_cost'].append(float(cfd_cost))
            history['residual_loss'].append(float(residual_loss))
            history['tv_loss'].append(float(tv_loss))
            history['entropy_term'].append(float(entropy_term))
            history['variance_term'].append(float(variance_term))
            
            if verbose and (epoch + 1) % 10 == 0:
                r = self.router(inputs_tf, training=False)
                r_np = r.numpy().squeeze()
                cfd_frac = np.sum(r_np * layout_mask) / np.sum(layout_mask) * 100
                
                print(f"Epoch {epoch+1:4d} | Loss: {float(total_loss):.4f} | "
                      f"CFD: {float(cfd_cost):.4f} | Res: {float(residual_loss):.4f} | "
                      f"TV: {float(tv_loss):.4f} | Ent: {float(entropy_term):.4f} | "
                      f"Var: {float(variance_term):.4f} | CFD%: {cfd_frac:.1f}%")
        
        return history
    
    def predict(self, inputs, threshold=0.5):
        """Get router predictions and binary mask."""
        r = self.router(inputs, training=False)
        r = r.numpy().squeeze()
        mask = (r >= threshold).astype(np.float32)
        return r, mask


def compute_pinn_uv_from_psi(pinn_model, xy):
    """Compute u, v from stream function."""
    xy_tf = tf.constant(xy, dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(xy_tf)
        psi_p = pinn_model(xy_tf, training=False)
        psi = psi_p[:, 0]
    grad_psi = tape.gradient(psi, xy_tf)
    u = grad_psi[:, 1].numpy()   # ∂ψ/∂y
    v = -grad_psi[:, 0].numpy()  # -∂ψ/∂x
    return u, v


def main():
    parser = argparse.ArgumentParser(
        description='Train CNN router for hybrid PINN-CFD cavity flow simulations'
    )
    
    # Model paths
    parser.add_argument('--model-path', type=str, 
                        default='./models/pinn_cavity_flow.h5',
                        help='Path to pre-trained PINN model')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save outputs')
    parser.add_argument('--output-base-dir', type=str, default='./router_output_cavity',
                        help='Base directory for outputs')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='CFD cost coefficient (higher = less CFD)')
    parser.add_argument('--lambda-tv', type=float, default=0.01,
                        help='Total variation regularization weight')
    parser.add_argument('--lambda-entropy', type=float, default=0.1,
                        help='Entropy regularization weight')
    parser.add_argument('--lambda-variance', type=float, default=0.05,
                        help='Variance regularization weight')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping norm')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Sigmoid temperature')
    
    # Residual weights
    parser.add_argument('--weight-continuity', type=float, default=1.0,
                        help='Weight for continuity residual')
    parser.add_argument('--weight-momentum', type=float, default=1.0,
                        help='Weight for momentum residual')
    parser.add_argument('--weight-bc-local', type=float, default=2.0,
                        help='Weight for local BC error')
    parser.add_argument('--weight-bc-propagated', type=float, default=1.5,
                        help='Weight for BC error propagation')
    
    # Domain parameters (cavity is square)
    parser.add_argument('--N', type=int, default=100,
                        help='Grid size (N x N)')
    parser.add_argument('--x-min', type=float, default=0.0,
                        help='Domain x minimum')
    parser.add_argument('--x-max', type=float, default=1.0,
                        help='Domain x maximum')
    parser.add_argument('--y-min', type=float, default=0.0,
                        help='Domain y minimum')
    parser.add_argument('--y-max', type=float, default=1.0,
                        help='Domain y maximum')
    parser.add_argument('--lid-velocity', type=float, default=1.0,
                        help='Lid velocity')
    
    # Physical parameters
    parser.add_argument('--nu', type=float, default=0.01,
                        help='Kinematic viscosity')
    parser.add_argument('--rho', type=float, default=1.0,
                        help='Fluid density')
    
    # Router architecture
    parser.add_argument('--base-filters', type=int, default=32,
                        help='Base number of filters in router CNN')
    
    # Inference
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Threshold for binary mask')
    
    args = parser.parse_args()
    
    # Create output directory
    if args.output_dir is None:
        beta_str = f"beta_{args.beta:.4f}".rstrip('0').rstrip('.')
        args.output_dir = os.path.join(args.output_base_dir, beta_str)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 60)
    print("CNN ROUTER TRAINING FOR HYBRID PINN-CFD (CAVITY FLOW)")
    print("=" * 60)
    
    # =========================================================================
    # Step 1: Load PINN model
    # =========================================================================
    print("\n[Step 1] Loading PINN model...")
    print(f"  Model path: {args.model_path}")
    
    # Build cavity network architecture
    network = CavityNetwork()
    pinn_model = network.build(
        num_inputs=2, 
        layers=[32, 16, 16, 32], 
        activation='swish', 
        num_outputs=2  # (psi, p)
    )
    
    try:
        pinn_model.load_weights(args.model_path)
        print("  ✓ PINN model loaded successfully")
    except Exception as e:
        print(f"  ✗ Failed to load PINN model: {e}")
        return
    
    # Test PINN model
    test_xy = np.array([[0.5, 0.5], [0.25, 0.75]])
    test_output = pinn_model.predict(test_xy, verbose=0)
    print(f"  Test output shape: {test_output.shape}")
    print(f"  Test output: psi={test_output[0,0]:.4f}, p={test_output[0,1]:.4f}")
    
    # =========================================================================
    # Step 2: Create domain setup
    # =========================================================================
    print("\n[Step 2] Creating domain setup...")
    
    X, Y, layout, bc_mask, bc_u, bc_v, bc_p = create_cavity_setup(
        N=args.N,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
        lid_velocity=args.lid_velocity
    )
    
    print(f"  Grid: {args.N} × {args.N}")
    print(f"  Domain: x=[{args.x_min}, {args.x_max}], y=[{args.y_min}, {args.y_max}]")
    print(f"  Lid velocity: {args.lid_velocity}")
    print(f"  Fluid points: {np.sum(layout):.0f} / {layout.size} ({100*np.mean(layout):.1f}%)")
    print(f"  BC points: {np.sum(bc_mask):.0f}")
    
    # =========================================================================
    # Step 2b: Compute PINN predictions for router input
    # =========================================================================
    print("\n[Step 2b] Computing PINN predictions for router input...")
    
    # Flatten coordinates for PINN
    xy_flat = np.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)
    
    # Get velocities from stream function
    u_flat, v_flat = compute_pinn_uv_from_psi(pinn_model, xy_flat)
    
    # Get pressure
    psi_p = pinn_model.predict(xy_flat, batch_size=len(xy_flat), verbose=0)
    
    # Reshape to grid
    pinn_u = u_flat.reshape(X.shape).astype(np.float32)
    pinn_v = v_flat.reshape(X.shape).astype(np.float32)
    pinn_p = psi_p[:, 1].reshape(X.shape).astype(np.float32)
    
    # Mask out (no obstacles in cavity, but apply anyway)
    pinn_u = pinn_u * layout
    pinn_v = pinn_v * layout
    pinn_p = pinn_p * layout
    
    print(f"  PINN u range: [{pinn_u.min():.4f}, {pinn_u.max():.4f}]")
    print(f"  PINN v range: [{pinn_v.min():.4f}, {pinn_v.max():.4f}]")
    print(f"  PINN p range: [{pinn_p.min():.4f}, {pinn_p.max():.4f}]")
    
    # Create router input tensor
    inputs = create_router_input(layout, bc_mask, bc_u, bc_v, bc_p,
                                  pinn_u, pinn_v, pinn_p)
    print(f"  Router input shape: {inputs.shape}")
    
    # =========================================================================
    # Step 3: Initialize router
    # =========================================================================
    print("\n[Step 3] Initializing router CNN...")
    
    router = RouterCNN(base_filters=args.base_filters, temperature=args.temperature)
    
    # Build the model
    _ = router(inputs)
    print(f"  Router parameters: {router.count_params():,}")
    print(f"  Temperature: {args.temperature}")
    
    # =========================================================================
    # Step 4: Initialize trainer
    # =========================================================================
    print("\n[Step 4] Initializing trainer...")
    
    residual_weights = {
        'continuity': args.weight_continuity,
        'momentum': args.weight_momentum,
        'bc_local': args.weight_bc_local,
        'bc_propagated': args.weight_bc_propagated
    }
    
    trainer = CavityRouterTrainer(
        router=router,
        pinn_model=pinn_model,
        beta=args.beta,
        lambda_tv=args.lambda_tv,
        lambda_entropy=args.lambda_entropy,
        lambda_variance=args.lambda_variance,
        grad_clip_norm=args.grad_clip if args.grad_clip > 0 else None,
        residual_weights=residual_weights,
        nu=args.nu,
        rho=args.rho,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max)
    )
    trainer.optimizer.learning_rate.assign(args.lr)
    
    print(f"  β (CFD cost): {args.beta}")
    print(f"  λ_tv (TV reg): {args.lambda_tv}")
    print(f"  λ_entropy: {args.lambda_entropy}")
    print(f"  λ_variance: {args.lambda_variance}")
    print(f"  Grad clip: {args.grad_clip if args.grad_clip > 0 else 'disabled'}")
    print(f"  Learning rate: {args.lr}")
    
    # =========================================================================
    # Step 5: Train router
    # =========================================================================
    print("\n[Step 5] Training router...")
    print("-" * 50)
    
    start_time = datetime.now()
    
    history = trainer.train(
        inputs=inputs,
        X=X,
        Y=Y,
        layout_mask=layout,
        bc_mask=bc_mask,
        bc_u=bc_u,
        bc_v=bc_v,
        epochs=args.epochs,
        verbose=True
    )
    
    training_time = (datetime.now() - start_time).total_seconds()
    print("-" * 50)
    print(f"  Training completed in {training_time:.1f} seconds")
    
    # =========================================================================
    # Step 6: Get final predictions
    # =========================================================================
    print("\n[Step 6] Generating predictions...")
    
    r, mask = trainer.predict(inputs, threshold=args.threshold)
    
    cfd_fraction = np.sum(mask * layout) / np.sum(layout) * 100
    pinn_fraction = 100 - cfd_fraction
    
    print(f"  CFD region: {cfd_fraction:.1f}%")
    print(f"  PINN region: {pinn_fraction:.1f}%")
    
    # =========================================================================
    # Step 7: Save results
    # =========================================================================
    print("\n[Step 7] Saving results...")
    
    # Save router weights
    router_path = os.path.join(args.output_dir, 'router.weights.h5')
    router.save_weights(router_path)
    print(f"  ✓ Saved router weights to {router_path}")
    
    # Save training history
    history_path = os.path.join(args.output_dir, 'training_history.npz')
    np.savez(history_path, **history)
    print(f"  ✓ Saved training history to {history_path}")
    
    # Save predictions
    predictions_path = os.path.join(args.output_dir, 'predictions.npz')
    np.savez(predictions_path, 
             router_output=r, 
             mask=mask,
             X=X, Y=Y, 
             layout=layout)
    print(f"  ✓ Saved predictions to {predictions_path}")
    
    # Save configuration
    config_path = os.path.join(args.output_dir, 'config.txt')
    with open(config_path, 'w') as f:
        f.write("# Cavity Router Training Configuration\n")
        f.write(f"python train_router_cavity.py \\\n")
        f.write(f"    --model-path {args.model_path} \\\n")
        f.write(f"    --epochs {args.epochs} \\\n")
        f.write(f"    --beta {args.beta} \\\n")
        f.write(f"    --lambda-tv {args.lambda_tv} \\\n")
        f.write(f"    --lr {args.lr} \\\n")
        f.write(f"    --weight-continuity {args.weight_continuity} \\\n")
        f.write(f"    --weight-momentum {args.weight_momentum} \\\n")
        f.write(f"    --N {args.N} \\\n")
        f.write(f"    --threshold {args.threshold}\n")
        f.write(f"\n# Results:\n")
        f.write(f"# CFD region: {cfd_fraction:.1f}%\n")
        f.write(f"# PINN region: {pinn_fraction:.1f}%\n")
        f.write(f"# Training time: {training_time:.1f}s\n")
    print(f"  ✓ Saved config to {config_path}")
    
    # =========================================================================
    # Step 8: Visualize results
    # =========================================================================
    print("\n[Step 8] Generating visualizations...")
    
    # Plot router output
    router_plot_path = os.path.join(args.output_dir, 'router_output.png')
    plot_router_output(
        r, X, Y, layout,
        title=f'Trained Router - Cavity (β={args.beta}, λ={args.lambda_tv})',
        save_path=router_plot_path,
        show_circle=None  # No cylinder in cavity
    )
    
    # Plot training history
    history_plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history, save_path=history_plot_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - router.weights.h5: Trained router model")
    print(f"  - predictions.npz: Router output and mask")
    print(f"  - training_history.npz: Loss history")
    print(f"  - router_output.png: Visualization")
    print(f"  - training_history.png: Loss curves")
    
    return router, trainer, history


if __name__ == "__main__":
    main()
