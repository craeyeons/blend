"""
Training script for the Rejector Network.

This script trains a neural network to learn optimal CFD/PINN region partitioning
by iteratively solving hybrid problems and minimizing a cost-accuracy tradeoff.

Usage:
    python train_rejector.py --re-range 0.1 1000 --num-samples 12
    python train_rejector.py --re-values 1.0 100.0 500.0 1000.0
"""

import argparse
import os
import json
import numpy as np
import tensorflow as tf
from datetime import datetime

# Set up paths
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.rejector_network import (
    RejectorNetwork, 
    compute_weighted_loss,
    create_rejector_training_data
)
from lib.cylinder_flow import (
    CylinderFlowSimulation,
    CylinderFlowHybridSimulation,
)
from cylinder_network import Network as CylinderNetwork


# Default Reynolds number samples covering key flow regimes
DEFAULT_RE_VALUES = [
    0.1,    # Creeping flow
    1.0,    # Low Re laminar
    5.0,    # Laminar
    10.0,   # Laminar
    20.0,   # Laminar
    40.0,   # Near vortex shedding onset
    100.0,  # Vortex shedding (steady approximation)
    200.0,  # Developed vortex shedding
    500.0,  # Unsteady laminar
    750.0,  # Transition regime
    1000.0, # Higher Re
]


class RejectorTrainer:
    """
    Trainer class for the rejector network.
    
    Training process:
    1. For each Re value:
       a. Load/train PINN model for that Re
       b. Compute ground truth CFD solution (full domain)
       c. Initialize rejector mask (e.g., all PINN or random)
       d. Iterate:
          - Solve hybrid with current mask
          - Compute error = |hybrid - ground_truth|
          - Update rejector based on error and cost
          - Repeat until convergence
    2. Aggregate training data across Re values
    3. Train final rejector model
    """
    
    def __init__(self, 
                 grid_size=100,
                 x_domain=(0, 2),
                 y_domain=(0, 1),
                 cylinder_center=(0.5, 0.5),
                 cylinder_radius=0.1,
                 inlet_velocity=1.0,
                 pinn_model_dir='./models',
                 output_dir='./rejector_output',
                 c_cfd=1.0,
                 c_pinn=0.01,
                 accuracy_weight=1.0,
                 cost_weight=0.1):
        """
        Initialize the rejector trainer.
        
        Parameters:
        -----------
        grid_size : int
            Grid size for simulations
        x_domain, y_domain : tuple
            Domain bounds
        cylinder_center : tuple
            Cylinder center coordinates
        cylinder_radius : float
            Cylinder radius
        inlet_velocity : float
            Inlet velocity
        pinn_model_dir : str
            Directory containing PINN models
        output_dir : str
            Directory to save outputs
        c_cfd, c_pinn : float
            Computational costs for CFD and PINN
        accuracy_weight, cost_weight : float
            Loss function weights
        """
        self.grid_size = grid_size
        self.x_domain = x_domain
        self.y_domain = y_domain
        self.cylinder_center = cylinder_center
        self.cylinder_radius = cylinder_radius
        self.inlet_velocity = inlet_velocity
        self.pinn_model_dir = pinn_model_dir
        self.output_dir = output_dir
        
        # Cost parameters
        self.c_cfd = c_cfd
        self.c_pinn = c_pinn
        self.accuracy_weight = accuracy_weight
        self.cost_weight = cost_weight
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Setup grid
        self._setup_grid()
        
        # Initialize rejector
        self.rejector = RejectorNetwork()
        self.rejector.build(num_inputs=3, layers=[64, 64, 64, 32])
        
        # Storage for training data
        self.training_data = {
            'X': [],  # (x, y, Re)
            'errors': [],  # Error at each point
            'optimal_masks': [],  # Learned optimal masks
        }
        
        # Training history
        self.history = {
            'Re_values': [],
            'final_losses': [],
            'cfd_fractions': [],
            'training_time': None,
        }
    
    def _setup_grid(self):
        """Setup computational grid."""
        x_ini, x_f = self.x_domain
        y_ini, y_f = self.y_domain
        Lx = x_f - x_ini
        Ly = y_f - y_ini
        aspect_ratio = Lx / Ly
        
        self.Ny = self.grid_size
        self.Nx = int(self.grid_size * aspect_ratio)
        
        x = np.linspace(x_ini, x_f, self.Nx)
        y = np.linspace(y_ini, y_f, self.Ny)
        self.X, self.Y = np.meshgrid(x, y)
        self.xy = np.column_stack([self.X.flatten(), self.Y.flatten()])
        
        # Cylinder mask (points inside cylinder)
        Cx, Cy = self.cylinder_center
        self.cylinder_mask = np.sqrt(
            (self.X - Cx)**2 + (self.Y - Cy)**2
        ) <= self.cylinder_radius
    
    def _get_pinn_model_path(self, Re):
        """Get path to PINN model for given Re."""
        # Try different naming conventions
        paths_to_try = [
            os.path.join(self.pinn_model_dir, f'pinn_cylinder_{Re}.h5'),
            os.path.join(self.pinn_model_dir, f'pinn_cylinder_{int(Re)}.h5'),
            os.path.join(self.pinn_model_dir, f'pinn_cylinder_{Re:.1f}.h5'),
        ]
        
        for path in paths_to_try:
            if os.path.exists(path):
                return path
        
        return None
    
    def _load_pinn_model(self, Re):
        """Load PINN model for given Reynolds number."""
        model_path = self._get_pinn_model_path(Re)
        
        if model_path is None:
            print(f"  Warning: No PINN model found for Re={Re}")
            print(f"  Available models in {self.pinn_model_dir}:")
            if os.path.exists(self.pinn_model_dir):
                for f in os.listdir(self.pinn_model_dir):
                    print(f"    {f}")
            return None, None
        
        print(f"  Loading PINN model from {model_path}")
        
        # Build and load network
        network_builder = CylinderNetwork()
        network = network_builder.build(num_inputs=2, layers=[48, 48, 48, 48], 
                                        activation='tanh', num_outputs=3)
        network.load_weights(model_path)
        
        # UV extraction function (direct output)
        def uv_func(net, xy):
            uvp = net.predict(xy, batch_size=len(xy))
            return uvp[..., 0], uvp[..., 1]
        
        return network, uv_func
    
    def compute_ground_truth_cfd(self, Re, max_iter=50000):
        """
        Compute ground truth CFD solution for the full domain.
        
        Parameters:
        -----------
        Re : float
            Reynolds number
        max_iter : int
            Maximum iterations for CFD solver
            
        Returns:
        --------
        u, v, p : ndarray
            Velocity and pressure fields
        """
        print(f"  Computing ground truth CFD for Re={Re}...")
        
        sim = CylinderFlowSimulation(
            Re=Re,
            N=self.grid_size,
            max_iter=max_iter,
            tol=1e-6,
            x_domain=self.x_domain,
            y_domain=self.y_domain,
            cylinder_center=self.cylinder_center,
            cylinder_radius=self.cylinder_radius,
            inlet_velocity=self.inlet_velocity
        )
        
        u, v, p = sim.solve()
        return u, v, p
    
    def compute_hybrid_solution(self, Re, network, uv_func, mask, max_iter=20000):
        """
        Compute hybrid PINN-CFD solution with given mask.
        
        Parameters:
        -----------
        Re : float
            Reynolds number
        network : tf.keras.Model
            PINN network
        uv_func : callable
            Function to extract u, v from network
        mask : ndarray
            Binary mask (1 = CFD, 0 = PINN)
        max_iter : int
            Maximum iterations
            
        Returns:
        --------
        u, v, p : ndarray
            Hybrid solution fields
        """
        sim = CylinderFlowHybridSimulation(
            network=network,
            uv_func=uv_func,
            mask=mask,
            Re=Re,
            N=self.grid_size,
            max_iter=max_iter,
            tol=1e-5,
            x_domain=self.x_domain,
            y_domain=self.y_domain,
            cylinder_center=self.cylinder_center,
            cylinder_radius=self.cylinder_radius,
            inlet_velocity=self.inlet_velocity
        )
        
        u, v, p = sim.solve()
        return u, v, p
    
    def compute_error_field(self, u_hybrid, v_hybrid, u_gt, v_gt):
        """
        Compute error field between hybrid and ground truth.
        
        Parameters:
        -----------
        u_hybrid, v_hybrid : ndarray
            Hybrid solution velocity fields
        u_gt, v_gt : ndarray
            Ground truth velocity fields
            
        Returns:
        --------
        error : ndarray
            Error magnitude at each point
        """
        # Velocity magnitude error
        vel_hybrid = np.sqrt(u_hybrid**2 + v_hybrid**2)
        vel_gt = np.sqrt(u_gt**2 + v_gt**2)
        
        # Normalized error
        error = np.abs(vel_hybrid - vel_gt)
        
        # Also consider component-wise errors
        error_u = np.abs(u_hybrid - u_gt)
        error_v = np.abs(v_hybrid - v_gt)
        
        # Combined error (L2 norm)
        error = np.sqrt(error_u**2 + error_v**2)
        
        return error
    
    def create_initial_mask(self, strategy='boundary_only'):
        """
        Create initial mask for training iteration.
        
        Parameters:
        -----------
        strategy : str
            'boundary_only': CFD only at boundaries
            'all_pinn': Start with all PINN
            'random': Random initialization
            
        Returns:
        --------
        mask : ndarray
            Initial binary mask
        """
        mask = np.zeros((self.Ny, self.Nx), dtype=np.int32)
        
        if strategy in ['boundary_only', 'all_pinn']:
            # Only boundaries use CFD
            boundary_width = max(5, self.grid_size // 20)
            mask[:boundary_width, :] = 1   # Bottom
            mask[-boundary_width:, :] = 1  # Top
            mask[:, :boundary_width] = 1   # Left
            mask[:, -boundary_width:] = 1  # Right
            
        elif strategy == 'random':
            mask = np.random.randint(0, 2, (self.Ny, self.Nx)).astype(np.int32)
            # Always enforce boundaries
            boundary_width = max(5, self.grid_size // 20)
            mask[:boundary_width, :] = 1
            mask[-boundary_width:, :] = 1
            mask[:, :boundary_width] = 1
            mask[:, -boundary_width:] = 1
        
        # Points inside cylinder are always CFD (will be set to 0 velocity anyway)
        mask[self.cylinder_mask] = 1
        
        return mask
    
    def update_mask_from_error(self, error, current_mask, learning_rate=0.1,
                                error_threshold=0.05):
        """
        Update mask based on error field using soft update.
        
        High error regions should switch to CFD.
        
        Parameters:
        -----------
        error : ndarray
            Error field
        current_mask : ndarray
            Current binary mask
        learning_rate : float
            How aggressively to update
        error_threshold : float
            Error threshold for switching to CFD
            
        Returns:
        --------
        new_mask : ndarray
            Updated binary mask
        """
        # Normalize error
        error_norm = error / (np.max(error) + 1e-10)
        
        # Points with high error should become CFD
        # Use sigmoid-like soft update
        cfd_probability = 1 / (1 + np.exp(-10 * (error_norm - error_threshold)))
        
        # Blend with current mask
        new_mask_prob = current_mask * (1 - learning_rate) + cfd_probability * learning_rate
        
        # Threshold to binary
        new_mask = (new_mask_prob > 0.5).astype(np.int32)
        
        # Enforce boundary conditions
        boundary_width = max(5, self.grid_size // 20)
        new_mask[:boundary_width, :] = 1
        new_mask[-boundary_width:, :] = 1
        new_mask[:, :boundary_width] = 1
        new_mask[:, -boundary_width:] = 1
        new_mask[self.cylinder_mask] = 1
        
        return new_mask
    
    def train_single_re(self, Re, max_outer_iter=10, max_cfd_iter=20000,
                        convergence_tol=1e-4):
        """
        Train rejector for a single Reynolds number.
        
        Parameters:
        -----------
        Re : float
            Reynolds number
        max_outer_iter : int
            Maximum outer iterations (mask updates)
        max_cfd_iter : int
            Maximum CFD solver iterations per hybrid solve
        convergence_tol : float
            Convergence tolerance for loss
            
        Returns:
        --------
        dict : Training results for this Re
        """
        print(f"\n{'='*60}")
        print(f"Training rejector for Re = {Re}")
        print(f"{'='*60}")
        
        # Load PINN model
        network, uv_func = self._load_pinn_model(Re)
        if network is None:
            print(f"  Skipping Re={Re} - no PINN model available")
            return None
        
        # Compute ground truth CFD (full domain)
        u_gt, v_gt, p_gt = self.compute_ground_truth_cfd(Re, max_iter=max_cfd_iter)
        
        # Initialize mask
        mask = self.create_initial_mask(strategy='boundary_only')
        
        # Training loop
        losses = []
        prev_loss = float('inf')
        
        for outer_iter in range(max_outer_iter):
            print(f"\n  Outer iteration {outer_iter + 1}/{max_outer_iter}")
            
            # Compute hybrid solution with current mask
            try:
                u_hybrid, v_hybrid, p_hybrid = self.compute_hybrid_solution(
                    Re, network, uv_func, mask, max_iter=max_cfd_iter
                )
            except Exception as e:
                print(f"  Error in hybrid solve: {e}")
                break
            
            # Compute error field
            error = self.compute_error_field(u_hybrid, v_hybrid, u_gt, v_gt)
            
            # Compute loss
            cfd_fraction = np.mean(mask)
            probs = mask.flatten().astype(float)
            loss, acc_loss, cost_loss = compute_weighted_loss(
                error.flatten(), probs,
                c_cfd=self.c_cfd, c_pinn=self.c_pinn,
                accuracy_weight=self.accuracy_weight,
                cost_weight=self.cost_weight
            )
            losses.append(loss)
            
            print(f"    Loss: {loss:.6f} (accuracy: {acc_loss:.6f}, cost: {cost_loss:.6f})")
            print(f"    CFD fraction: {cfd_fraction:.2%}")
            print(f"    Max error: {np.max(error):.6f}, Mean error: {np.mean(error):.6f}")
            
            # Check convergence
            if abs(loss - prev_loss) < convergence_tol:
                print(f"    Converged at iteration {outer_iter + 1}")
                break
            prev_loss = loss
            
            # Update mask based on error
            if outer_iter < max_outer_iter - 1:  # Don't update on last iteration
                mask = self.update_mask_from_error(
                    error, mask, 
                    learning_rate=0.3,
                    error_threshold=0.05 * np.max(error)
                )
        
        # Store training data
        Re_normalized = Re / 1000.0
        X = np.column_stack([
            self.xy[:, 0],
            self.xy[:, 1],
            np.full(len(self.xy), Re_normalized)
        ])
        
        self.training_data['X'].append(X)
        self.training_data['errors'].append(error.flatten())
        self.training_data['optimal_masks'].append(mask.flatten())
        
        # Store history
        self.history['Re_values'].append(Re)
        self.history['final_losses'].append(losses[-1] if losses else None)
        self.history['cfd_fractions'].append(np.mean(mask))
        
        return {
            'Re': Re,
            'final_loss': losses[-1] if losses else None,
            'cfd_fraction': np.mean(mask),
            'iterations': len(losses),
            'error_field': error,
            'optimal_mask': mask,
            'losses': losses,
        }
    
    def train_all(self, Re_values, **kwargs):
        """
        Train rejector across multiple Reynolds numbers.
        
        Parameters:
        -----------
        Re_values : list
            List of Reynolds numbers to train on
        **kwargs : 
            Additional arguments passed to train_single_re
        """
        start_time = datetime.now()
        
        print("\n" + "="*70)
        print("REJECTOR NETWORK TRAINING")
        print("="*70)
        print(f"Training on {len(Re_values)} Reynolds numbers: {Re_values}")
        print(f"Grid size: {self.grid_size}")
        print(f"Cost weights: c_cfd={self.c_cfd}, c_pinn={self.c_pinn}")
        print(f"Loss weights: accuracy={self.accuracy_weight}, cost={self.cost_weight}")
        
        results = []
        for Re in Re_values:
            result = self.train_single_re(Re, **kwargs)
            if result is not None:
                results.append(result)
        
        # Train the neural network on collected data
        if self.training_data['X']:
            self._train_network()
        
        # Record training time
        self.history['training_time'] = str(datetime.now() - start_time)
        
        print("\n" + "="*70)
        print("TRAINING COMPLETE")
        print("="*70)
        print(f"Total training time: {self.history['training_time']}")
        print(f"Successfully trained on {len(results)} Re values")
        
        return results
    
    def _train_network(self, epochs=100, batch_size=1024, validation_split=0.2):
        """
        Train the rejector neural network on collected data.
        """
        print("\n" + "-"*60)
        print("Training Rejector Neural Network")
        print("-"*60)
        
        # Concatenate all training data
        X = np.vstack(self.training_data['X'])
        errors = np.concatenate(self.training_data['errors'])
        masks = np.concatenate(self.training_data['optimal_masks'])
        
        print(f"Training data shape: X={X.shape}, masks={masks.shape}")
        
        # Normalize errors for training
        errors_norm = errors / (np.max(errors) + 1e-10)
        
        # Target: where error is high, we want CFD (1)
        # Use error-weighted target
        y = np.clip(errors_norm * 2, 0, 1)  # Scale and clip
        
        # Add some weight to actual optimal masks
        y = 0.7 * y + 0.3 * masks
        
        # Compile model
        self.rejector.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5
            )
        ]
        
        # Train
        history = self.rejector.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        self.rejector.history = history.history
        
        print(f"Final training loss: {history.history['loss'][-1]:.4f}")
        print(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    
    def save_model(self, filepath=None):
        """Save the trained rejector model."""
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'rejector_model.weights.h5')
        
        self.rejector.save(filepath)
        
        # Also save training history
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved training history to {history_path}")
        
        # Save config
        config = {
            'grid_size': self.grid_size,
            'x_domain': self.x_domain,
            'y_domain': self.y_domain,
            'cylinder_center': self.cylinder_center,
            'cylinder_radius': self.cylinder_radius,
            'c_cfd': self.c_cfd,
            'c_pinn': self.c_pinn,
            'accuracy_weight': self.accuracy_weight,
            'cost_weight': self.cost_weight,
        }
        config_path = os.path.join(self.output_dir, 'rejector_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Saved config to {config_path}")
        
        return filepath
    
    def load_model(self, filepath=None):
        """Load a trained rejector model."""
        if filepath is None:
            filepath = os.path.join(self.output_dir, 'rejector_model.weights.h5')
        
        self.rejector.load(filepath)
        
        # Load history if available
        history_path = os.path.join(self.output_dir, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.history = json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description='Train rejector network for CFD/PINN region selection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default Re values
  python train_rejector.py

  # Train with specific Re values
  python train_rejector.py --re-values 1.0 100.0 1000.0

  # Train with custom grid size and iterations
  python train_rejector.py --grid-size 64 --max-outer-iter 5
  
  # Adjust cost/accuracy tradeoff
  python train_rejector.py --c-cfd 1.0 --c-pinn 0.001 --accuracy-weight 2.0
        """
    )
    
    # Reynolds number arguments
    parser.add_argument('--re-values', type=float, nargs='+', default=None,
                       help='Specific Re values to train on')
    parser.add_argument('--re-min', type=float, default=0.1,
                       help='Minimum Re for auto-generated range')
    parser.add_argument('--re-max', type=float, default=1000.0,
                       help='Maximum Re for auto-generated range')
    parser.add_argument('--num-re-samples', type=int, default=12,
                       help='Number of Re samples if not specifying --re-values')
    
    # Grid parameters
    parser.add_argument('--grid-size', type=int, default=100,
                       help='Grid size for simulations')
    parser.add_argument('--x-min', type=float, default=0.0,
                       help='Domain x minimum')
    parser.add_argument('--x-max', type=float, default=2.0,
                       help='Domain x maximum')
    parser.add_argument('--y-min', type=float, default=0.0,
                       help='Domain y minimum')
    parser.add_argument('--y-max', type=float, default=1.0,
                       help='Domain y maximum')
    parser.add_argument('--cylinder-x', type=float, default=0.5,
                       help='Cylinder center x')
    parser.add_argument('--cylinder-y', type=float, default=0.5,
                       help='Cylinder center y')
    parser.add_argument('--cylinder-radius', type=float, default=0.1,
                       help='Cylinder radius')
    
    # Training parameters
    parser.add_argument('--max-outer-iter', type=int, default=10,
                       help='Maximum outer iterations per Re')
    parser.add_argument('--max-cfd-iter', type=int, default=20000,
                       help='Maximum CFD solver iterations')
    parser.add_argument('--convergence-tol', type=float, default=1e-4,
                       help='Convergence tolerance')
    
    # Cost parameters
    parser.add_argument('--c-cfd', type=float, default=1.0,
                       help='Computational cost of CFD')
    parser.add_argument('--c-pinn', type=float, default=0.01,
                       help='Computational cost of PINN')
    parser.add_argument('--accuracy-weight', type=float, default=1.0,
                       help='Weight for accuracy term in loss')
    parser.add_argument('--cost-weight', type=float, default=0.1,
                       help='Weight for cost term in loss')
    
    # Output
    parser.add_argument('--model-dir', type=str, default='./models',
                       help='Directory containing PINN models')
    parser.add_argument('--output-dir', type=str, default='./rejector_output',
                       help='Directory to save outputs')
    
    args = parser.parse_args()
    
    # Determine Re values
    if args.re_values is not None:
        Re_values = args.re_values
    else:
        # Use default values or generate log-spaced
        Re_values = DEFAULT_RE_VALUES
    
    # Filter to available models
    print("Checking available PINN models...")
    available_Re = []
    for Re in Re_values:
        paths = [
            os.path.join(args.model_dir, f'pinn_cylinder_{Re}.h5'),
            os.path.join(args.model_dir, f'pinn_cylinder_{int(Re)}.h5'),
            os.path.join(args.model_dir, f'pinn_cylinder_{Re:.1f}.h5'),
        ]
        if any(os.path.exists(p) for p in paths):
            available_Re.append(Re)
    
    if not available_Re:
        print(f"No PINN models found in {args.model_dir}")
        print("Available files:")
        if os.path.exists(args.model_dir):
            for f in os.listdir(args.model_dir):
                print(f"  {f}")
        print("\nPlease train PINN models first or specify correct model directory.")
        return
    
    print(f"Found models for Re: {available_Re}")
    
    # Create trainer
    trainer = RejectorTrainer(
        grid_size=args.grid_size,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
        cylinder_center=(args.cylinder_x, args.cylinder_y),
        cylinder_radius=args.cylinder_radius,
        pinn_model_dir=args.model_dir,
        output_dir=args.output_dir,
        c_cfd=args.c_cfd,
        c_pinn=args.c_pinn,
        accuracy_weight=args.accuracy_weight,
        cost_weight=args.cost_weight,
    )
    
    # Train
    results = trainer.train_all(
        available_Re,
        max_outer_iter=args.max_outer_iter,
        max_cfd_iter=args.max_cfd_iter,
        convergence_tol=args.convergence_tol,
    )
    
    # Save model
    trainer.save_model()
    
    print("\nTraining complete!")
    print(f"Model saved to: {args.output_dir}/rejector_model.weights.h5")


if __name__ == "__main__":
    main()
