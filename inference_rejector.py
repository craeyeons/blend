"""
Inference script for the trained Rejector Network.

This script loads a trained rejector model and uses it to predict
optimal CFD/PINN masks for new simulations.

Usage:
    python inference_rejector.py --Re 150.0
    python inference_rejector.py --Re 150.0 --threshold 0.3 --visualize
"""

import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from lib.rejector_network import RejectorNetwork
from lib.cylinder_flow import (
    CylinderFlowSimulation,
    CylinderFlowHybridSimulation,
)
from lib.plotting import plot_solution, plot_hybrid_solution
from cylinder_network import Network as CylinderNetwork


class RejectorInference:
    """
    Inference class for using trained rejector to predict CFD/PINN masks.
    """
    
    def __init__(self, model_dir='./rejector_output', pinn_model_dir='./models'):
        """
        Initialize inference with trained rejector.
        
        Parameters:
        -----------
        model_dir : str
            Directory containing rejector model and config
        pinn_model_dir : str
            Directory containing PINN models
        """
        self.model_dir = model_dir
        self.pinn_model_dir = pinn_model_dir
        
        # Load config
        config_path = os.path.join(model_dir, 'rejector_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Default config
            self.config = {
                'grid_size': 100,
                'x_domain': [0, 2],
                'y_domain': [0, 1],
                'cylinder_center': [0.5, 0.5],
                'cylinder_radius': 0.1,
            }
        
        # Build and load rejector
        self.rejector = RejectorNetwork()
        self.rejector.build(num_inputs=3, layers=[64, 64, 64, 32])
        
        weights_path = os.path.join(model_dir, 'rejector_model.weights.h5')
        if os.path.exists(weights_path):
            self.rejector.load(weights_path)
            print(f"Loaded rejector model from {weights_path}")
        else:
            raise FileNotFoundError(f"No model found at {weights_path}")
        
        # Setup grid from config
        self._setup_grid()
    
    def _setup_grid(self):
        """Setup computational grid from config."""
        self.grid_size = self.config['grid_size']
        self.x_domain = tuple(self.config['x_domain'])
        self.y_domain = tuple(self.config['y_domain'])
        self.cylinder_center = tuple(self.config['cylinder_center'])
        self.cylinder_radius = self.config['cylinder_radius']
        
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
        
        # Cylinder mask
        Cx, Cy = self.cylinder_center
        self.cylinder_mask = np.sqrt(
            (self.X - Cx)**2 + (self.Y - Cy)**2
        ) <= self.cylinder_radius
    
    def predict_mask(self, Re, threshold=0.5, enforce_boundary=True, 
                     boundary_width=None):
        """
        Predict CFD/PINN mask for given Reynolds number.
        
        Parameters:
        -----------
        Re : float
            Reynolds number
        threshold : float
            Threshold for converting probability to binary (default 0.5)
        enforce_boundary : bool
            Whether to force boundaries to be CFD
        boundary_width : int, optional
            Width of boundary region (default: grid_size // 20)
            
        Returns:
        --------
        mask : ndarray
            Binary mask (1 = CFD, 0 = PINN)
        probability : ndarray
            Raw probability field
        """
        if boundary_width is None:
            boundary_width = max(5, self.grid_size // 20)
        
        # Get mask from rejector
        mask = self.rejector.predict_mask_grid(
            self.X, self.Y, Re, 
            threshold=threshold,
            enforce_boundary=enforce_boundary,
            boundary_width=boundary_width
        )
        
        # Get probability field
        Re_norm = Re / 1000.0
        xy_re = np.column_stack([
            self.X.flatten(),
            self.Y.flatten(),
            np.full(self.X.size, Re_norm)
        ])
        probability = self.rejector.predict_probability(xy_re).reshape(self.Ny, self.Nx)
        
        # Enforce cylinder region
        mask[self.cylinder_mask] = 1
        
        return mask, probability
    
    def _load_pinn_model(self, Re):
        """Load PINN model for given Re."""
        # Try different naming conventions
        paths_to_try = [
            os.path.join(self.pinn_model_dir, f'pinn_cylinder_{Re}.h5'),
            os.path.join(self.pinn_model_dir, f'pinn_cylinder_{int(Re)}.h5'),
            os.path.join(self.pinn_model_dir, f'pinn_cylinder_{Re:.1f}.h5'),
        ]
        
        for path in paths_to_try:
            if os.path.exists(path):
                print(f"Loading PINN model from {path}")
                network_builder = CylinderNetwork()
                network = network_builder.build(
                    num_inputs=2, layers=[48, 48, 48, 48],
                    activation='tanh', num_outputs=3
                )
                network.load_weights(path)
                
                def uv_func(net, xy):
                    uvp = net.predict(xy, batch_size=len(xy))
                    return uvp[..., 0], uvp[..., 1]
                
                return network, uv_func
        
        # Try to find closest available model
        available = []
        if os.path.exists(self.pinn_model_dir):
            for f in os.listdir(self.pinn_model_dir):
                if f.startswith('pinn_cylinder_') and f.endswith('.h5'):
                    try:
                        re_str = f.replace('pinn_cylinder_', '').replace('.h5', '')
                        available.append(float(re_str))
                    except:
                        pass
        
        if available:
            # Find closest
            closest = min(available, key=lambda x: abs(x - Re))
            print(f"Using closest available model: Re={closest} (requested Re={Re})")
            return self._load_pinn_model(closest)
        
        raise FileNotFoundError(f"No PINN model found for Re={Re}")
    
    def run_simulation(self, Re, threshold=0.5, max_iter=50000, 
                       compare_with_cfd=False, save_results=True, 
                       output_dir='./results'):
        """
        Run full simulation using rejector-predicted mask.
        
        Parameters:
        -----------
        Re : float
            Reynolds number
        threshold : float
            Mask threshold
        max_iter : int
            Maximum iterations
        compare_with_cfd : bool
            Whether to also run pure CFD for comparison
        save_results : bool
            Whether to save result plots
        output_dir : str
            Directory to save results
            
        Returns:
        --------
        dict : Results including u, v, p, mask, and optionally comparison metrics
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"Running simulation with rejector mask for Re = {Re}")
        print(f"{'='*60}")
        
        # Get predicted mask
        mask, probability = self.predict_mask(Re, threshold=threshold)
        
        cfd_fraction = np.mean(mask)
        print(f"Predicted CFD fraction: {cfd_fraction:.2%}")
        
        # Load PINN model
        network, uv_func = self._load_pinn_model(Re)
        
        # Run hybrid simulation
        print("\nRunning hybrid simulation...")
        hybrid_sim = CylinderFlowHybridSimulation(
            network=network,
            uv_func=uv_func,
            mask=mask,
            Re=Re,
            N=self.grid_size,
            max_iter=max_iter,
            tol=1e-6,
            x_domain=self.x_domain,
            y_domain=self.y_domain,
            cylinder_center=self.cylinder_center,
            cylinder_radius=self.cylinder_radius,
        )
        u_hybrid, v_hybrid, p_hybrid = hybrid_sim.solve()
        
        results = {
            'Re': Re,
            'threshold': threshold,
            'mask': mask,
            'probability': probability,
            'cfd_fraction': cfd_fraction,
            'u': u_hybrid,
            'v': v_hybrid,
            'p': p_hybrid,
        }
        
        # Compare with pure CFD if requested
        if compare_with_cfd:
            print("\nRunning pure CFD for comparison...")
            cfd_sim = CylinderFlowSimulation(
                Re=Re,
                N=self.grid_size,
                max_iter=max_iter,
                tol=1e-6,
                x_domain=self.x_domain,
                y_domain=self.y_domain,
                cylinder_center=self.cylinder_center,
                cylinder_radius=self.cylinder_radius,
            )
            u_cfd, v_cfd, p_cfd = cfd_sim.solve()
            
            # Compute errors
            vel_hybrid = np.sqrt(u_hybrid**2 + v_hybrid**2)
            vel_cfd = np.sqrt(u_cfd**2 + v_cfd**2)
            error = np.abs(vel_hybrid - vel_cfd)
            
            results['u_cfd'] = u_cfd
            results['v_cfd'] = v_cfd
            results['p_cfd'] = p_cfd
            results['error'] = error
            results['max_error'] = np.max(error)
            results['mean_error'] = np.mean(error)
            results['l2_error'] = np.sqrt(np.mean(error**2))
            
            print(f"\nError metrics:")
            print(f"  Max error: {results['max_error']:.6f}")
            print(f"  Mean error: {results['mean_error']:.6f}")
            print(f"  L2 error: {results['l2_error']:.6f}")
        
        # Save visualizations
        if save_results:
            self._save_visualizations(results, output_dir)
        
        return results
    
    def _save_visualizations(self, results, output_dir):
        """Save visualization plots."""
        Re = results['Re']
        mask = results['mask']
        probability = results['probability']
        
        # Plot probability field
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Probability
        im0 = axes[0].contourf(self.X, self.Y, probability, levels=50, cmap='RdYlBu_r')
        axes[0].set_title(f'CFD Probability (Re={Re})')
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('y')
        plt.colorbar(im0, ax=axes[0])
        
        # Add cylinder
        circle0 = plt.Circle(self.cylinder_center, self.cylinder_radius, 
                            color='gray', fill=True)
        axes[0].add_patch(circle0)
        axes[0].set_aspect('equal')
        
        # Binary mask
        im1 = axes[1].contourf(self.X, self.Y, mask, levels=[0, 0.5, 1], 
                               colors=['blue', 'red'], alpha=0.7)
        axes[1].set_title(f'CFD/PINN Mask (Red=CFD, Blue=PINN)')
        axes[1].set_xlabel('x')
        axes[1].set_ylabel('y')
        
        circle1 = plt.Circle(self.cylinder_center, self.cylinder_radius, 
                            color='gray', fill=True)
        axes[1].add_patch(circle1)
        axes[1].set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'rejector_mask_Re{int(Re)}.png'), 
                    dpi=150, bbox_inches='tight')
        plt.close()
        
        # Plot solution
        show_circle = (*self.cylinder_center, self.cylinder_radius)
        plot_hybrid_solution(
            results['u'], results['v'], results['p'], mask,
            x=self.X, y=self.Y,
            save_path=os.path.join(output_dir, f'rejector_solution_Re{int(Re)}.png'),
            show_circle=show_circle
        )
        
        # If comparison available, plot error
        if 'error' in results:
            fig, ax = plt.subplots(figsize=(12, 5))
            cf = ax.contourf(self.X, self.Y, results['error'], levels=50, cmap='hot')
            ax.set_title(f'Error vs Pure CFD (Re={Re})')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            plt.colorbar(cf, ax=ax)
            
            circle = plt.Circle(self.cylinder_center, self.cylinder_radius, 
                               color='white', fill=True)
            ax.add_patch(circle)
            ax.set_aspect('equal')
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'rejector_error_Re{int(Re)}.png'),
                        dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Saved visualizations to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Run inference with trained rejector model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict mask for Re=150
  python inference_rejector.py --Re 150.0
  
  # Run full simulation with comparison
  python inference_rejector.py --Re 150.0 --compare-with-cfd
  
  # Use different threshold
  python inference_rejector.py --Re 150.0 --threshold 0.3
        """
    )
    
    parser.add_argument('--Re', type=float, required=True,
                       help='Reynolds number')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for mask (default: 0.5)')
    parser.add_argument('--model-dir', type=str, default='./rejector_output',
                       help='Directory containing rejector model')
    parser.add_argument('--pinn-model-dir', type=str, default='./models',
                       help='Directory containing PINN models')
    parser.add_argument('--output-dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--max-iter', type=int, default=50000,
                       help='Maximum iterations')
    parser.add_argument('--compare-with-cfd', action='store_true',
                       help='Compare with pure CFD solution')
    parser.add_argument('--visualize-only', action='store_true',
                       help='Only predict and visualize mask, no simulation')
    
    args = parser.parse_args()
    
    # Create inference object
    inference = RejectorInference(
        model_dir=args.model_dir,
        pinn_model_dir=args.pinn_model_dir
    )
    
    if args.visualize_only:
        # Just predict and visualize mask
        mask, probability = inference.predict_mask(args.Re, threshold=args.threshold)
        
        print(f"\nMask Statistics for Re={args.Re}:")
        print(f"  CFD fraction: {np.mean(mask):.2%}")
        print(f"  Mean probability: {np.mean(probability):.4f}")
        
        # Quick visualization
        os.makedirs(args.output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        im0 = axes[0].contourf(inference.X, inference.Y, probability, 
                               levels=50, cmap='RdYlBu_r')
        axes[0].set_title(f'CFD Probability (Re={args.Re})')
        plt.colorbar(im0, ax=axes[0])
        axes[0].set_aspect('equal')
        
        im1 = axes[1].contourf(inference.X, inference.Y, mask, 
                               levels=[0, 0.5, 1], colors=['blue', 'red'], alpha=0.7)
        axes[1].set_title(f'Binary Mask (threshold={args.threshold})')
        axes[1].set_aspect('equal')
        
        # Add cylinders
        for ax in axes:
            circle = plt.Circle(inference.cylinder_center, inference.cylinder_radius,
                               color='gray', fill=True)
            ax.add_patch(circle)
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f'rejector_mask_Re{int(args.Re)}.png'),
                    dpi=150, bbox_inches='tight')
        plt.show()
        
    else:
        # Run full simulation
        results = inference.run_simulation(
            Re=args.Re,
            threshold=args.threshold,
            max_iter=args.max_iter,
            compare_with_cfd=args.compare_with_cfd,
            save_results=True,
            output_dir=args.output_dir
        )
        
        print("\n" + "="*60)
        print("SIMULATION COMPLETE")
        print("="*60)
        print(f"Re: {results['Re']}")
        print(f"CFD fraction: {results['cfd_fraction']:.2%}")
        
        if 'l2_error' in results:
            print(f"L2 Error vs CFD: {results['l2_error']:.6f}")


if __name__ == "__main__":
    main()
