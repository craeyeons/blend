"""
Example: Learned Router for Hybrid PINN-CFD Domain Segregation

This script demonstrates how to:
1. Train a learned router using PINN predictions and CFD ground truth
2. Use the trained router to generate smooth, connected masks
3. Run hybrid PINN-CFD simulation with learned routing

The learned router addresses limitations of threshold-based segregation:
- Produces spatially smooth, connected regions
- Learns boundary condition awareness (rejects PINN near inlet)
- Captures upstream error propagation
- Avoids patchy masks

Training data: 5 Reynolds numbers × 5 cylinder positions = 25 cases
"""

import numpy as np
import time
import os
import json
import tensorflow as tf
from lib.network import Network
from lib.router import (
    create_router, 
    RouterFeatureExtractor,
    RouterCNN,
    RouterLoss,
    RouterTrainer,
    LearnedRouter
)
from lib.cylinder_flow import (
    CylinderFlowSimulation,
    CylinderFlowLearnedRouterSimulation,
    CylinderFlowDynamicHybridSimulation
)
from lib.plotting import plot_solution, plot_hybrid_solution, plot_streamlines


# =============================================================================
# Configuration
# =============================================================================

# Reynolds numbers for training
RE_VALUES = [1.0, 10.0, 40.0, 100.0, 200.0]

# Cylinder positions (Cx, Cy) for training
CYLINDER_POSITIONS = [
    (0.5, 0.3),
    (0.5, 0.4),
    (0.5, 0.5),
    (0.5, 0.6),
    (0.5, 0.7),
]

# Domain settings
X_DOMAIN = (0.0, 2.0)
Y_DOMAIN = (0.0, 1.0)
N = 100
CYLINDER_RADIUS = 0.1
INLET_VELOCITY = 1.0

# Data directory
DATA_DIR = './data'


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_pinn_model(model_path: str) -> tf.keras.Model:
    """
    Load a PINN model from .h5 file.
    """
    network = Network().build(num_inputs=2, num_outputs=3)
    network.load_weights(model_path)
    return network


def get_pinn_predictions_from_model(network: tf.keras.Model, X: np.ndarray, Y: np.ndarray) -> tuple:
    """
    Get PINN predictions for the entire domain from a loaded model.
    """
    xy = np.stack([X.flatten(), Y.flatten()], axis=-1)
    uvp = network.predict(xy, batch_size=len(xy), verbose=0)
    
    u = uvp[..., 0].reshape(X.shape)
    v = uvp[..., 1].reshape(X.shape)
    p = uvp[..., 2].reshape(X.shape)
    
    return u, v, p


def load_case(base_dir: str, case_info: dict) -> dict:
    """
    Load a single training case from saved CFD .npz file and query PINN model.
    
    PINN model is expected in the same directory as the CFD solution:
        data/Re_{Re}/cyl_pos_{Cx}_{Cy}/
            ├── cfd_solution.npz
            └── pinn_model.h5
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing CFD data
    case_info : dict
        Case metadata from cases.json
    
    Returns:
    --------
    dict with all case data
    """
    # Load CFD solution
    cfd_path = os.path.join(base_dir, case_info['cfd_path'])
    cfd = np.load(cfd_path)
    
    X = cfd['X']
    Y = cfd['Y']
    
    # Load PINN model from same directory as CFD
    pinn_model_path = os.path.join(base_dir, case_info['pinn_model_path'])
    
    network = load_pinn_model(pinn_model_path)
    u_pinn, v_pinn, p_pinn = get_pinn_predictions_from_model(network, X, Y)
    
    return {
        'id': case_info['id'],
        'Re': case_info['Re'],
        'cylinder_center': tuple(case_info['cylinder_center']),
        'cylinder_radius': case_info['cylinder_radius'],
        'inlet_velocity': case_info['inlet_velocity'],
        'u_cfd': cfd['u'],
        'v_cfd': cfd['v'],
        'p_cfd': cfd['p'],
        'u_pinn': u_pinn,
        'v_pinn': v_pinn,
        'p_pinn': p_pinn,
        'X': X,
        'Y': Y,
        'cylinder_mask': cfd['cylinder_mask']
    }


def load_all_cases(base_dir: str = DATA_DIR) -> tuple:
    """
    Load all training cases from the data directory.
    
    CFD solutions are loaded from .npz files.
    PINN predictions are computed by querying the PINN models.
    
    Each case directory should contain:
        - cfd_solution.npz
        - pinn_model.h5
    
    Parameters:
    -----------
    base_dir : str
        Base directory containing CFD data and cases.json
    
    Returns:
    --------
    cases : list of dict
        All loaded cases
    metadata : dict
        Metadata from cases.json
    """
    metadata_path = os.path.join(base_dir, 'cases.json')
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    cases = []
    for case_info in metadata['cases']:
        try:
            case = load_case(base_dir, case_info)
            cases.append(case)
            print(f"  ✓ Loaded: {case_info['id']}")
        except Exception as e:
            print(f"  ✗ Failed to load {case_info['id']}: {e}")
    
    return cases, metadata


def create_metadata_template(output_path: str = 'data/cases.json'):
    """
    Create a template cases.json file.
    
    Each case directory should contain:
        data/Re_{Re}/cyl_pos_{Cx}_{Cy}/
            ├── cfd_solution.npz
            └── pinn_model.h5
    """
    cases = []
    
    for Re in RE_VALUES:
        for (Cx, Cy) in CYLINDER_POSITIONS:
            case_id = f"Re{int(Re)}_cyl_{Cx}_{Cy}"
            case_dir = f"Re_{int(Re)}/cyl_pos_{Cx}_{Cy}"
            cases.append({
                "id": case_id,
                "Re": Re,
                "cylinder_center": [Cx, Cy],
                "cylinder_radius": CYLINDER_RADIUS,
                "inlet_velocity": INLET_VELOCITY,
                "cfd_path": f"{case_dir}/cfd_solution.npz",
                "pinn_model_path": f"{case_dir}/pinn_model.h5"
            })
    
    metadata = {
        "description": "Training cases for learned router (5 Re x 5 cylinder positions)",
        "note": "PINN models expected in same directory as CFD: {case_dir}/pinn_model.h5",
        "domain": {
            "x_domain": list(X_DOMAIN),
            "y_domain": list(Y_DOMAIN),
            "N": N
        },
        "cases": cases
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created template: {output_path}")
    print(f"Total cases: {len(cases)}")
    print(f"\nExpected structure per case:")
    print(f"  data/Re_{{Re}}/cyl_pos_{{Cx}}_{{Cy}}/")
    print(f"      ├── cfd_solution.npz")
    print(f"      └── pinn_model.h5")
    return metadata


# =============================================================================
# Training Functions
# =============================================================================

def train_router_multi_case(cases: list, 
                             metadata: dict,
                             epochs: int = 200,
                             error_threshold: float = 0.1,
                             save_path: str = './models/router_multi.h5',
                             batch_size: int = 4) -> tuple:
    """
    Train the learned router on multiple cases.
    
    Parameters:
    -----------
    cases : list of dict
        Loaded training cases
    metadata : dict
        Metadata with domain info
    epochs : int
        Number of training epochs
    error_threshold : float
        PINN-CFD error threshold for label generation
    save_path : str
        Path to save trained router
    batch_size : int
        Number of cases per training batch
    
    Returns:
    --------
    learned_router : LearnedRouter
        Trained router
    history : dict
        Training history
    """
    print("\n" + "=" * 70)
    print("Training Learned Router on Multiple Cases")
    print("=" * 70)
    print(f"Number of cases: {len(cases)}")
    print(f"Epochs: {epochs}")
    
    # Get domain info
    domain = metadata['domain']
    x_domain = tuple(domain['x_domain'])
    y_domain = tuple(domain['y_domain'])
    
    # Use first case to get grid info
    first_case = cases[0]
    X, Y = first_case['X'], first_case['Y']
    Ny, Nx = X.shape
    dx = (x_domain[1] - x_domain[0]) / (Nx - 1)
    dy = (y_domain[1] - y_domain[0]) / (Ny - 1)
    
    # Create router with "average" parameters (will adapt to different Re/positions via features)
    avg_Re = np.mean([c['Re'] for c in cases])
    avg_cyl = (
        np.mean([c['cylinder_center'][0] for c in cases]),
        np.mean([c['cylinder_center'][1] for c in cases])
    )
    avg_radius = np.mean([c['cylinder_radius'] for c in cases])
    
    learned_router, trainer = create_router(
        x_domain=x_domain,
        y_domain=y_domain,
        cylinder_center=avg_cyl,
        cylinder_radius=avg_radius,
        inlet_velocity=INLET_VELOCITY,
        nu=1.0 / avg_Re,  # Average viscosity
        # Loss weights
        threshold=1.0,
        lambda_spatial=0.1,
        lambda_upstream=1.0,
        lambda_connect=0.5,
        learning_rate=1e-3
    )
    
    # Prepare training data for all cases
    print("\nPreparing training data for all cases...")
    all_training_data = []
    
    for case in cases:
        # Update feature extractor for this case's parameters
        trainer.feature_extractor = RouterFeatureExtractor(
            x_domain=x_domain,
            y_domain=y_domain,
            cylinder_center=case['cylinder_center'],
            cylinder_radius=case['cylinder_radius'],
            inlet_velocity=case['inlet_velocity'],
            nu=1.0 / case['Re']
        )
        
        # Prepare data
        data = trainer.prepare_training_data(
            case['u_pinn'], case['v_pinn'], case['p_pinn'],
            case['X'], case['Y'], dx, dy,
            u_cfd=case['u_cfd'], v_cfd=case['v_cfd'],
            error_threshold=error_threshold
        )
        all_training_data.append(data)
        
        # Compute and print error for this case
        u_ref = np.max(np.abs(case['u_cfd'])) + 1e-10
        error = np.sqrt(np.mean((case['u_pinn'] - case['u_cfd'])**2 + 
                                (case['v_pinn'] - case['v_cfd'])**2)) / u_ref
        print(f"  {case['id']}: PINN-CFD error = {error:.4f}")
    
    # Train router on all cases
    print(f"\nTraining for {epochs} epochs on {len(cases)} cases...")
    history = trainer.train(all_training_data, epochs=epochs, verbose=True)
    
    # Save trained router
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    learned_router.save(save_path)
    print(f"\nRouter saved to {save_path}")
    
    return learned_router, history


def evaluate_router_on_cases(learned_router: LearnedRouter,
                              cases: list,
                              output_dir: str = './images/router_evaluation') -> dict:
    """
    Evaluate trained router on all cases.
    
    Parameters:
    -----------
    learned_router : LearnedRouter
        Trained router
    cases : list
        Test cases
    output_dir : str
        Output directory for plots
    
    Returns:
    --------
    results : dict
        Evaluation results per case
    """
    print("\n" + "=" * 70)
    print("Evaluating Router on All Cases")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    results = {}
    
    for case in cases:
        case_id = case['id']
        print(f"\n--- Evaluating {case_id} ---")
        
        # Update router's feature extractor for this case
        learned_router.feature_extractor = RouterFeatureExtractor(
            x_domain=(case['X'].min(), case['X'].max()),
            y_domain=(case['Y'].min(), case['Y'].max()),
            cylinder_center=case['cylinder_center'],
            cylinder_radius=case['cylinder_radius'],
            inlet_velocity=case['inlet_velocity'],
            nu=1.0 / case['Re']
        )
        
        # Get grid spacing
        Ny, Nx = case['X'].shape
        dx = (case['X'].max() - case['X'].min()) / (Nx - 1)
        dy = (case['Y'].max() - case['Y'].min()) / (Ny - 1)
        
        # Predict mask
        mask, r_prob = learned_router.predict_mask(
            case['u_pinn'], case['v_pinn'], case['p_pinn'],
            case['X'], case['Y'], dx, dy,
            cylinder_mask=case['cylinder_mask']
        )
        
        # Compute statistics
        total = Ny * Nx
        cfd_pct = 100 * np.sum(mask) / total
        
        # Compute PINN error in regions
        u_ref = np.max(np.abs(case['u_cfd'])) + 1e-10
        error = np.sqrt((case['u_pinn'] - case['u_cfd'])**2 + 
                       (case['v_pinn'] - case['v_cfd'])**2)
        
        error_in_pinn_region = np.mean(error[mask == 0]) / u_ref if np.any(mask == 0) else 0
        error_in_cfd_region = np.mean(error[mask == 1]) / u_ref if np.any(mask == 1) else 0
        
        results[case_id] = {
            'mask': mask,
            'r_prob': r_prob,
            'cfd_percentage': cfd_pct,
            'pinn_percentage': 100 - cfd_pct,
            'error_in_pinn_region': error_in_pinn_region,
            'error_in_cfd_region': error_in_cfd_region,
        }
        
        print(f"  CFD region: {cfd_pct:.1f}%")
        print(f"  Avg error in PINN region: {error_in_pinn_region:.4f}")
        print(f"  Avg error in CFD region: {error_in_cfd_region:.4f}")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("Summary Statistics")
    print("=" * 70)
    
    avg_cfd_pct = np.mean([r['cfd_percentage'] for r in results.values()])
    avg_pinn_error = np.mean([r['error_in_pinn_region'] for r in results.values()])
    
    print(f"Average CFD region: {avg_cfd_pct:.1f}%")
    print(f"Average error in PINN regions: {avg_pinn_error:.4f}")
    
    return results


# =============================================================================
# Visualization Functions
# =============================================================================

def plot_multi_case_results(cases: list, 
                            results: dict,
                            output_dir: str = './images/router_evaluation'):
    """
    Create visualization of router results across multiple cases.
    """
    import matplotlib.pyplot as plt
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Group cases by Re
    re_groups = {}
    for case in cases:
        Re = case['Re']
        if Re not in re_groups:
            re_groups[Re] = []
        re_groups[Re].append(case)
    
    # Plot masks for each Re group
    for Re, group_cases in re_groups.items():
        n_cases = len(group_cases)
        fig, axes = plt.subplots(2, n_cases, figsize=(4*n_cases, 8))
        
        for i, case in enumerate(group_cases):
            case_id = case['id']
            result = results[case_id]
            
            # Top row: rejection probability
            ax = axes[0, i] if n_cases > 1 else axes[0]
            cf = ax.contourf(case['X'], case['Y'], result['r_prob'], 
                           levels=50, cmap='RdBu_r', vmin=0, vmax=1)
            circle = plt.Circle(case['cylinder_center'], case['cylinder_radius'], 
                              fc='gray', ec='white', linewidth=2)
            ax.add_patch(circle)
            ax.set_aspect('equal')
            ax.set_title(f"Cyl @ ({case['cylinder_center'][0]}, {case['cylinder_center'][1]})")
            ax.set_xlabel('x')
            if i == 0:
                ax.set_ylabel('Rejection Prob.\ny')
            
            # Bottom row: binary mask
            ax = axes[1, i] if n_cases > 1 else axes[1]
            ax.contourf(case['X'], case['Y'], result['mask'], 
                       levels=[0, 0.5, 1], colors=['blue', 'red'], alpha=0.7)
            circle = plt.Circle(case['cylinder_center'], case['cylinder_radius'], 
                              fc='black', ec='white', linewidth=2)
            ax.add_patch(circle)
            ax.set_aspect('equal')
            ax.set_xlabel('x')
            if i == 0:
                ax.set_ylabel('Binary Mask\ny')
            ax.text(0.5, -0.15, f"CFD: {result['cfd_percentage']:.0f}%", 
                   transform=ax.transAxes, ha='center')
        
        plt.suptitle(f'Re = {int(Re)}', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'masks_Re{int(Re)}.png'), dpi=150)
        plt.close()
    
    print(f"\nPlots saved to {output_dir}/")


# =============================================================================
# Single Case Functions (for testing/demo)
# =============================================================================

def get_pinn_predictions(network, X, Y):
    """Get PINN predictions for the entire domain."""
    xy = np.stack([X.flatten(), Y.flatten()], axis=-1)
    uvp = network.predict(xy, batch_size=len(xy))
    
    u = uvp[..., 0].reshape(X.shape)
    v = uvp[..., 1].reshape(X.shape)
    p = uvp[..., 2].reshape(X.shape)
    
    return u, v, p


def run_cfd_simulation(Re=100, N=100, x_domain=(0, 2), y_domain=(0, 1),
                        cylinder_center=(0.5, 0.5), cylinder_radius=0.1,
                        inlet_velocity=1.0, max_iter=50000, tol=1e-6):
    """Run pure CFD simulation to get ground truth."""
    print("\n" + "=" * 70)
    print("Running CFD simulation (ground truth)")
    print("=" * 70)
    
    sim = CylinderFlowSimulation(
        Re=Re, N=N, max_iter=max_iter, tol=tol,
        x_domain=x_domain, y_domain=y_domain,
        cylinder_center=cylinder_center, cylinder_radius=cylinder_radius,
        inlet_velocity=inlet_velocity
    )
    
    start_time = time.time()
    u_cfd, v_cfd, p_cfd = sim.solve()
    elapsed = time.time() - start_time
    
    print(f"CFD simulation completed in {elapsed:.2f} seconds")
    
    return u_cfd, v_cfd, p_cfd, sim


# =============================================================================
# Main Functions
# =============================================================================

def main_train_from_saved_data():
    """
    Main function: Train router using pre-saved CFD and PINN data.
    """
    print("\n" + "=" * 70)
    print("LEARNED ROUTER TRAINING (Multi-Case)")
    print("=" * 70)
    
    # Load all cases
    print("\nLoading training cases...")
    try:
        cases, metadata = load_all_cases(DATA_DIR)
    except FileNotFoundError:
        print(f"\n⚠ Data not found at {DATA_DIR}/cases.json")
        print("Creating template file structure...")
        create_metadata_template()
        print("\nPlease populate the data directory with CFD and PINN solutions.")
        print("See DATA_FORMAT.md for file format specifications.")
        return
    
    if len(cases) == 0:
        print("No cases loaded. Please check your data directory.")
        return
    
    print(f"\nLoaded {len(cases)} cases successfully")
    
    # Train router
    learned_router, history = train_router_multi_case(
        cases, metadata,
        epochs=200,
        error_threshold=0.1,
        save_path='./models/router_multi.h5'
    )
    
    # Evaluate on all cases
    results = evaluate_router_on_cases(learned_router, cases)
    
    # Plot results
    plot_multi_case_results(cases, results)
    
    # Plot training history
    from lib.plotting import plot_router_training_history
    plot_router_training_history(history, save_path='./images/router_evaluation/training_history.png')
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)


def main_generate_template():
    """Generate template files for data preparation."""
    print("\n" + "=" * 70)
    print("GENERATING DATA TEMPLATE")
    print("=" * 70)
    
    create_metadata_template('data/cases.json')
    
    # Create directory structure
    for Re in RE_VALUES:
        for (Cx, Cy) in CYLINDER_POSITIONS:
            case_dir = f"data/Re_{int(Re)}/cyl_pos_{Cx}_{Cy}"
            os.makedirs(case_dir, exist_ok=True)
    
    print("\nDirectory structure created.")
    print("\nNext steps:")
    print("1. Run CFD simulations and save to data/Re_*/cyl_pos_*/cfd_solution.npz")
    print("2. Get PINN predictions and save to data/Re_*/cyl_pos_*/pinn_solution.npz")
    print("3. Run: python example_learned_router.py train")


def main_demo_single_case():
    """
    Demo: Train and test router on a single case (requires running CFD).
    """
    print("\n" + "=" * 70)
    print("LEARNED ROUTER DEMO (Single Case)")
    print("=" * 70)
    
    Re = 100
    cylinder_center = (0.5, 0.5)
    
    # Load PINN model
    print("\n--- Loading PINN Model ---")
    network = Network().build(num_inputs=2, num_outputs=3)
    
    model_path = f'./models/pinn_cylinder_{float(Re)}.h5'
    try:
        network.load_weights(model_path)
        print(f"✓ Loaded PINN weights from {model_path}")
    except Exception as e:
        print(f"⚠ Could not load PINN model: {e}")
        print("  Using untrained network")
    
    # Create simulation for grid info
    sim = CylinderFlowSimulation(
        Re=Re, N=N, max_iter=1, tol=1e-6,
        x_domain=X_DOMAIN, y_domain=Y_DOMAIN,
        cylinder_center=cylinder_center, cylinder_radius=CYLINDER_RADIUS,
        inlet_velocity=INLET_VELOCITY
    )
    
    # Get PINN predictions
    u_pinn, v_pinn, p_pinn = get_pinn_predictions(network, sim.X, sim.Y)
    
    # Run CFD for ground truth
    u_cfd, v_cfd, p_cfd, sim = run_cfd_simulation(
        Re=Re, N=N,
        x_domain=X_DOMAIN, y_domain=Y_DOMAIN,
        cylinder_center=cylinder_center, cylinder_radius=CYLINDER_RADIUS,
        inlet_velocity=INLET_VELOCITY,
        max_iter=50000, tol=1e-6
    )
    
    # Create single-case training data
    cases = [{
        'id': f'Re{Re}_demo',
        'Re': Re,
        'cylinder_center': cylinder_center,
        'cylinder_radius': CYLINDER_RADIUS,
        'inlet_velocity': INLET_VELOCITY,
        'u_cfd': u_cfd, 'v_cfd': v_cfd, 'p_cfd': p_cfd,
        'u_pinn': u_pinn, 'v_pinn': v_pinn, 'p_pinn': p_pinn,
        'X': sim.X, 'Y': sim.Y,
        'cylinder_mask': sim.cylinder_mask
    }]
    
    metadata = {
        'domain': {
            'x_domain': list(X_DOMAIN),
            'y_domain': list(Y_DOMAIN),
            'N': N
        }
    }
    
    # Train router
    learned_router, history = train_router_multi_case(
        cases, metadata,
        epochs=100,
        save_path='./models/router_demo.h5'
    )
    
    # Evaluate
    results = evaluate_router_on_cases(learned_router, cases)
    
    print("\n✓ Demo completed!")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'train':
            main_train_from_saved_data()
        elif command == 'template':
            main_generate_template()
        elif command == 'demo':
            main_demo_single_case()
        else:
            print(f"Unknown command: {command}")
            print("\nUsage:")
            print("  python example_learned_router.py train     # Train on saved data")
            print("  python example_learned_router.py template  # Generate data template")
            print("  python example_learned_router.py demo      # Run single-case demo")
    else:
        print("Usage:")
        print("  python example_learned_router.py train     # Train on saved data")
        print("  python example_learned_router.py template  # Generate data template")
        print("  python example_learned_router.py demo      # Run single-case demo")
        print("\nRunning template generation...")
        main_generate_template()

