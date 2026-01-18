"""
Example: Dynamic Segregation for Hybrid PINN-CFD Simulations

This script demonstrates how to use complexity-based segregation for hybrid solvers.
Instead of hard-coded masks, the domain is automatically divided based on local flow
diagnostics (strain rate, vorticity, residuals).
"""

import numpy as np
import time
from lib.cavity_flow import CavityFlowDynamicHybridSimulation
from lib.network import Network
from lib.plotting import plot_solution, plot_hybrid_solution, plot_streamlines


def example_cavity_flow_dynamic_segregation():
    """
    Example: Cavity flow with dynamic complexity-based segregation.
    
    This demonstrates:
    1. Computing complexity score from initial flow field
    2. Automatic segregation into CFD and PINN regions
    3. Running hybrid solver with dynamic mask
    """
    
    print("\n" + "="*70)
    print("EXAMPLE: CAVITY FLOW WITH DYNAMIC SEGREGATION")
    print("="*70)
    
    # Physical parameters
    Re = 100
    N = 100
    u0 = 1.0
    L = 1.0
    nu = 1.0 / Re
    
    print(f"\nPhysical Configuration:")
    print(f"  Reynolds number: {Re}")
    print(f"  Grid size: {N}×{N}")
    print(f"  Domain: [0,1] × [0,1]")
    print(f"  Kinematic viscosity: {nu:.6f}")
    
    # ========================================================================
    # Step 1: Load PINN model
    # ========================================================================
    print(f"\nStep 1: Loading PINN model...")
    network = Network().build()
    
    try:
        network.load_weights('./models/pinn_cavity_flow.h5')
        print("  ✓ Model loaded from ./models/pinn_cavity_flow.h5")
    except Exception as e:
        print(f"  ⚠ Warning: Could not load model: {e}")
        print("  Using untrained network (prediction will be inaccurate)")
    
    # ========================================================================
    # Step 2: Create initial flow field
    # ========================================================================
    print(f"\nStep 2: Creating initial flow field...")
    
    # Create coordinate grid
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    X, Y = np.meshgrid(x, y)
    xy = np.stack([X.flatten(), Y.flatten()], axis=-1)
    
    # Get PINN prediction for initial field
    psi_p_init = network.predict(xy, batch_size=len(xy))
    
    # Extract initial velocities (from stream function)
    import tensorflow as tf
    xy_tf = tf.constant(xy, dtype=tf.float32)
    with tf.GradientTape() as g:
        g.watch(xy_tf)
        psi_p = network(xy_tf)
    psi_p_j = g.batch_jacobian(psi_p, xy_tf)
    
    u_init = psi_p_j[..., 0, 1].numpy().reshape(X.shape)  # ∂ψ/∂y
    v_init = -psi_p_j[..., 0, 0].numpy().reshape(X.shape)  # -∂ψ/∂x
    p_init = psi_p_init[..., 1].reshape(X.shape)
    
    print(f"  Initial velocity range:")
    print(f"    u: [{u_init.min():.4f}, {u_init.max():.4f}]")
    print(f"    v: [{v_init.min():.4f}, {v_init.max():.4f}]")
    print(f"  Pressure range: [{p_init.min():.4f}, {p_init.max():.4f}]")
    
    # ========================================================================
    # Step 3: Experiment with different thresholds
    # ========================================================================
    thresholds = [0.5, 1.0, 1.5, 2.0]
    
    for threshold in thresholds:
        print(f"\n" + "-"*70)
        print(f"Configuration: complexity_threshold = {threshold}")
        print("-"*70)
        
        # Create simulation with dynamic segregation
        sim = CavityFlowDynamicHybridSimulation(
            network=network,
            uv_func=lambda net, xy_pts: (
                _compute_uv_from_psi(net, xy_pts)[0],
                _compute_uv_from_psi(net, xy_pts)[1]
            ),
            u_init=u_init,
            v_init=v_init,
            p_init=p_init,
            Re=Re,
            N=N,
            max_iter=200000,
            tol=1e-6,
            complexity_threshold=threshold,
            # Use default weights that emphasize residuals
            complexity_weights={
                'strain': 1.0,
                'vorticity': 1.0,
                'momentum': 2.0,      # Residual-based terms get higher weight
                'continuity': 2.0
            },
            normalization='mean'
        )
        
        # Run solver
        print(f"\nSolving hybrid system...")
        start_time = time.time()
        u, v, p = sim.solve()
        elapsed = time.time() - start_time
        
        print(f"Elapsed time: {elapsed:.2f} seconds")
        
        # Save results
        filename_base = f'cavity_dynamic_threshold_{threshold:.1f}'
        
        # Plot solution
        plot_solution(u, v, p, 
                     title_prefix=f'Dynamic Segregation (τ={threshold}): ',
                     save_path=f'{filename_base}_solution.png')
        
        # Plot with segregation mask
        plot_hybrid_solution(u, v, p, sim.mask,
                            title_prefix=f'Segregation (τ={threshold}): ',
                            save_path=f'{filename_base}_mask.png')
        
        # Plot streamlines
        plot_streamlines(u, v,
                        save_path=f'{filename_base}_streamlines.png',
                        title=f'Streamlines: Dynamic Segregation (τ={threshold})')
    
    print("\n" + "="*70)
    print("Examples completed! Results saved with filenames:")
    print("  cavity_dynamic_threshold_*.png")
    print("="*70)
    
    return sim


def _compute_uv_from_psi(network, xy):
    """Helper function to compute u,v from stream function."""
    import tensorflow as tf
    xy_tf = tf.constant(xy, dtype=tf.float32)
    with tf.GradientTape() as g:
        g.watch(xy_tf)
        psi_p = network(xy_tf)
    psi_p_j = g.batch_jacobian(psi_p, xy_tf)
    u = psi_p_j[..., 0, 1].numpy()  # ∂ψ/∂y
    v = -psi_p_j[..., 0, 0].numpy()  # -∂ψ/∂x
    return u, v


def example_analysis_complexity_scores():
    """
    Example: Analyze complexity scores to guide threshold selection.
    """
    
    print("\n" + "="*70)
    print("EXAMPLE: ANALYZING COMPLEXITY SCORES")
    print("="*70)
    
    from lib.complexity_scoring import ComplexityScorer, compute_mask_statistics
    
    # Load model and create initial field (same as above)
    Re = 100
    N = 100
    nu = 1.0 / Re
    
    network = Network().build()
    try:
        network.load_weights('./models/pinn_cavity_flow.h5')
    except:
        print("Warning: Model not found")
    
    # Create grid and initial field
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    xy = np.stack([X.flatten(), Y.flatten()], axis=-1)
    
    # Get initial velocity
    psi_p = network.predict(xy, batch_size=len(xy))
    u_init = psi_p[..., 0].reshape(X.shape)
    v_init = psi_p[..., 1].reshape(X.shape)
    p_init = psi_p[..., 2].reshape(X.shape)
    
    # Compute complexity scores
    print(f"\nComputing complexity scores...")
    scorer = ComplexityScorer(
        weights={
            'strain': 1.0,
            'vorticity': 1.0,
            'momentum': 2.0,
            'continuity': 2.0
        },
        normalization='mean'
    )
    
    complexity_score = scorer.compute_complexity_score(
        u_init, v_init, p_init, nu,
        dx=1.0/(N-1), dy=1.0/(N-1)
    )
    
    # Analyze statistics
    print(f"\nComplexity Score Statistics:")
    print(f"  Minimum: {np.min(complexity_score):.6f}")
    print(f"  25th percentile: {np.percentile(complexity_score, 25):.6f}")
    print(f"  Median (50th): {np.percentile(complexity_score, 50):.6f}")
    print(f"  75th percentile: {np.percentile(complexity_score, 75):.6f}")
    print(f"  Maximum: {np.max(complexity_score):.6f}")
    print(f"  Mean: {np.mean(complexity_score):.6f}")
    print(f"  Std Dev: {np.std(complexity_score):.6f}")
    
    # Visualize distribution
    print(f"\nComplexity score distribution:")
    bins = np.linspace(np.min(complexity_score), np.max(complexity_score), 20)
    hist, _ = np.histogram(complexity_score, bins=bins)
    
    for i, count in enumerate(hist):
        bar = "█" * max(1, count // 20)
        print(f"  {bins[i]:6.3f} - {bins[i+1]:6.3f}: {bar} ({count})")
    
    # Recommend thresholds
    print(f"\nRecommended threshold values:")
    for percentile in [25, 50, 75]:
        thresh = np.percentile(complexity_score, percentile)
        mask = (complexity_score > thresh).astype(np.int32)
        stats = compute_mask_statistics(mask, complexity_score)
        print(f"  τ = {thresh:.3f} (percentile {percentile}): "
              f"{stats['cfd_percentage']:.1f}% CFD, "
              f"{stats['pinn_percentage']:.1f}% PINN")
    
    # Get diagnostic components
    print(f"\nDiagnostic component statistics:")
    diags = scorer.get_diagnostics(u_init, v_init, p_init, nu,
                                   dx=1.0/(N-1), dy=1.0/(N-1))
    
    for component in ['strain', 'vorticity', 'momentum', 'continuity']:
        data = diags[component]
        ref = diags['ref_scales'][component]
        print(f"  {component:12s}: "
              f"max={np.max(data):.6e}, "
              f"mean={np.mean(data):.6e}, "
              f"ref={ref:.6e}")


if __name__ == '__main__':
    # Run cavity flow example
    example_cavity_flow_dynamic_segregation()
    
    # Run complexity analysis
    example_analysis_complexity_scores()
    
    print("\n✓ Examples completed successfully!")
