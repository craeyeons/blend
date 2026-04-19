"""
Train the CNN-based router for hybrid PINN-FDM elasticity simulations.

Usage:
    python train_router.py --problem plate_with_hole --model-path ./models/pinn_plate_with_hole.h5
    python train_router.py --problem l_bracket --model-path ./models/pinn_l_bracket.h5 --beta 0.5
"""

import argparse
import os
import json
import time
import numpy as np
import tensorflow as tf
from datetime import datetime

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

from lib.network import Network
from lib.domains import create_plate_with_hole, create_l_bracket
from lib.router import (
    RouterCNN,
    RouterTrainer,
    create_router_input,
    compute_bc_error_field,
    solve_error_transport,
    plot_router_output,
    plot_training_history,
    plot_coverage_evolution,
)


def main():
    parser = argparse.ArgumentParser(
        description='Train CNN router for hybrid PINN-FDM elasticity'
    )

    parser.add_argument('--problem', type=str, default='plate_with_hole',
                        choices=['plate_with_hole', 'l_bracket'])
    parser.add_argument('--model-path', type=str,
                        default='./models/pinn_plate_with_hole.h5')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--output-base-dir', type=str, default='./router_output')

    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--lambda-tv', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--grad-clip', type=float, default=1.0)

    # Residual weights
    parser.add_argument('--weight-eq-x', type=float, default=1.0)
    parser.add_argument('--weight-eq-y', type=float, default=1.0)
    parser.add_argument('--residual-source', type=str, default='combined',
                        choices=['combined', 'pde', 'ete'],
                        help="Residual source for router loss: 'combined' (PDE+BC error), 'pde', or 'ete'")

    # Grid
    parser.add_argument('--nx', type=int, default=200)
    parser.add_argument('--ny', type=int, default=200)

    # Domain
    parser.add_argument('--x-min', type=float, default=-2.0)
    parser.add_argument('--x-max', type=float, default=2.0)
    parser.add_argument('--y-min', type=float, default=-2.0)
    parser.add_argument('--y-max', type=float, default=2.0)

    # Plate with hole
    parser.add_argument('--hole-x', type=float, default=0.0)
    parser.add_argument('--hole-y', type=float, default=0.0)
    parser.add_argument('--hole-radius', type=float, default=0.5)
    parser.add_argument('--applied-stress', type=float, default=10.0)

    # L-bracket
    parser.add_argument('--corner-x', type=float, default=1.0)
    parser.add_argument('--corner-y', type=float, default=1.0)
    parser.add_argument('--fillet-radius', type=float, default=0.04,
                        help='Fillet radius at L-bracket re-entrant corner (0=sharp)')

    # Material
    parser.add_argument('--E', type=float, default=1.0)
    parser.add_argument('--nu', type=float, default=0.3)

    # Router
    parser.add_argument('--layers', type=int, nargs='+', default=[128, 128, 128, 128])
    parser.add_argument('--base-filters', type=int, default=32)
    parser.add_argument('--threshold', type=float, default=0.0)

    args = parser.parse_args()

    # Adjust domain for L-bracket
    if args.problem == 'l_bracket' and args.x_min == -2.0:
        args.x_min = 0.0
        args.y_min = 0.0

    if args.output_dir is None:
        beta_str = f"beta_{args.beta:.4f}".rstrip('0').rstrip('.')
        args.output_dir = os.path.join(
            args.output_base_dir, args.problem, beta_str
        )
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(f"ROUTER TRAINING: {args.problem}")
    print("=" * 60)

    # Step 1: Load PINN
    print("\n[Step 1] Loading PINN model...")
    network = Network()
    input_range = [(args.x_min, args.x_max), (args.y_min, args.y_max)]
    hard_bc_params = None
    if args.problem == 'l_bracket':
        hard_bc_params = {'corner_x': args.corner_x, 'corner_y': args.corner_y}
    pinn_model = network.build(num_inputs=2, layers=args.layers,
                               activation='tanh', num_outputs=2,
                               input_range=input_range,
                               hard_bc=args.problem,
                               hard_bc_params=hard_bc_params)
    try:
        pinn_model.load_weights(args.model_path)
        print(f"  Loaded: {args.model_path}")
    except Exception as e:
        print(f"  Failed to load PINN: {e}")
        return

    # Step 2: Create domain setup
    print("\n[Step 2] Creating domain setup...")
    if args.problem == 'plate_with_hole':
        X, Y, layout, disp_bc_mask, trac_bc_mask, bc_ux, bc_uy, bc_tx, bc_ty = \
            create_plate_with_hole(
                Nx=args.nx, Ny=args.ny,
                x_domain=(args.x_min, args.x_max),
                y_domain=(args.y_min, args.y_max),
                hole_center=(args.hole_x, args.hole_y),
                hole_radius=args.hole_radius,
                applied_stress=args.applied_stress,
            )
        show_hole = (args.hole_x, args.hole_y, args.hole_radius)
    else:
        X, Y, layout, disp_bc_mask, trac_bc_mask, bc_ux, bc_uy, bc_tx, bc_ty = \
            create_l_bracket(
                Nx=args.nx, Ny=args.ny,
                x_domain=(args.x_min, args.x_max),
                y_domain=(args.y_min, args.y_max),
                corner_x=args.corner_x,
                corner_y=args.corner_y,
                applied_stress=args.applied_stress,
                fillet_radius=args.fillet_radius,
            )
        show_hole = None

    print(f"  Grid: {args.nx} x {args.ny}")
    print(f"  Material points: {np.sum(layout):.0f} / {layout.size}")

    # Step 2b: PINN predictions (timed — this is part of the hybrid pipeline)
    print("\n[Step 2b] Computing PINN predictions...")
    xy_flat = np.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)
    # Warmup pass (JIT / first-call overhead)
    _ = pinn_model.predict(xy_flat[:1], verbose=0)
    t_pinn_start = time.perf_counter()
    pinn_out = pinn_model.predict(xy_flat, batch_size=len(xy_flat), verbose=0)
    t_pinn_end = time.perf_counter()
    pinn_inference_s = t_pinn_end - t_pinn_start
    pinn_ux = pinn_out[:, 0].reshape(X.shape).astype(np.float32) * layout
    pinn_uy = pinn_out[:, 1].reshape(X.shape).astype(np.float32) * layout
    print(f"  PINN inference time: {pinn_inference_s:.4f}s")

    # Compute von Mises from PINN (finite differences on predictions)
    dx = (args.x_max - args.x_min) / (args.nx - 1)
    dy = (args.y_max - args.y_min) / (args.ny - 1)
    C11 = args.E / (1.0 - args.nu ** 2)
    C12 = args.nu * args.E / (1.0 - args.nu ** 2)
    C66 = args.E / (2.0 * (1.0 + args.nu))

    exx = np.zeros_like(pinn_ux)
    eyy = np.zeros_like(pinn_uy)
    exy = np.zeros_like(pinn_ux)
    exx[1:-1, 1:-1] = (pinn_ux[1:-1, 2:] - pinn_ux[1:-1, :-2]) / (2 * dx)
    eyy[1:-1, 1:-1] = (pinn_uy[2:, 1:-1] - pinn_uy[:-2, 1:-1]) / (2 * dy)
    exy[1:-1, 1:-1] = 0.5 * (
        (pinn_ux[2:, 1:-1] - pinn_ux[:-2, 1:-1]) / (2 * dy)
        + (pinn_uy[1:-1, 2:] - pinn_uy[1:-1, :-2]) / (2 * dx)
    )
    sxx = C11 * exx + C12 * eyy
    syy = C12 * exx + C11 * eyy
    sxy = 2 * C66 * exy
    pinn_vm = np.sqrt(sxx ** 2 - sxx * syy + syy ** 2 + 3 * sxy ** 2).astype(np.float32) * layout

    print(f"  PINN ux range: [{pinn_ux.min():.4f}, {pinn_ux.max():.4f}]")
    print(f"  PINN uy range: [{pinn_uy.min():.4f}, {pinn_uy.max():.4f}]")
    print(f"  PINN VM range: [{pinn_vm.min():.4f}, {pinn_vm.max():.4f}]")

    # Step 2c: BC error and error transport (ETE equivalent)
    print("\n[Step 2c] Computing error transport field...")
    bc_error = compute_bc_error_field(disp_bc_mask, bc_ux, bc_uy, pinn_ux, pinn_uy, layout)
    error_transport = solve_error_transport(
        bc_error, layout, E=args.E, nu=args.nu,
        x_domain=(args.x_min, args.x_max),
        y_domain=(args.y_min, args.y_max),
    )
    print(f"  BC error range: [{bc_error.min():.4f}, {bc_error.max():.4f}]")
    print(f"  Error transport range: [{error_transport.min():.4f}, {error_transport.max():.4f}]")
    print(f"  Nonzero fraction: {np.mean(error_transport > 0.01)*100:.1f}%")

    # Create router input (channel 8 = transported error, not raw bc_error)
    inputs = create_router_input(
        layout, disp_bc_mask, bc_ux, bc_uy, trac_bc_mask,
        pinn_ux, pinn_uy, pinn_vm, error_transport,
    )
    print(f"  Router input shape: {inputs.shape}")

    # Step 3: Init router
    print("\n[Step 3] Initializing router...")
    router = RouterCNN(base_filters=args.base_filters)
    _ = router(inputs)
    print(f"  Parameters: {router.count_params():,}")

    # Step 4: Init trainer
    print("\n[Step 4] Initializing trainer...")
    residual_weights = {'eq_x': args.weight_eq_x, 'eq_y': args.weight_eq_y}
    trainer = RouterTrainer(
        router=router,
        pinn_model=pinn_model,
        beta=args.beta,
        lambda_tv=args.lambda_tv,
        grad_clip_norm=args.grad_clip if args.grad_clip > 0 else None,
        residual_source=args.residual_source,
        residual_weights=residual_weights,
        E=args.E,
        nu=args.nu,
    )
    trainer.optimizer.learning_rate.assign(args.lr)
    print(f"  beta={args.beta}, lambda_tv={args.lambda_tv}, lr={args.lr}")
    print(f"  Residual source: {args.residual_source}")

    # Step 5: Train
    print("\n[Step 5] Training...")
    print("-" * 50)
    start = datetime.now()
    history = trainer.train(inputs, X, Y, layout, epochs=args.epochs,
                            verbose=True, lr=args.lr)
    elapsed = (datetime.now() - start).total_seconds()
    print(f"\n  Training time: {elapsed:.1f}s")

    # Step 6: Predictions (timed — router inference is part of the hybrid pipeline)
    print("\n[Step 6] Predictions...")
    # Warmup pass
    _ = trainer.predict(inputs, threshold=args.threshold)
    t_router_start = time.perf_counter()
    r, mask = trainer.predict(inputs, threshold=args.threshold)
    t_router_end = time.perf_counter()
    router_inference_s = t_router_end - t_router_start
    fdm_frac = np.sum(mask * layout) / np.sum(layout) * 100
    print(f"  FDM: {fdm_frac:.1f}%, PINN: {100-fdm_frac:.1f}%")
    print(f"  Router inference time: {router_inference_s:.4f}s")

    # Step 7: Save
    print("\n[Step 7] Saving...")
    router.save_weights(os.path.join(args.output_dir, 'router.weights.h5'))
    np.savez(os.path.join(args.output_dir, 'training_history.npz'), **history)
    np.savez(os.path.join(args.output_dir, 'predictions.npz'),
             router_output=r, mask=mask, X=X, Y=Y, layout=layout)

    # Save metrics and timing
    metrics = {
        'problem': args.problem,
        'beta': args.beta,
        'lambda_tv': args.lambda_tv,
        'lr': args.lr,
        'epochs': args.epochs,
        'grid': f'{args.nx}x{args.ny}',
        'E': args.E,
        'nu': args.nu,
        'router_params': int(router.count_params()),
        'training_time_s': elapsed,
        'pinn_inference_s': pinn_inference_s,
        'router_inference_s': router_inference_s,
        'hybrid_overhead_s': pinn_inference_s + router_inference_s,
        'fdm_fraction_pct': float(fdm_frac),
        'pinn_fraction_pct': float(100 - fdm_frac),
        'final_total_loss': float(history['total_loss'][-1]),
        'final_logistic_loss': float(history['logistic_loss'][-1]),
        'final_tv_loss': float(history['tv_loss'][-1]),
    }
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")

    # Step 8: Visualize
    print("\n[Step 8] Visualizing...")
    plot_router_output(r, X, Y, layout,
                       title=f'Router ({args.problem}, beta={args.beta})',
                       save_path=os.path.join(args.output_dir, 'router_output.png'),
                       show_hole=show_hole)
    plot_training_history(history,
                          save_path=os.path.join(args.output_dir, 'training_history.png'))

    coverage_plot_path = os.path.join(args.output_dir, 'coverage_evolution.png')
    coverage_metrics, _ = plot_coverage_evolution(
        r, layout,
        save_path=coverage_plot_path,
        title=f'Coverage Evolution ({args.problem})'
    )
    np.savez(os.path.join(args.output_dir, 'coverage_evolution.npz'),
             target_coverage=coverage_metrics['target_coverage'],
             threshold=coverage_metrics['threshold'],
             actual_coverage=coverage_metrics['actual_coverage'])

    print("  Coverage evolution (target -> achieved, threshold):")
    for tc, ac, th in zip(coverage_metrics['target_coverage'],
                          coverage_metrics['actual_coverage'],
                          coverage_metrics['threshold']):
        print(f"    {tc*100:5.1f}% -> {ac*100:5.1f}%   (thr={th:.6f})")

    # Generate grid visualization of mask evolution at each decile
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    axes = axes.flatten()
    decile_coverages = np.linspace(0.0, 1.0, 11)
    for i, cov in enumerate(decile_coverages):
        ax = axes[i]
        # Find threshold for this coverage
        if cov <= 0.0:
            threshold = r.max() + 1.0
        elif cov >= 1.0:
            threshold = r.min() - 1.0
        else:
            material_logits = r[layout > 0]
            sorted_logits = np.sort(material_logits)[::-1]
            n_material = len(sorted_logits)
            n_fdm = int(cov * n_material)
            n_fdm = max(1, min(n_fdm, n_material))
            threshold = sorted_logits[n_fdm - 1]
        
        # Create mask and visualization
        mask = (r >= threshold).astype(np.float32)
        combined = np.zeros_like(r)
        combined[layout == 0] = 0
        combined[(layout == 1) & (r < threshold)] = 1
        combined[(layout == 1) & (r >= threshold)] = 2
        
        ax.contourf(X, Y, combined, levels=[-0.5, 0.5, 1.5, 2.5],
                    colors=['gray', 'blue', 'red'], alpha=0.7)
        ax.set_aspect('equal')
        actual_cov = np.sum((r >= threshold) & (layout > 0)) / np.sum(layout > 0) * 100
        ax.set_title(f'{cov*100:.0f}% (actual: {actual_cov:.0f}%)', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    
    axes[-1].axis('off')
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='gray', label='Void'),
        Patch(facecolor='blue', alpha=0.7, label='PINN'),
        Patch(facecolor='red', alpha=0.7, label='FDM')
    ]
    fig.legend(handles=legend_elements, loc='lower right', fontsize=10)
    plt.suptitle(f'Coverage Evolution - {args.problem} (\u03b2={args.beta})', fontsize=12)
    plt.tight_layout()
    grid_path = os.path.join(args.output_dir, 'coverage_grid.png')
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved coverage grid to {grid_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
