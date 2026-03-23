"""
Train the CNN-based router on MULTIPLE cylinder configurations.

The router learns to generalize across different cylinder positions, radii,
and inlet velocities. After training, it is evaluated on held-out test configs.

Usage:
    python train_router_multi.py --config configs.json --epochs 500

Config JSON format:
{
  "train": [
    {"inlet_velocity": 1.0, "cylinder_radius": 0.1, "cylinder_x": 0.5, "cylinder_y": 0.5,
     "pinn_path": "./models/pinn_A.h5"},
    ...
  ],
  "test": [
    {"inlet_velocity": 2.0, "cylinder_radius": 0.12, "cylinder_x": 0.5, "cylinder_y": 0.6,
     "pinn_path": "./models/pinn_C.h5"}
  ]
}
"""

import argparse
import json
import os
import numpy as np
import tensorflow as tf
from datetime import datetime

# Configure TensorFlow GPU memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(f"GPU memory growth setting failed: {e}")

from lib.router import (
    RouterCNN,
    PINNResidualComputer,
    create_router_input,
    compute_smeared_bc_error,
    create_cylinder_setup,
    plot_router_output,
    plot_training_history
)
from cylinder_network import Network as CylinderNetwork


def load_pinn_model(model_path):
    """Load a pre-trained cylinder PINN model."""
    network = CylinderNetwork()
    model = network.build(
        num_inputs=2,
        layers=[48, 48, 48, 48],
        activation='tanh',
        num_outputs=3
    )
    model.load_weights(model_path)
    return model


def prepare_config(cfg, nx, ny, x_domain, y_domain, nu, rho, residual_weights):
    """
    Prepare a single domain configuration: load PINN, create domain setup,
    compute PINN predictions, and pre-compute residuals.

    Returns a dict with all tensors needed for training.
    """
    cx = cfg.get('cylinder_x', 0.5)
    cy = cfg.get('cylinder_y', 0.5)
    cr = cfg.get('cylinder_radius', 0.1)
    u_inlet = cfg.get('inlet_velocity', 1.0)
    pinn_path = cfg['pinn_path']

    label = f"cx={cx}, cy={cy}, r={cr}, u={u_inlet}"
    print(f"  Preparing: {label}")
    print(f"    PINN: {pinn_path}")

    # Load PINN
    pinn_model = load_pinn_model(pinn_path)

    # Create domain setup
    X, Y, layout, bc_mask, bc_u, bc_v, bc_p = create_cylinder_setup(
        Nx=nx, Ny=ny,
        x_domain=x_domain, y_domain=y_domain,
        cylinder_center=(cx, cy),
        cylinder_radius=cr,
        inlet_velocity=u_inlet
    )

    # Compute PINN predictions
    xy_flat = np.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)
    pinn_uvp = pinn_model.predict(xy_flat, batch_size=len(xy_flat), verbose=0)
    pinn_u = pinn_uvp[:, 0].reshape(X.shape).astype(np.float32) * layout
    pinn_v = pinn_uvp[:, 1].reshape(X.shape).astype(np.float32) * layout
    pinn_p = pinn_uvp[:, 2].reshape(X.shape).astype(np.float32) * layout

    # Compute smeared BC error
    smeared_bc_err = compute_smeared_bc_error(
        bc_mask, bc_u, bc_v, pinn_u, pinn_v, layout
    )

    # Create router input (9 channels)
    inputs = create_router_input(layout, bc_mask, bc_u, bc_v, bc_p,
                                  pinn_u, pinn_v, pinn_p, smeared_bc_err)

    # Pre-compute PINN residuals (frozen, constant across training)
    residual_computer = PINNResidualComputer(
        pinn_model, nu, rho,
        x_domain=x_domain, y_domain=y_domain,
        cylinder_center=(cx, cy),
        cylinder_radius=cr,
        inlet_velocity=u_inlet
    )

    X_tf = tf.constant(X, dtype=tf.float32)
    Y_tf = tf.constant(Y, dtype=tf.float32)
    bc_mask_tf = tf.constant(bc_mask, dtype=tf.float32)
    bc_u_tf = tf.constant(bc_u, dtype=tf.float32)
    bc_v_tf = tf.constant(bc_v, dtype=tf.float32)

    total_residual = residual_computer.compute_total_residual_with_bc(
        X_tf, Y_tf, bc_mask_tf, bc_u_tf, bc_v_tf, residual_weights
    )

    # Normalize by median (robust to heavy-tailed, right-skewed residuals)
    residual_flat = tf.reshape(total_residual, [-1])
    median = tf.sort(residual_flat)[tf.shape(residual_flat)[0] // 2]
    total_residual_norm = total_residual / (median + 1e-10)

    fluid_points = np.sum(layout)
    print(f"    Fluid points: {fluid_points:.0f}/{layout.size} ({100*np.mean(layout):.1f}%)")
    print(f"    Residual range: [{float(tf.reduce_min(total_residual_norm)):.4f}, "
          f"{float(tf.reduce_max(total_residual_norm)):.4f}]")

    return {
        'label': label,
        'inputs': tf.constant(inputs, dtype=tf.float32),
        'layout': tf.constant(layout, dtype=tf.float32),
        'residual': total_residual_norm,
        'X': X, 'Y': Y,
        'layout_np': layout,
        'cylinder_center': (cx, cy),
        'cylinder_radius': cr,
        'inlet_velocity': u_inlet,
        'pinn_path': pinn_path,
    }


def compute_total_variation(r_4d):
    """Compute total variation for spatial smoothness."""
    tv_h = tf.reduce_mean(tf.abs(r_4d[:, :, 1:, :] - r_4d[:, :, :-1, :]))
    tv_v = tf.reduce_mean(tf.abs(r_4d[:, 1:, :, :] - r_4d[:, :-1, :, :]))
    return tv_h + tv_v


@tf.function
def train_step_precomputed(router, optimizer, inputs, layout_mask, residual_norm,
                            beta, lambda_tv, lambda_entropy, grad_clip_norm):
    """
    One training step using pre-computed residuals.

    This avoids re-computing PINN residuals each step (they're constant
    since the PINN is frozen).

    Uses logistic loss: 1/N * sum(beta * softplus(s) + residual * softplus(-s))
    """
    with tf.GradientTape() as tape:
        s = router(inputs, training=True)
        s = s[0, :, :, 0]  # (H, W)

        num_fluid = tf.reduce_sum(layout_mask) + 1e-10

        # Logistic loss
        logistic_loss = tf.reduce_sum(
            (beta * tf.math.softplus(s) +
             residual_norm * tf.math.softplus(-s)) * layout_mask
        ) / num_fluid

        # Total variation
        s_masked = s * layout_mask
        s_4d = tf.reshape(s_masked, [1, tf.shape(s)[0], tf.shape(s)[1], 1])
        tv_loss = lambda_tv * compute_total_variation(s_4d)

        total_loss = logistic_loss + tv_loss

    gradients = tape.gradient(total_loss, router.trainable_variables)
    if grad_clip_norm > 0:
        gradients, _ = tf.clip_by_global_norm(gradients, grad_clip_norm)
    optimizer.apply_gradients(zip(gradients, router.trainable_variables))

    cfd_fraction = tf.reduce_sum(tf.cast(s > 0, tf.float32) * layout_mask) / num_fluid

    return {
        'total_loss': total_loss,
        'logistic_loss': logistic_loss,
        'tv_loss': tv_loss,
        'cfd_fraction': cfd_fraction,
    }


def evaluate_on_config(router, cfg_data, threshold=0.0):
    """Evaluate the router on a single config, return metrics."""
    inputs = cfg_data['inputs']
    layout = cfg_data['layout']

    r = router(inputs, training=False)
    r = r[0, :, :, 0].numpy()

    layout_np = cfg_data['layout_np']
    fluid_mask = layout_np == 1
    num_fluid = np.sum(fluid_mask)

    mask = (r >= threshold).astype(np.float32)
    cfd_fraction = np.sum(mask[fluid_mask]) / num_fluid * 100
    pinn_fraction = 100 - cfd_fraction

    # Residual in PINN region
    residual_np = cfg_data['residual'].numpy()
    pinn_region = (r < threshold) & fluid_mask
    if np.sum(pinn_region) > 0:
        mean_pinn_residual = np.mean(residual_np[pinn_region])
    else:
        mean_pinn_residual = 0.0

    return {
        'cfd_fraction': cfd_fraction,
        'pinn_fraction': pinn_fraction,
        'mean_pinn_residual': mean_pinn_residual,
        'router_mean': float(np.mean(r[fluid_mask])),
        'router_std': float(np.std(r[fluid_mask])),
        'router_output': r,
        'mask': mask,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Train CNN router on multiple cylinder configurations'
    )

    # Config
    parser.add_argument('--config', type=str, required=True,
                        help='Path to JSON config file with train/test configs')
    parser.add_argument('--output-dir', type=str, default='./router_output_multi',
                        help='Directory to save outputs')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of training epochs')
    parser.add_argument('--beta', type=float, default=1.0,
                        help='CFD cost coefficient')
    parser.add_argument('--lambda-tv', type=float, default=0.1,
                        help='Total variation regularization weight')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping norm (0 to disable)')

    # Residual weights
    parser.add_argument('--weight-continuity', type=float, default=1.0)
    parser.add_argument('--weight-momentum', type=float, default=1.0)
    # Domain parameters (shared across all configs)
    parser.add_argument('--nx', type=int, default=200)
    parser.add_argument('--ny', type=int, default=100)
    parser.add_argument('--x-min', type=float, default=0.0)
    parser.add_argument('--x-max', type=float, default=2.0)
    parser.add_argument('--y-min', type=float, default=0.0)
    parser.add_argument('--y-max', type=float, default=1.0)

    # Physical parameters
    parser.add_argument('--nu', type=float, default=0.01)
    parser.add_argument('--rho', type=float, default=1.0)

    # Router architecture
    parser.add_argument('--base-filters', type=int, default=32)

    # Inference
    parser.add_argument('--threshold', type=float, default=0.0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # =========================================================================
    # Load config
    # =========================================================================
    print("=" * 60)
    print("MULTI-CONFIG CNN ROUTER TRAINING")
    print("=" * 60)

    with open(args.config) as f:
        config = json.load(f)

    train_cfgs = config['train']
    test_cfgs = config.get('test', [])

    print(f"\nTraining configs: {len(train_cfgs)}")
    print(f"Test configs: {len(test_cfgs)}")

    x_domain = (args.x_min, args.x_max)
    y_domain = (args.y_min, args.y_max)

    residual_weights = {
        'continuity': args.weight_continuity,
        'momentum': args.weight_momentum,
    }

    # =========================================================================
    # Prepare all configs (load PINNs, compute residuals)
    # =========================================================================
    print("\n[Step 1] Preparing training configurations...")
    train_data = []
    for i, cfg in enumerate(train_cfgs):
        print(f"\n  Config {i+1}/{len(train_cfgs)}:")
        data = prepare_config(cfg, args.nx, args.ny, x_domain, y_domain,
                              args.nu, args.rho, residual_weights)
        train_data.append(data)

    test_data = []
    if test_cfgs:
        print("\n[Step 1b] Preparing test configurations...")
        for i, cfg in enumerate(test_cfgs):
            print(f"\n  Config {i+1}/{len(test_cfgs)}:")
            data = prepare_config(cfg, args.nx, args.ny, x_domain, y_domain,
                                  args.nu, args.rho, residual_weights)
            test_data.append(data)

    # =========================================================================
    # Initialize router
    # =========================================================================
    print("\n[Step 2] Initializing router CNN...")
    router = RouterCNN(base_filters=args.base_filters)

    # Build with first training config's input shape
    _ = router(train_data[0]['inputs'])
    print(f"  Parameters: {router.count_params():,}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)

    print(f"\n  beta={args.beta}, lambda_tv={args.lambda_tv}")
    print(f"  lr={args.lr}, grad_clip={args.grad_clip}")

    # =========================================================================
    # Train router (cycle through configs each epoch)
    # =========================================================================
    print("\n[Step 3] Training router...")
    print("-" * 60)

    start_time = datetime.now()
    n_train = len(train_data)

    # Convert hyperparams to tensors for tf.function
    beta_tf = tf.constant(args.beta, dtype=tf.float32)
    ltv_tf = tf.constant(args.lambda_tv, dtype=tf.float32)
    lent_tf = tf.constant(0.0, dtype=tf.float32)  # unused, kept for API compat
    gc_tf = tf.constant(args.grad_clip, dtype=tf.float32)

    history = {
        'total_loss': [], 'logistic_loss': [],
        'tv_loss': []
    }

    for epoch in range(args.epochs):
        epoch_losses = []

        # Cycle through all training configs
        for ci in range(n_train):
            d = train_data[ci]
            metrics = train_step_precomputed(
                router, optimizer,
                d['inputs'], d['layout'], d['residual'],
                beta_tf, ltv_tf, lent_tf, gc_tf
            )
            epoch_losses.append(float(metrics['total_loss']))

        # Record average loss across configs
        avg_loss = np.mean(epoch_losses)
        history['total_loss'].append(avg_loss)
        history['logistic_loss'].append(float(metrics['logistic_loss']))
        history['tv_loss'].append(float(metrics['tv_loss']))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} - "
                  f"AvgLoss: {avg_loss:.4f}, "
                  f"Logistic: {float(metrics['logistic_loss']):.4f}, "
                  f"TV: {float(metrics['tv_loss']):.4f}, "
                  f"CFD%: {float(metrics['cfd_fraction'])*100:.1f}%")

    training_time = (datetime.now() - start_time).total_seconds()
    print("-" * 60)
    print(f"Training completed in {training_time:.1f} seconds")

    # =========================================================================
    # Evaluate on training configs
    # =========================================================================
    print("\n[Step 4] Evaluating on training configs...")
    for i, d in enumerate(train_data):
        result = evaluate_on_config(router, d, args.threshold)
        print(f"  Train {i+1} ({d['label']}): "
              f"CFD={result['cfd_fraction']:.1f}%, "
              f"PINN={result['pinn_fraction']:.1f}%, "
              f"PINN_res={result['mean_pinn_residual']:.4f}")

    # =========================================================================
    # Evaluate on test configs
    # =========================================================================
    if test_data:
        print("\n[Step 5] Evaluating on TEST configs (unseen during training)...")
        test_results = []
        for i, d in enumerate(test_data):
            result = evaluate_on_config(router, d, args.threshold)
            test_results.append(result)
            print(f"  Test {i+1} ({d['label']}): "
                  f"CFD={result['cfd_fraction']:.1f}%, "
                  f"PINN={result['pinn_fraction']:.1f}%, "
                  f"PINN_res={result['mean_pinn_residual']:.4f}, "
                  f"router_mean={result['router_mean']:.4f}, "
                  f"router_std={result['router_std']:.4f}")

    # =========================================================================
    # Save results
    # =========================================================================
    print("\n[Step 6] Saving results...")

    # Router weights
    router_path = os.path.join(args.output_dir, 'router.weights.h5')
    router.save_weights(router_path)
    print(f"  Saved router weights: {router_path}")

    # Training history
    history_path = os.path.join(args.output_dir, 'training_history.npz')
    np.savez(history_path, **history)
    print(f"  Saved training history: {history_path}")

    # Save config used
    config_save_path = os.path.join(args.output_dir, 'config_used.json')
    with open(config_save_path, 'w') as f:
        json.dump({
            'configs': config,
            'args': vars(args),
            'training_time_s': training_time,
        }, f, indent=2)
    print(f"  Saved config: {config_save_path}")

    # =========================================================================
    # Visualize results
    # =========================================================================
    print("\n[Step 7] Generating visualizations...")

    # Plot training history
    history_plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(history, save_path=history_plot_path)

    # Plot router output for each training config
    for i, d in enumerate(train_data):
        result = evaluate_on_config(router, d, args.threshold)
        cx, cy = d['cylinder_center']
        cr = d['cylinder_radius']
        save_path = os.path.join(args.output_dir, f'router_train_{i}.png')
        plot_router_output(
            result['router_output'], d['X'], d['Y'], d['layout_np'],
            title=f'Train {i+1}: {d["label"]}',
            save_path=save_path,
            show_circle=(cx, cy, cr)
        )

    # Plot router output for each test config
    for i, d in enumerate(test_data):
        result = evaluate_on_config(router, d, args.threshold)
        cx, cy = d['cylinder_center']
        cr = d['cylinder_radius']
        save_path = os.path.join(args.output_dir, f'router_test_{i}.png')
        plot_router_output(
            result['router_output'], d['X'], d['Y'], d['layout_np'],
            title=f'TEST {i+1}: {d["label"]}',
            save_path=save_path,
            show_circle=(cx, cy, cr)
        )

        # Save test predictions
        pred_path = os.path.join(args.output_dir, f'predictions_test_{i}.npz')
        np.savez(pred_path,
                 router_output=result['router_output'],
                 mask=result['mask'],
                 X=d['X'], Y=d['Y'],
                 layout=d['layout_np'])

    print("\n" + "=" * 60)
    print("MULTI-CONFIG TRAINING COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - router.weights.h5: Trained router model")
    print(f"  - training_history.npz/png: Loss history")
    print(f"  - router_train_*.png: Router output on training configs")
    if test_data:
        print(f"  - router_test_*.png: Router output on TEST configs")
        print(f"  - predictions_test_*.npz: Test predictions")

    return router, history


if __name__ == "__main__":
    main()
