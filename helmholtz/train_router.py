"""
Train the Helmholtz router on a single (PINN, geometry, k, source) config.

Usage:
    python train_router.py --k 12.566 --tag exp2_hole_k4pi \
        --domain square_hole --source gaussian --x-s 0.7 --y-s 0.5 \
        --hole-radius 0.15 --epochs 2000 --beta 0.1
"""

import argparse
import json
import os
import time

import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

from lib.domains import (create_square, create_square_with_hole,
                         gaussian_source, manufactured_solution)
from lib.network import build_pinn
from lib.router import (RouterCNN, RouterTrainer,
                        HelmholtzResidualComputer, create_router_input,
                        median_normalize, compute_ete_fft)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=float, required=True)
    p.add_argument('--tag', type=str, required=True)
    p.add_argument('--domain', type=str, default='square_hole',
                   choices=['square', 'square_hole'])
    p.add_argument('--source', type=str, default='gaussian',
                   choices=['manufactured', 'gaussian'])
    p.add_argument('--x-s', type=float, default=0.7)
    p.add_argument('--y-s', type=float, default=0.5)
    p.add_argument('--sigma', type=float, default=0.05)
    p.add_argument('--amplitude', type=float, default=1.0)
    p.add_argument('--hole-center-x', type=float, default=0.5)
    p.add_argument('--hole-center-y', type=float, default=0.5)
    p.add_argument('--hole-radius', type=float, default=0.15)
    p.add_argument('--nx', type=int, default=201)
    p.add_argument('--ny', type=int, default=201)
    p.add_argument('--pinn-dir', type=str, default='./models')
    p.add_argument('--pinn-meta-dir', type=str, default='./history')
    p.add_argument('--output-dir', type=str, default='./router_models')
    p.add_argument('--history-dir', type=str, default='./router_history')
    p.add_argument('--epochs', type=int, default=2000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr-min', type=float, default=1e-5)
    p.add_argument('--beta', type=float, default=0.1,
                   help='Cost coefficient (higher = less FEM). Default 0.1 '
                        'for R = normalize(|r| + |e|) (median = 1).')
    p.add_argument('--lambda-tv', type=float, default=0.01)
    p.add_argument('--base-filters', type=int, default=32)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.history_dir, exist_ok=True)

    print("=" * 60)
    print(f"ROUTER TRAINING  tag={args.tag}  k={args.k:.4f}")
    print("=" * 60)

    # Grid + layout
    if args.domain == 'square_hole':
        X, Y, layout, _, _ = create_square_with_hole(
            Nx=args.nx, Ny=args.ny,
            hole_center=(args.hole_center_x, args.hole_center_y),
            hole_radius=args.hole_radius)
    else:
        X, Y, layout, _, _ = create_square(Nx=args.nx, Ny=args.ny)

    # Forcing on grid
    if args.source == 'gaussian':
        f_grid = gaussian_source(X, Y, x_s=args.x_s, y_s=args.y_s,
                                 sigma=args.sigma, amplitude=args.amplitude)
    else:
        _, f_grid = manufactured_solution(X, Y, args.k)

    # Load PINN (architecture from its training meta)
    meta_path = os.path.join(args.pinn_meta_dir, f'training_meta_{args.tag}.json')
    with open(meta_path) as f:
        pinn_meta = json.load(f)
    layers_cfg = pinn_meta.get('layers', [256, 256, 256, 256, 256])
    fourier_m = int(pinn_meta.get('fourier_m', 64))
    fourier_scale = float(pinn_meta.get('fourier_scale', args.k / (2 * np.pi)))

    pinn = build_pinn(num_inputs=2, layers=tuple(layers_cfg),
                      activation='tanh',
                      input_range=((0.0, 1.0), (0.0, 1.0)),
                      fourier_m=fourier_m, fourier_scale=fourier_scale)
    _ = pinn(tf.zeros((1, 2)))
    pinn_weights = os.path.join(args.pinn_dir, f'pinn_helmholtz_{args.tag}.weights.h5')
    pinn.load_weights(pinn_weights)
    print(f"Loaded PINN from {pinn_weights}")

    # PINN prediction on grid
    xy = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)
    pinn_u = pinn.predict(xy, batch_size=len(xy), verbose=0).reshape(X.shape)
    # Mask hole interior to zero so it doesn't pollute channel normalization.
    pinn_u = pinn_u * layout

    # PDE residual (signed, so we can feed it to the FFT ETE; |r| is the
    # router channel and the training label).
    residual_comp = HelmholtzResidualComputer(pinn, args.k)
    t0 = time.perf_counter()
    residual_signed = residual_comp.compute_signed_residual(X, Y, f_grid)
    residual_signed = residual_signed * layout
    residual = np.abs(residual_signed)
    residual_time_s = time.perf_counter() - t0
    print(f"Residual field time: {residual_time_s:.3f}s  "
          f"median (solid)={float(np.median(residual[layout>0])):.3e}")

    # FFT-based ETE channel (free-space pseudo-inverse of L e = r).
    t0 = time.perf_counter()
    ete = compute_ete_fft(residual_signed, args.k, layout=layout)
    ete_time_s = time.perf_counter() - t0
    print(f"FFT-ETE channel time: {ete_time_s*1000:.2f}ms  "
          f"median (solid)={float(np.median(ete[layout>0])):.3e}")

    # Router input (5 channels: layout, f, u, |r|, |e_ete|)
    inputs = create_router_input(layout, f_grid, pinn_u, residual, ete=ete)

    # R = normalize(|r| + |e_ete|): sum first, then median-normalize on solid.
    residual_label = median_normalize(residual + ete, layout).astype(np.float32)

    # Router
    router = RouterCNN(base_filters=args.base_filters)
    _ = router(tf.constant(inputs, dtype=tf.float32))  # build
    print(f"Router params: {router.count_params():,}")

    trainer = RouterTrainer(router, pinn, args.k,
                            beta=args.beta, lambda_tv=args.lambda_tv)

    t0 = time.perf_counter()
    history = trainer.train(inputs, residual_label, layout,
                            epochs=args.epochs, lr=args.lr,
                            lr_min=args.lr_min, verbose=True)
    train_time_s = time.perf_counter() - t0

    weights_path = os.path.join(
        args.output_dir, f'router_helmholtz_{args.tag}.weights.h5')
    router.save_weights(weights_path)
    print(f"Saved router weights: {weights_path}")

    hist_path = os.path.join(args.history_dir, f'router_history_{args.tag}.npz')
    np.savez(hist_path, **{k: np.array(v) for k, v in history.items()})

    meta_out = os.path.join(args.history_dir, f'router_meta_{args.tag}.json')
    with open(meta_out, 'w') as f:
        json.dump({
            'k': args.k,
            'tag': args.tag,
            'domain': args.domain,
            'source': args.source,
            'x_s': args.x_s, 'y_s': args.y_s,
            'sigma': args.sigma, 'amplitude': args.amplitude,
            'hole_center_x': args.hole_center_x,
            'hole_center_y': args.hole_center_y,
            'hole_radius': args.hole_radius,
            'epochs': args.epochs,
            'lr': args.lr,
            'beta': args.beta,
            'lambda_tv': args.lambda_tv,
            'base_filters': args.base_filters,
            'train_time_s': train_time_s,
            'residual_time_s': residual_time_s,
        }, f, indent=2)
    print(f"Saved router meta: {meta_out}")


if __name__ == '__main__':
    main()
