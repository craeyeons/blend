"""
Experiment 1 sanity plot: PINN vs FEM vs exact for manufactured Helmholtz.

Usage:
    python plot_sanity.py --k 12.566 --tag k4pi
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

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from lib.network import build_pinn


def rel_l2(pred, exact):
    return float(np.sqrt(np.mean((pred - exact) ** 2))
                 / (np.sqrt(np.mean(exact ** 2)) + 1e-12))


def rel_h1(pred, exact, dx, dy):
    # Central differences for gradient; fall back to forward at boundary.
    gx_pred = np.gradient(pred, dx, axis=1)
    gy_pred = np.gradient(pred, dy, axis=0)
    gx_ex = np.gradient(exact, dx, axis=1)
    gy_ex = np.gradient(exact, dy, axis=0)
    num = np.sqrt(np.mean((pred - exact) ** 2)
                  + np.mean((gx_pred - gx_ex) ** 2)
                  + np.mean((gy_pred - gy_ex) ** 2))
    den = np.sqrt(np.mean(exact ** 2)
                  + np.mean(gx_ex ** 2) + np.mean(gy_ex ** 2))
    return float(num / (den + 1e-12))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--k', type=float, required=True)
    p.add_argument('--tag', type=str, default=None)
    p.add_argument('--models-dir', type=str, default='./models')
    p.add_argument('--results-dir', type=str, default='./results')
    p.add_argument('--history-dir', type=str, default='./history')
    p.add_argument('--output-dir', type=str, default='./sanity_plots')
    p.add_argument('--layers', type=int, nargs='+', default=[64, 64, 64, 64])
    p.add_argument('--activation', type=str, default='tanh')
    args = p.parse_args()

    tag = args.tag or f'k{args.k:.3f}'.replace('.', 'p')
    os.makedirs(args.output_dir, exist_ok=True)

    fem_path = os.path.join(args.results_dir, f'fem_helmholtz_{tag}.npz')
    pinn_path = os.path.join(args.models_dir, f'pinn_helmholtz_{tag}.weights.h5')
    meta_path = os.path.join(args.history_dir, f'training_meta_{tag}.json')

    d = np.load(fem_path)
    X, Y = d['X'], d['Y']
    u_fem = d['u_fem']
    u_exact = d['u_exact']
    k = float(d['k'])
    fem_solve_time_s = float(d['solve_time_s'])

    model = build_pinn(num_inputs=2, layers=tuple(args.layers),
                       activation=args.activation,
                       input_range=((0.0, 1.0), (0.0, 1.0)))
    _ = model(tf.zeros((1, 2)))
    model.load_weights(pinn_path)

    xy = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)
    t0 = time.perf_counter()
    u_pinn = model.predict(xy, batch_size=len(xy), verbose=0).reshape(X.shape)
    pinn_infer_time_s = time.perf_counter() - t0

    dx = float(X[0, 1] - X[0, 0])
    dy = float(Y[1, 0] - Y[0, 0])

    pinn_l2 = rel_l2(u_pinn, u_exact)
    fem_l2 = rel_l2(u_fem, u_exact)
    pinn_h1 = rel_h1(u_pinn, u_exact, dx, dy)
    fem_h1 = rel_h1(u_fem, u_exact, dx, dy)

    pinn_train_time_s = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        pinn_train_time_s = meta.get('train_time_s')

    metrics = {
        'k': k,
        'tag': tag,
        'pinn_L2_rel': pinn_l2,
        'pinn_H1_rel': pinn_h1,
        'fem_L2_rel': fem_l2,
        'fem_H1_rel': fem_h1,
        'pinn_train_time_s': pinn_train_time_s,
        'pinn_infer_time_s': pinn_infer_time_s,
        'fem_solve_time_s': fem_solve_time_s,
    }

    metrics_path = os.path.join(args.output_dir, f'metrics_{tag}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # Plot: u_exact, PINN error, FEM error
    err_pinn = u_pinn - u_exact
    err_fem = u_fem - u_exact
    emax = max(np.abs(err_pinn).max(), np.abs(err_fem).max(), 1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    im0 = axes[0].pcolormesh(X, Y, u_exact, shading='auto', cmap='RdBu_r')
    axes[0].set_title(f'u* = sin(kx)sin(ky),  k={k:.3f}')
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(X, Y, err_pinn, shading='auto',
                             cmap='RdBu_r', vmin=-emax, vmax=emax)
    axes[1].set_title(f'PINN - u*  (rel L2 = {pinn_l2:.2e})')
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(X, Y, err_fem, shading='auto',
                             cmap='RdBu_r', vmin=-emax, vmax=emax)
    axes[2].set_title(f'FEM - u*  (rel L2 = {fem_l2:.2e})')
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_aspect('equal')
        ax.set_xlabel('x'); ax.set_ylabel('y')

    fig.suptitle(f'Helmholtz manufactured solution (tag={tag})', y=1.02)
    fig.tight_layout()
    out_png = os.path.join(args.output_dir, f'plot_sanity_{tag}.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: {out_png}")


if __name__ == '__main__':
    main()
