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


def rmse(pred, exact):
    return float(np.sqrt(np.mean((pred - exact) ** 2)))


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
    k = float(d['k'])
    fem_solve_time_s = float(d['solve_time_s'])
    source = str(d['source']) if 'source' in d.files else 'manufactured'
    has_exact = 'u_exact' in d.files
    u_exact = d['u_exact'] if has_exact else None
    # When no analytic solution (e.g. gaussian source), FEM is the reference
    # for PINN error; FEM error is reported as NaN.
    u_ref = u_exact if has_exact else u_fem

    fourier_m = 64
    fourier_scale = k / (2.0 * np.pi)
    layers = args.layers
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            _meta_preview = json.load(f)
        fourier_m = int(_meta_preview.get('fourier_m', fourier_m))
        fourier_scale = float(_meta_preview.get('fourier_scale', fourier_scale))
        layers = list(_meta_preview.get('layers', layers))

    model = build_pinn(num_inputs=2, layers=tuple(layers),
                       activation=args.activation,
                       input_range=((0.0, 1.0), (0.0, 1.0)),
                       fourier_m=fourier_m,
                       fourier_scale=fourier_scale)
    _ = model(tf.zeros((1, 2)))
    model.load_weights(pinn_path)

    xy = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)
    t0 = time.perf_counter()
    u_pinn = model.predict(xy, batch_size=len(xy), verbose=0).reshape(X.shape)
    pinn_infer_time_s = time.perf_counter() - t0

    dx = float(X[0, 1] - X[0, 0])
    dy = float(Y[1, 0] - Y[0, 0])

    pinn_l2 = rmse(u_pinn, u_ref)
    pinn_h1 = rel_h1(u_pinn, u_ref, dx, dy)
    if has_exact:
        fem_l2 = rmse(u_fem, u_exact)
        fem_h1 = rel_h1(u_fem, u_exact, dx, dy)
    else:
        fem_l2 = float('nan')
        fem_h1 = float('nan')

    pinn_train_time_s = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        pinn_train_time_s = meta.get('train_time_s')

    metrics = {
        'k': k,
        'tag': tag,
        'pinn_rmse': pinn_l2,
        'pinn_H1_rel': pinn_h1,
        'fem_rmse': fem_l2,
        'fem_H1_rel': fem_h1,
        'pinn_train_time_s': pinn_train_time_s,
        'pinn_infer_time_s': pinn_infer_time_s,
        'fem_solve_time_s': fem_solve_time_s,
    }

    metrics_path = os.path.join(args.output_dir, f'metrics_{tag}.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # Plot: top row = solutions (reference, PINN, FEM) on shared scale;
    # bottom row = errors against reference.
    err_pinn = u_pinn - u_ref
    if has_exact:
        err_fem = u_fem - u_exact
        emax = max(np.abs(err_pinn).max(), np.abs(err_fem).max(), 1e-12)
    else:
        err_fem = None
        emax = max(float(np.abs(err_pinn).max()), 1e-12)
    umax = max(np.abs(u_ref).max(), np.abs(u_pinn).max(),
               np.abs(u_fem).max(), 1e-12)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8.4))

    ref_title = (f'u* = sin(kx)sin(ky),  k={k:.3f}' if has_exact
                 else f'FEM u (reference),  k={k:.3f}')
    im00 = axes[0, 0].pcolormesh(X, Y, u_ref, shading='auto',
                                 cmap='RdBu_r', vmin=-umax, vmax=umax)
    axes[0, 0].set_title(ref_title)
    plt.colorbar(im00, ax=axes[0, 0])

    im01 = axes[0, 1].pcolormesh(X, Y, u_pinn, shading='auto',
                                 cmap='RdBu_r', vmin=-umax, vmax=umax)
    axes[0, 1].set_title('PINN u')
    plt.colorbar(im01, ax=axes[0, 1])

    im02 = axes[0, 2].pcolormesh(X, Y, u_fem, shading='auto',
                                 cmap='RdBu_r', vmin=-umax, vmax=umax)
    axes[0, 2].set_title('FEM u')
    plt.colorbar(im02, ax=axes[0, 2])

    axes[1, 0].axis('off')

    ref_lbl = 'u*' if has_exact else 'u_FEM'
    im11 = axes[1, 1].pcolormesh(X, Y, err_pinn, shading='auto',
                                 cmap='RdBu_r', vmin=-emax, vmax=emax)
    axes[1, 1].set_title(f'PINN - {ref_lbl}  (RMSE = {pinn_l2:.2e})')
    plt.colorbar(im11, ax=axes[1, 1])

    if has_exact:
        im12 = axes[1, 2].pcolormesh(X, Y, err_fem, shading='auto',
                                     cmap='RdBu_r', vmin=-emax, vmax=emax)
        axes[1, 2].set_title(f'FEM - u*  (RMSE = {fem_l2:.2e})')
        plt.colorbar(im12, ax=axes[1, 2])
    else:
        axes[1, 2].axis('off')

    for ax in axes.ravel():
        if ax.has_data():
            ax.set_aspect('equal')
            ax.set_xlabel('x'); ax.set_ylabel('y')

    title_src = 'manufactured' if has_exact else 'gaussian source'
    fig.suptitle(f'Helmholtz {title_src} (tag={tag})', y=1.00)
    fig.tight_layout()
    out_png = os.path.join(args.output_dir, f'plot_sanity_{tag}.png')
    fig.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: {out_png}")


if __name__ == '__main__':
    main()
