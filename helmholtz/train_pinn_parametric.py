"""
Train a parametric PINN for 2D Helmholtz on the square-with-hole domain.

Inputs: (x, y, k, x_s, y_s)  ->  u(x, y; k, x_s, y_s)
PDE:    -Delta u - k^2 u = f(x, y; x_s, y_s)   (Gaussian source, sigma fixed)
BCs:    homogeneous Dirichlet on outer + hole boundary.

Per step we sample a batch of configs (k, x_s, y_s) and a cloud of
(x, y) collocation + boundary points for each. Autodiff is taken w.r.t.
(x, y) only -- parameters are inputs, not differentiated.

Usage:
    python train_pinn_parametric.py --tag exp3_parametric --epochs 80000
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

from lib.network import build_parametric_pinn


def sample_configs(n, rng, k_range, xs_range, ys_range, hole):
    """Sample (k, x_s, y_s) triples, rejecting sources inside the hole."""
    k = rng.uniform(k_range[0], k_range[1], size=(n,)).astype(np.float32)
    cx, cy, r = hole
    out_xs = np.empty((n,), dtype=np.float32)
    out_ys = np.empty((n,), dtype=np.float32)
    filled = 0
    while filled < n:
        nb = 2 * (n - filled)
        xs_cand = rng.uniform(xs_range[0], xs_range[1], size=(nb,))
        ys_cand = rng.uniform(ys_range[0], ys_range[1], size=(nb,))
        d2 = (xs_cand - cx) ** 2 + (ys_cand - cy) ** 2
        keep = d2 > (r + 0.02) ** 2  # small margin so source is clearly outside
        xs_k = xs_cand[keep]; ys_k = ys_cand[keep]
        take = min(len(xs_k), n - filled)
        out_xs[filled:filled + take] = xs_k[:take]
        out_ys[filled:filled + take] = ys_k[:take]
        filled += take
    return k, out_xs, out_ys


def sample_interior(n, rng, hole):
    cx, cy, r = hole
    out = np.empty((n, 2), dtype=np.float32)
    filled = 0
    while filled < n:
        batch = rng.uniform(0.0, 1.0, size=(2 * (n - filled), 2))
        d2 = (batch[:, 0] - cx) ** 2 + (batch[:, 1] - cy) ** 2
        keep = batch[d2 > r ** 2]
        take = min(len(keep), n - filled)
        out[filled:filled + take] = keep[:take]
        filled += take
    return out


def sample_boundary(n, rng, hole):
    cx, cy, r = hole
    n_outer = n // 2
    n_hole = n - n_outer
    per_edge = n_outer // 4
    parts = []
    for edge in range(4):
        t = rng.uniform(0.0, 1.0, size=(per_edge,)).astype(np.float32)
        if edge == 0:
            xy = np.stack([t, np.zeros_like(t)], axis=-1)
        elif edge == 1:
            xy = np.stack([t, np.ones_like(t)], axis=-1)
        elif edge == 2:
            xy = np.stack([np.zeros_like(t), t], axis=-1)
        else:
            xy = np.stack([np.ones_like(t), t], axis=-1)
        parts.append(xy)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=(n_hole,)).astype(np.float32)
    xy_hole = np.stack(
        [cx + r * np.cos(theta), cy + r * np.sin(theta)], axis=-1
    ).astype(np.float32)
    parts.append(xy_hole)
    return np.concatenate(parts, axis=0)


def f_gaussian_tf(xy, x_s, y_s, sigma, amplitude):
    r2 = (xy[:, 0:1] - x_s) ** 2 + (xy[:, 1:2] - y_s) ** 2
    return amplitude * tf.exp(-r2 / (2.0 * sigma ** 2))


def compute_pde_residual(model, xyp, k_col, xs_col, ys_col, sigma, amp):
    """PDE residual Delta u + k^2 u + f at (x, y; k, x_s, y_s).

    xyp : (N, 5) tensor where columns are (x, y, k, x_s, y_s).
    Autodiff is w.r.t. x, y only (first two columns)."""
    xy = xyp[:, 0:2]
    params = xyp[:, 2:5]
    with tf.GradientTape() as t2:
        t2.watch(xy)
        with tf.GradientTape() as t1:
            t1.watch(xy)
            inp5 = tf.concat([xy, params], axis=-1)
            u = model(inp5, training=True)
        grads = t1.gradient(u, xy)
    hess = t2.batch_jacobian(grads, xy)
    uxx = hess[:, 0, 0:1]
    uyy = hess[:, 1, 1:2]
    f = f_gaussian_tf(xy, xs_col, ys_col, sigma, amp)
    return uxx + uyy + (k_col ** 2) * u + f


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=80000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--n-configs', type=int, default=16)
    parser.add_argument('--n-domain-per-config', type=int, default=1024)
    parser.add_argument('--n-boundary-per-config', type=int, default=256)
    parser.add_argument('--w-bc', type=float, default=100.0)
    parser.add_argument('--layers', type=int, nargs='+',
                        default=[256, 256, 256, 256, 256])
    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--fourier-m', type=int, default=128)
    parser.add_argument('--fourier-scale', type=float, default=3.0,
                        help='Std of B for k-modulated Fourier features. '
                             '~3 covers both wave-scale (|B|~1) and '
                             'source-scale (|B|~6) spatial structure.')
    parser.add_argument('--fourier-seed', type=int, default=0)
    parser.add_argument('--k-min', type=float, default=2.0 * np.pi)
    parser.add_argument('--k-max', type=float, default=6.0 * np.pi)
    parser.add_argument('--xs-min', type=float, default=0.05)
    parser.add_argument('--xs-max', type=float, default=0.95)
    parser.add_argument('--ys-min', type=float, default=0.05)
    parser.add_argument('--ys-max', type=float, default=0.95)
    parser.add_argument('--sigma', type=float, default=0.05)
    parser.add_argument('--amplitude', type=float, default=1.0)
    parser.add_argument('--hole-center-x', type=float, default=0.5)
    parser.add_argument('--hole-center-y', type=float, default=0.5)
    parser.add_argument('--hole-radius', type=float, default=0.15)
    parser.add_argument('--resample-every', type=int, default=1,
                        help='Re-sample configs this often (1 = every step).')
    parser.add_argument('--grad-clip', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tag', type=str, default='exp3_parametric')
    parser.add_argument('--output-dir', type=str, default='./models')
    parser.add_argument('--history-dir', type=str, default='./history')
    parser.add_argument('--print-every', type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.history_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)

    print("=" * 60)
    print(f"PARAMETRIC PINN  tag={args.tag}")
    print(f"  k in [{args.k_min:.3f}, {args.k_max:.3f}]  "
          f"x_s in [{args.xs_min}, {args.xs_max}]  "
          f"y_s in [{args.ys_min}, {args.ys_max}]")
    print(f"  configs/step={args.n_configs}  "
          f"n_domain={args.n_domain_per_config}  "
          f"n_bd={args.n_boundary_per_config}")
    print("=" * 60)

    fourier_scale = float(args.fourier_scale)
    model = build_parametric_pinn(
        layers=tuple(args.layers),
        activation=args.activation,
        spatial_range=((0.0, 1.0), (0.0, 1.0)),
        param_range=((args.k_min, args.k_max),
                     (args.xs_min, args.xs_max),
                     (args.ys_min, args.ys_max)),
        fourier_m=args.fourier_m,
        fourier_scale=fourier_scale,
        fourier_seed=args.fourier_seed)
    print(f"Params: {model.count_params():,}")

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr, decay_steps=args.epochs, alpha=1e-2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    hole = (args.hole_center_x, args.hole_center_y, args.hole_radius)
    k_range = (args.k_min, args.k_max)
    xs_range = (args.xs_min, args.xs_max)
    ys_range = (args.ys_min, args.ys_max)
    sigma_tf = tf.constant(args.sigma, dtype=tf.float32)
    amp_tf = tf.constant(args.amplitude, dtype=tf.float32)

    @tf.function
    def train_step(xyp_int, k_int, xs_int, ys_int,
                   xyp_bd):
        with tf.GradientTape() as tape:
            r = compute_pde_residual(model, xyp_int,
                                     k_int, xs_int, ys_int, sigma_tf, amp_tf)
            loss_pde = tf.reduce_mean(r ** 2)
            u_pred_bd = model(xyp_bd, training=True)
            loss_bc = tf.reduce_mean(u_pred_bd ** 2)
            total = loss_pde + args.w_bc * loss_bc
        grads = tape.gradient(total, model.trainable_variables)
        if args.grad_clip > 0:
            grads, _ = tf.clip_by_global_norm(grads, args.grad_clip)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total, loss_pde, loss_bc

    def build_batches():
        B = args.n_configs
        n_int = args.n_domain_per_config
        n_bd = args.n_boundary_per_config
        k_c, xs_c, ys_c = sample_configs(B, rng, k_range, xs_range, ys_range, hole)
        int_xy = np.concatenate(
            [sample_interior(n_int, rng, hole) for _ in range(B)], axis=0)
        bd_xy = np.concatenate(
            [sample_boundary(n_bd, rng, hole) for _ in range(B)], axis=0)
        k_int = np.repeat(k_c, n_int).astype(np.float32)[:, None]
        xs_int = np.repeat(xs_c, n_int).astype(np.float32)[:, None]
        ys_int = np.repeat(ys_c, n_int).astype(np.float32)[:, None]
        k_bd = np.repeat(k_c, n_bd).astype(np.float32)[:, None]
        xs_bd = np.repeat(xs_c, n_bd).astype(np.float32)[:, None]
        ys_bd = np.repeat(ys_c, n_bd).astype(np.float32)[:, None]
        xyp_int = np.concatenate([int_xy, k_int, xs_int, ys_int], axis=-1)
        xyp_bd = np.concatenate([bd_xy, k_bd, xs_bd, ys_bd], axis=-1)
        return (tf.constant(xyp_int), tf.constant(k_int), tf.constant(xs_int),
                tf.constant(ys_int), tf.constant(xyp_bd))

    history = {'total': [], 'pde': [], 'bc': []}
    t_train_start = time.perf_counter()
    cached = None
    for epoch in range(args.epochs):
        if cached is None or (epoch % args.resample_every == 0):
            cached = build_batches()
        xyp_int, k_int, xs_int, ys_int, xyp_bd = cached
        total, loss_pde, loss_bc = train_step(
            xyp_int, k_int, xs_int, ys_int, xyp_bd)
        history['total'].append(float(total))
        history['pde'].append(float(loss_pde))
        history['bc'].append(float(loss_bc))
        if (epoch + 1) % args.print_every == 0:
            print(f"Epoch {epoch+1}/{args.epochs}  "
                  f"total={float(total):.4e}  pde={float(loss_pde):.4e}  "
                  f"bc={float(loss_bc):.4e}")
    train_time_s = time.perf_counter() - t_train_start
    print(f"\nTraining time: {train_time_s:.1f}s")

    weights_path = os.path.join(
        args.output_dir, f'pinn_helmholtz_{args.tag}.weights.h5')
    model.save_weights(weights_path)
    print(f"Saved weights: {weights_path}")

    hist_path = os.path.join(args.history_dir, f'history_{args.tag}.npz')
    np.savez(hist_path, **{k: np.array(v) for k, v in history.items()})
    print(f"Saved history: {hist_path}")

    meta_path = os.path.join(args.history_dir,
                             f'training_meta_{args.tag}.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'parametric': True,
            'k_min': args.k_min, 'k_max': args.k_max,
            'xs_min': args.xs_min, 'xs_max': args.xs_max,
            'ys_min': args.ys_min, 'ys_max': args.ys_max,
            'sigma': args.sigma, 'amplitude': args.amplitude,
            'hole_center_x': args.hole_center_x,
            'hole_center_y': args.hole_center_y,
            'hole_radius': args.hole_radius,
            'epochs': args.epochs, 'lr': args.lr,
            'n_configs': args.n_configs,
            'n_domain_per_config': args.n_domain_per_config,
            'n_boundary_per_config': args.n_boundary_per_config,
            'resample_every': args.resample_every,
            'layers': args.layers, 'w_bc': args.w_bc,
            'fourier_m': args.fourier_m,
            'fourier_scale': fourier_scale,
            'train_time_s': train_time_s,
            'tag': args.tag,
        }, f, indent=2)
    print(f"Saved training meta: {meta_path}")


if __name__ == '__main__':
    main()
