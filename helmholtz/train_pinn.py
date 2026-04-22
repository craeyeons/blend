"""
Train a scalar PINN for the 2D Helmholtz equation

    -Delta u - k^2 u = f   on [0, 1]^2
    u = u_star            on boundary

with u_star = sin(k x) sin(k y), f = k^2 u_star (manufactured).

Usage:
    python train_pinn.py --k 12.566 --epochs 20000 --tag k4pi
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

from lib.network import build_pinn


def sample_interior(n, rng, hole=None):
    """Uniform samples in [0,1]^2; if `hole=(cx, cy, r)`, rejection-sample
    to exclude points inside the circular hole."""
    if hole is None:
        return rng.uniform(0.0, 1.0, size=(n, 2)).astype(np.float32)
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


def sample_boundary(n, rng, hole=None):
    """Uniformly sample outer-square boundary (4 edges) and, if hole, the
    hole circle. Total returned ~ `n` (may round down slightly)."""
    if hole is None:
        per_edge = n // 4
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
        return np.concatenate(parts, axis=0)

    cx, cy, r = hole
    # Split: half on outer boundary (4 edges), half on hole circle.
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


def u_star_tf(xy, k):
    return tf.sin(k * xy[:, 0:1]) * tf.sin(k * xy[:, 1:2])


def f_manufactured_tf(xy, k):
    return (k ** 2) * u_star_tf(xy, k)


def f_gaussian_tf(xy, x_s, y_s, sigma, amplitude):
    r2 = (xy[:, 0:1] - x_s) ** 2 + (xy[:, 1:2] - y_s) ** 2
    return amplitude * tf.exp(-r2 / (2.0 * sigma ** 2))


def compute_pde_residual(model, xy, k, f_fn):
    """Residual of Delta u + k^2 u + f = 0 (equivalently -Delta u - k^2 u = f)."""
    with tf.GradientTape() as t2:
        t2.watch(xy)
        with tf.GradientTape() as t1:
            t1.watch(xy)
            u = model(xy, training=True)
        grads = t1.gradient(u, xy)  # (N, 2)
    hess = t2.batch_jacobian(grads, xy)  # (N, 2, 2)
    uxx = hess[:, 0, 0:1]
    uyy = hess[:, 1, 1:2]
    f = f_fn(xy)
    return uxx + uyy + (k ** 2) * u + f


def rmse(pred, exact):
    return float(np.sqrt(np.mean((pred - exact) ** 2)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=float, required=True,
                        help='Wavenumber')
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--n-domain', type=int, default=20000)
    parser.add_argument('--n-boundary', type=int, default=4000)
    parser.add_argument('--w-bc', type=float, default=100.0)
    parser.add_argument('--layers', type=int, nargs='+',
                        default=[256, 256, 256, 256, 256])
    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--fourier-m', type=int, default=128,
                        help='Number of Fourier features (0 disables)')
    parser.add_argument('--fourier-scale', type=float, default=None,
                        help='Std of Fourier freq matrix B. '
                             'Defaults to k/(2 pi) when None.')
    parser.add_argument('--fourier-seed', type=int, default=0)
    parser.add_argument('--source', type=str, default='manufactured',
                        choices=['manufactured', 'gaussian'])
    parser.add_argument('--x-s', type=float, default=0.5)
    parser.add_argument('--y-s', type=float, default=0.5)
    parser.add_argument('--sigma', type=float, default=0.05)
    parser.add_argument('--amplitude', type=float, default=1.0)
    parser.add_argument('--domain', type=str, default='square',
                        choices=['square', 'square_hole'])
    parser.add_argument('--hole-center-x', type=float, default=0.5)
    parser.add_argument('--hole-center-y', type=float, default=0.5)
    parser.add_argument('--hole-radius', type=float, default=0.15)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tag', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default='./models')
    parser.add_argument('--history-dir', type=str, default='./history')
    parser.add_argument('--print-every', type=int, default=500)
    args = parser.parse_args()

    tag = args.tag or f'k{args.k:.3f}'.replace('.', 'p')
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.history_dir, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)

    print("=" * 60)
    print(f"PINN TRAINING: Helmholtz  k={args.k:.4f}  tag={tag}")
    print("=" * 60)

    fourier_scale = (args.fourier_scale if args.fourier_scale is not None
                     else args.k / (2.0 * np.pi))
    model = build_pinn(num_inputs=2, layers=tuple(args.layers),
                       activation=args.activation,
                       input_range=((0.0, 1.0), (0.0, 1.0)),
                       fourier_m=args.fourier_m,
                       fourier_scale=fourier_scale,
                       fourier_seed=args.fourier_seed)
    print(f"Params: {model.count_params():,}")

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=args.lr, decay_steps=args.epochs, alpha=1e-2)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    k_tf = tf.constant(args.k, dtype=tf.float32)
    if args.source == 'manufactured':
        def f_fn(xy):
            return f_manufactured_tf(xy, k_tf)
        def bc_fn(xy_bd):
            return u_star_tf(xy_bd, k_tf)
    else:
        xs_tf = tf.constant(args.x_s, dtype=tf.float32)
        ys_tf = tf.constant(args.y_s, dtype=tf.float32)
        sigma_tf = tf.constant(args.sigma, dtype=tf.float32)
        amp_tf = tf.constant(args.amplitude, dtype=tf.float32)
        def f_fn(xy):
            return f_gaussian_tf(xy, xs_tf, ys_tf, sigma_tf, amp_tf)
        def bc_fn(xy_bd):
            return tf.zeros((tf.shape(xy_bd)[0], 1), dtype=tf.float32)

    @tf.function
    def train_step(xy_int, xy_bd, u_bd):
        with tf.GradientTape() as tape:
            r = compute_pde_residual(model, xy_int, k_tf, f_fn)
            loss_pde = tf.reduce_mean(r ** 2)
            u_pred_bd = model(xy_bd, training=True)
            loss_bc = tf.reduce_mean((u_pred_bd - u_bd) ** 2)
            total = loss_pde + args.w_bc * loss_bc
        grads = tape.gradient(total, model.trainable_variables)
        if args.grad_clip > 0:
            grads, _ = tf.clip_by_global_norm(grads, args.grad_clip)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total, loss_pde, loss_bc

    hole = None
    if args.domain == 'square_hole':
        hole = (args.hole_center_x, args.hole_center_y, args.hole_radius)
        # In hole mode, BC target is always zero (homogeneous Dirichlet),
        # regardless of --source setting.
        def bc_fn(xy_bd):
            return tf.zeros((tf.shape(xy_bd)[0], 1), dtype=tf.float32)

    history = {'total': [], 'pde': [], 'bc': []}
    t_train_start = time.perf_counter()
    for epoch in range(args.epochs):
        xy_int = tf.constant(sample_interior(args.n_domain, rng, hole=hole))
        xy_bd = tf.constant(sample_boundary(args.n_boundary, rng, hole=hole))
        u_bd = bc_fn(xy_bd)

        total, loss_pde, loss_bc = train_step(xy_int, xy_bd, u_bd)
        history['total'].append(float(total))
        history['pde'].append(float(loss_pde))
        history['bc'].append(float(loss_bc))

        if (epoch + 1) % args.print_every == 0:
            print(f"Epoch {epoch+1}/{args.epochs}  "
                  f"total={float(total):.4e}  pde={float(loss_pde):.4e}  "
                  f"bc={float(loss_bc):.4e}")
    train_time_s = time.perf_counter() - t_train_start
    print(f"\nTraining time: {train_time_s:.1f}s")

    # Final accuracy on a 201x201 grid vs u_star.
    N = 201
    xs = np.linspace(0.0, 1.0, N, dtype=np.float32)
    X, Y = np.meshgrid(xs, xs)
    xy_eval = np.stack([X.ravel(), Y.ravel()], axis=-1)
    t0 = time.perf_counter()
    u_pred = model.predict(xy_eval, batch_size=len(xy_eval), verbose=0)
    infer_time_s = time.perf_counter() - t0
    u_pred = u_pred.reshape(N, N)
    print(f"PINN inference time: {infer_time_s:.4f}s")
    if args.source == 'manufactured':
        u_exact = np.sin(args.k * X) * np.sin(args.k * Y)
        err = rmse(u_pred, u_exact)
        print(f"PINN RMSE vs u* on 201x201 grid: {err:.4e}")
    else:
        err = None
        print("Gaussian source: no analytic u*; use plot_sanity.py "
              "to compare against FEM.")

    weights_path = os.path.join(
        args.output_dir, f'pinn_helmholtz_{tag}.weights.h5')
    model.save_weights(weights_path)
    print(f"Saved weights: {weights_path}")

    hist_path = os.path.join(args.history_dir, f'history_{tag}.npz')
    np.savez(hist_path, **{k: np.array(v) for k, v in history.items()})
    print(f"Saved history: {hist_path}")

    meta_path = os.path.join(args.history_dir, f'training_meta_{tag}.json')
    with open(meta_path, 'w') as f:
        json.dump({
            'k': args.k,
            'epochs': args.epochs,
            'lr': args.lr,
            'layers': args.layers,
            'w_bc': args.w_bc,
            'fourier_m': args.fourier_m,
            'fourier_scale': fourier_scale,
            'train_time_s': train_time_s,
            'infer_time_s': infer_time_s,
            'final_rmse': err,
            'source': args.source,
            'x_s': args.x_s,
            'y_s': args.y_s,
            'sigma': args.sigma,
            'amplitude': args.amplitude,
            'domain': args.domain,
            'hole_center_x': args.hole_center_x,
            'hole_center_y': args.hole_center_y,
            'hole_radius': args.hole_radius,
            'tag': tag,
        }, f, indent=2)
    print(f"Saved training meta: {meta_path}")


if __name__ == '__main__':
    main()
