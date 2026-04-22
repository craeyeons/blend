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


def sample_interior(n, rng):
    xy = rng.uniform(0.0, 1.0, size=(n, 2)).astype(np.float32)
    return xy


def sample_boundary(n, rng):
    per_edge = n // 4
    parts = []
    # bottom (y=0), top (y=1), left (x=0), right (x=1)
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


def u_star_tf(xy, k):
    return tf.sin(k * xy[:, 0:1]) * tf.sin(k * xy[:, 1:2])


def f_source_tf(xy, k):
    return (k ** 2) * u_star_tf(xy, k)


def compute_pde_residual(model, xy, k):
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
    f = f_source_tf(xy, k)
    return uxx + uyy + (k ** 2) * u + f


def relative_l2(pred, exact):
    num = np.sqrt(np.mean((pred - exact) ** 2))
    den = np.sqrt(np.mean(exact ** 2))
    return float(num / (den + 1e-12))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=float, required=True,
                        help='Wavenumber')
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--n-domain', type=int, default=10000)
    parser.add_argument('--n-boundary', type=int, default=2000)
    parser.add_argument('--w-bc', type=float, default=100.0)
    parser.add_argument('--layers', type=int, nargs='+',
                        default=[128, 128, 128, 128])
    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--fourier-m', type=int, default=64,
                        help='Number of Fourier features (0 disables)')
    parser.add_argument('--fourier-scale', type=float, default=None,
                        help='Std of Fourier freq matrix B. '
                             'Defaults to k/(2 pi) when None.')
    parser.add_argument('--fourier-seed', type=int, default=0)
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

    @tf.function
    def train_step(xy_int, xy_bd, u_bd):
        with tf.GradientTape() as tape:
            r = compute_pde_residual(model, xy_int, k_tf)
            loss_pde = tf.reduce_mean(r ** 2)
            u_pred_bd = model(xy_bd, training=True)
            loss_bc = tf.reduce_mean((u_pred_bd - u_bd) ** 2)
            total = loss_pde + args.w_bc * loss_bc
        grads = tape.gradient(total, model.trainable_variables)
        if args.grad_clip > 0:
            grads, _ = tf.clip_by_global_norm(grads, args.grad_clip)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return total, loss_pde, loss_bc

    history = {'total': [], 'pde': [], 'bc': []}
    t_train_start = time.perf_counter()
    for epoch in range(args.epochs):
        xy_int = tf.constant(sample_interior(args.n_domain, rng))
        xy_bd = tf.constant(sample_boundary(args.n_boundary, rng))
        u_bd = u_star_tf(xy_bd, k_tf)

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
    u_exact = np.sin(args.k * X) * np.sin(args.k * Y)
    rel_l2 = relative_l2(u_pred, u_exact)
    print(f"PINN inference time: {infer_time_s:.4f}s")
    print(f"PINN relative L2 error on 201x201 grid: {rel_l2:.4e}")

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
            'final_rel_l2': rel_l2,
            'tag': tag,
        }, f, indent=2)
    print(f"Saved training meta: {meta_path}")


if __name__ == '__main__':
    main()
