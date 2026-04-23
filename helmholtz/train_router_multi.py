"""
Train a single router across many (k, x_s, y_s) configs for Exp 3.

Workflow:
  1. Load parametric PINN weights.
  2. Load configs JSON, filter to split='id' (training configs).
  3. For each training config: build router input (5 channels: layout,
     f_source, pinn_u, |r|/median, |e|/median) and router target
     R(x) = normalize(|r| + |e|) on solid cells.
  4. Train one RouterCNN by cycling through configs each step, random
     shuffle within epoch.

Usage:
    python train_router_multi.py --pinn-tag exp3_parametric \
        --configs configs_exp3.json --tag exp3 --epochs 4000 --beta 0.1
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

from lib.domains import create_square_with_hole, gaussian_source
from lib.network import build_parametric_pinn
from lib.router import (RouterCNN, HelmholtzResidualComputer,
                        create_router_input, compute_ete_fft,
                        median_normalize)


def _build_config_tensors(pinn, rcomp_fn, X, Y, layout, cfg,
                          sigma, amplitude):
    """Return (router_inputs, target_R) for one config.

    router_inputs : (1, Ny, Nx, 5) float32
    target_R      : (Ny, Nx) float32, = median_normalize(|r|+|e|).
    """
    k = float(cfg['k']); xs = float(cfg['x_s']); ys = float(cfg['y_s'])

    f_grid = gaussian_source(X, Y, x_s=xs, y_s=ys,
                             sigma=sigma, amplitude=amplitude)
    # PINN on grid at this config.
    N = X.size
    xy = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)
    params = np.stack([
        np.full(N, k, dtype=np.float32),
        np.full(N, xs, dtype=np.float32),
        np.full(N, ys, dtype=np.float32),
    ], axis=-1)
    xyp = np.concatenate([xy, params], axis=-1)
    pinn_u_flat = pinn.predict(xyp, batch_size=len(xyp), verbose=0)
    pinn_u = pinn_u_flat.reshape(X.shape) * layout

    # Residual via parametric autodiff: we need a residual computer that
    # takes the parametric PINN. We close over (k, xs, ys) to form a
    # closure that looks like the scalar PINN from outside.
    rcomp = rcomp_fn(pinn, k, xs, ys)
    signed_r = rcomp.compute_signed_residual(X, Y, f_grid) * layout
    residual = np.abs(signed_r)
    ete = compute_ete_fft(signed_r, k, layout=layout)

    router_inputs = create_router_input(
        layout, f_grid, pinn_u, residual, ete=ete)
    target_R = median_normalize(residual + ete, layout).astype(np.float32)
    return router_inputs, target_R


class ParametricResidualComputer:
    """Wraps the 5-input parametric PINN so it looks like a scalar-input
    PINN to the autodiff residual code, by baking in (k, x_s, y_s).
    """

    def __init__(self, pinn, k, xs, ys):
        self.pinn = pinn
        self.k = float(k)
        self.xs = float(xs)
        self.ys = float(ys)
        self._k_tf = tf.constant(self.k, dtype=tf.float32)
        self._xs_tf = tf.constant(self.xs, dtype=tf.float32)
        self._ys_tf = tf.constant(self.ys, dtype=tf.float32)

    @tf.function
    def _compute_flat(self, xy_flat, f_flat):
        N = tf.shape(xy_flat)[0]
        k_col = tf.fill([N, 1], self._k_tf)
        xs_col = tf.fill([N, 1], self._xs_tf)
        ys_col = tf.fill([N, 1], self._ys_tf)
        with tf.GradientTape() as t2:
            t2.watch(xy_flat)
            with tf.GradientTape() as t1:
                t1.watch(xy_flat)
                inp5 = tf.concat([xy_flat, k_col, xs_col, ys_col], axis=-1)
                u = self.pinn(inp5, training=False)
            grads = t1.gradient(u, xy_flat)
        hess = t2.batch_jacobian(grads, xy_flat)
        uxx = hess[:, 0, 0:1]
        uyy = hess[:, 1, 1:2]
        r = uxx + uyy + (self._k_tf ** 2) * u + f_flat
        return tf.reshape(r, [-1])

    def compute_signed_residual(self, X, Y, f_grid):
        shape = X.shape
        xy = np.stack([X.ravel(), Y.ravel()], axis=-1).astype(np.float32)
        f_flat = f_grid.ravel().astype(np.float32).reshape(-1, 1)
        r_flat = self._compute_flat(tf.constant(xy),
                                    tf.constant(f_flat)).numpy()
        return r_flat.reshape(shape).astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pinn-tag', type=str, required=True)
    p.add_argument('--configs', type=str, required=True)
    p.add_argument('--tag', type=str, required=True)
    p.add_argument('--pinn-dir', type=str, default='./models')
    p.add_argument('--pinn-meta-dir', type=str, default='./history')
    p.add_argument('--output-dir', type=str, default='./router_models')
    p.add_argument('--history-dir', type=str, default='./router_history')
    p.add_argument('--nx', type=int, default=201)
    p.add_argument('--ny', type=int, default=201)
    p.add_argument('--epochs', type=int, default=4000)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--lr-min', type=float, default=1e-5)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--lambda-tv', type=float, default=0.01)
    p.add_argument('--base-filters', type=int, default=32)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.history_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tf.random.set_seed(args.seed)

    # Load PINN meta + weights
    with open(os.path.join(args.pinn_meta_dir,
                           f'training_meta_{args.pinn_tag}.json')) as f:
        pmeta = json.load(f)
    assert pmeta.get('parametric', False), \
        "--pinn-tag must point at a parametric PINN meta"
    cx = float(pmeta['hole_center_x']); cy = float(pmeta['hole_center_y'])
    r_h = float(pmeta['hole_radius'])
    sigma = float(pmeta['sigma']); amplitude = float(pmeta['amplitude'])

    X, Y, layout, _, _ = create_square_with_hole(
        Nx=args.nx, Ny=args.ny,
        hole_center=(cx, cy), hole_radius=r_h)

    pinn = build_parametric_pinn(
        layers=tuple(pmeta['layers']),
        activation='tanh',
        spatial_range=((0.0, 1.0), (0.0, 1.0)),
        param_range=((pmeta['k_min'], pmeta['k_max']),
                     (pmeta['xs_min'], pmeta['xs_max']),
                     (pmeta['ys_min'], pmeta['ys_max'])),
        fourier_m=int(pmeta['fourier_m']),
        fourier_scale=float(pmeta['fourier_scale']),
    )
    _ = pinn(tf.zeros((1, 5)))
    pinn.load_weights(os.path.join(
        args.pinn_dir, f'pinn_helmholtz_{args.pinn_tag}.weights.h5'))

    # Load + filter configs.
    with open(args.configs) as f:
        configs = json.load(f)
    train_configs = [c for c in configs if c.get('split', 'id') == 'id']
    print(f"Training router on {len(train_configs)} ID configs")

    # Precompute router inputs + targets per config (one-time cost).
    per_cfg = []
    for i, cfg in enumerate(train_configs):
        print(f"  [{i+1}/{len(train_configs)}] precomputing for {cfg['name']}")
        ri, tg = _build_config_tensors(
            pinn, lambda m, k, xs, ys: ParametricResidualComputer(m, k, xs, ys),
            X, Y, layout, cfg, sigma, amplitude)
        per_cfg.append({
            'name': cfg['name'],
            'inputs': tf.constant(ri, dtype=tf.float32),
            'target': tf.constant(tg, dtype=tf.float32),
        })

    # Build router.
    router = RouterCNN(base_filters=args.base_filters)
    _ = router(per_cfg[0]['inputs'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    layout_tf = tf.constant(layout.astype(np.float32))
    beta = float(args.beta); lam_tv = float(args.lambda_tv)

    def _tv(r):
        tv_h = tf.reduce_mean(tf.abs(r[:, :, 1:, :] - r[:, :, :-1, :]))
        tv_v = tf.reduce_mean(tf.abs(r[:, 1:, :, :] - r[:, :-1, :, :]))
        return tv_h + tv_v

    @tf.function
    def train_step(inputs, target_R):
        with tf.GradientTape() as tape:
            s = router(inputs, training=True)
            s = s[0, :, :, 0]
            num = tf.reduce_sum(layout_tf) + 1e-10
            logistic = tf.reduce_sum(
                (beta * tf.math.softplus(s)
                 + target_R * tf.math.softplus(-s)) * layout_tf
            ) / num
            s4 = tf.reshape(s * layout_tf,
                            [1, tf.shape(s)[0], tf.shape(s)[1], 1])
            tv = lam_tv * _tv(s4)
            total = logistic + tv
        grads = tape.gradient(total, router.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optimizer.apply_gradients(zip(grads, router.trainable_variables))
        rejected = tf.reduce_sum(
            tf.cast(s > 0, tf.float32) * layout_tf) / num
        return total, logistic, tv, rejected

    # Cosine LR schedule over epochs.
    history = {'total': [], 'logistic': [], 'tv': [], 'reject': []}
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        progress = epoch / max(args.epochs - 1, 1)
        cur_lr = args.lr_min + 0.5 * (args.lr - args.lr_min) * (
            1 + np.cos(np.pi * progress))
        optimizer.learning_rate.assign(cur_lr)

        order = rng.permutation(len(per_cfg))
        ep_totals = []
        for idx in order:
            e = per_cfg[idx]
            total, logistic, tv, rejected = train_step(e['inputs'], e['target'])
            ep_totals.append(float(total))
        ep_total = float(np.mean(ep_totals))
        history['total'].append(ep_total)
        history['logistic'].append(float(logistic))
        history['tv'].append(float(tv))
        history['reject'].append(float(rejected))

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}  "
                  f"mean_L={ep_total:.4f}  last_logistic={float(logistic):.4f}  "
                  f"tv={float(tv):.4f}  reject%={float(rejected)*100:.1f}  "
                  f"lr={cur_lr:.2e}")
    train_time = time.perf_counter() - t0
    print(f"\nRouter training time: {train_time:.1f}s")

    w_path = os.path.join(args.output_dir,
                          f'router_helmholtz_{args.tag}.weights.h5')
    router.save_weights(w_path)
    print(f"Saved: {w_path}")

    meta = {
        'tag': args.tag, 'pinn_tag': args.pinn_tag,
        'configs': args.configs,
        'n_train_configs': len(train_configs),
        'epochs': args.epochs, 'lr': args.lr, 'lr_min': args.lr_min,
        'beta': beta, 'lambda_tv': lam_tv,
        'base_filters': args.base_filters,
        'train_time_s': train_time,
    }
    with open(os.path.join(args.history_dir,
                           f'router_meta_{args.tag}.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    np.savez(os.path.join(args.history_dir,
                          f'router_history_{args.tag}.npz'),
             **{k: np.array(v) for k, v in history.items()})


if __name__ == '__main__':
    main()
