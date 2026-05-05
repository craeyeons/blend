"""
Train a mixed-variable PINN for 2D static linear elasticity on L-bracket.

Adapted from Bansal et al.'s approach:
    output = uv_network(x,y) * dist_network(x,y) + part_network(x,y)

Three networks:
    - dist_network: learns per-output distance to BC boundaries (5 outputs)
    - part_network: learns a particular solution satisfying BCs (5 outputs)
    - uv_network:   learns the physics (5 outputs: u, v, s11, s22, s12)

Key advantage: the equilibrium PDE only needs 1st derivatives of the stress
outputs (not 2nd derivatives of displacement), making it much easier to train.

Usage:
    python train_pinn_mixed.py --problem l_bracket --epochs 20000
"""

import argparse
import os
import numpy as np
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass


def build_mlp(num_inputs, num_outputs, layers, activation='tanh', name='mlp'):
    """Build a simple MLP with Xavier initialization."""
    act_fn = {'tanh': 'tanh', 'swish': tf.keras.activations.swish}
    inputs = tf.keras.layers.Input(shape=(num_inputs,))
    x = inputs
    for units in layers:
        x = tf.keras.layers.Dense(
            units, activation=act_fn.get(activation, activation),
            kernel_initializer='glorot_normal')(x)
    outputs = tf.keras.layers.Dense(
        num_outputs, kernel_initializer='glorot_normal')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def lhs_sample(dim, n):
    """Simple Latin Hypercube Sampling in [0, 1]^dim."""
    result = np.zeros((n, dim))
    for d in range(dim):
        perm = np.random.permutation(n)
        result[:, d] = (perm + np.random.uniform(size=n)) / n
    return result


def create_l_bracket_data(args):
    """
    Create collocation and BC data for L-bracket mixed-variable PINN.

    Returns dict with all training data arrays.
    """
    cx, cy = args.corner_x, args.corner_y
    x_min, x_max = args.x_min, args.x_max
    y_min, y_max = args.y_min, args.y_max
    sigma = args.applied_stress

    # --- Collocation points in L-bracket interior ---
    # Oversample and reject void
    n_coll = args.n_domain
    pts = np.array([x_min, y_min]) + np.array([x_max - x_min, y_max - y_min]) * lhs_sample(2, int(n_coll * 1.5))
    # Reject void (x > cx AND y > cy)
    mask = ~((pts[:, 0] > cx) & (pts[:, 1] > cy))
    pts = pts[mask]
    # Refinement near re-entrant corner
    n_ref = n_coll // 3
    pts_ref = np.array([cx - 0.3, cy - 0.3]) + np.array([0.6, 0.6]) * lhs_sample(2, n_ref)
    mask_ref = ~((pts_ref[:, 0] > cx) & (pts_ref[:, 1] > cy))
    pts_ref = pts_ref[mask_ref]
    xy_coll = np.concatenate([pts[:n_coll], pts_ref], axis=0).astype(np.float32)

    # --- Distance training data ---
    # Distance function per output component:
    # dist_u  = y - y_min       (u=0 at bottom)
    # dist_v  = y - y_min       (v=0 at bottom)
    # dist_s11 = 1.0 everywhere (no Dirichlet-type BC on s11 directly)
    # dist_s22 = 1.0 everywhere
    # dist_s12 = 1.0 everywhere
    # Actually, following the reference more carefully:
    # The distance function should be ~0 where a boundary condition constrains
    # that particular output component, so that part_network takes over there.
    #
    # For L-bracket:
    # - Bottom (y=0): u=0, v=0  → dist_u, dist_v ~ 0 at y=0
    # - Top (y=2, x<cx): σ_yy = -σ, σ_xy = 0  → dist_s22, dist_s12 ~ 0 at top
    # - Left (x=0): σ_xx = 0, σ_xy = 0  → dist_s11, dist_s12 ~ 0 at left
    # - Right (x=2, y<cy): σ_xx = 0, σ_xy = 0  → dist_s11, dist_s12 ~ 0 at right
    # - Inner edges: traction-free → dist for stress components ~ 0 there

    # Sample points on a grid for distance training
    n_dist = 80
    x_d = np.linspace(x_min, x_max, n_dist)
    y_d = np.linspace(y_min, y_max, n_dist)
    xg, yg = np.meshgrid(x_d, y_d)
    # Remove void
    dst_mask = ~((xg > cx) & (yg > cy))
    x_flat = xg[dst_mask].flatten()[:, None]
    y_flat = yg[dst_mask].flatten()[:, None]
    # Add surface refinement points near re-entrant corner
    n_surf = 200
    theta = np.linspace(0, np.pi / 2, n_surf)
    r_ref = args.fillet_radius if args.fillet_radius > 0 else 0.02
    x_surf = cx - r_ref * np.cos(theta)
    y_surf = cy - r_ref * np.sin(theta)
    # Clamp to material
    valid = ~((x_surf > cx) & (y_surf > cy))
    x_surf = x_surf[valid].flatten()[:, None]
    y_surf = y_surf[valid].flatten()[:, None]
    x_flat = np.concatenate([x_flat, x_surf], 0)
    y_flat = np.concatenate([y_flat, y_surf], 0)

    xy_dist = np.concatenate([x_flat, y_flat], axis=1).astype(np.float32)

    # Compute distance targets for each of 5 output components
    x_ = xy_dist[:, 0:1]
    y_ = xy_dist[:, 1:2]

    # dist_u: zero at bottom (y=y_min)
    dist_u = y_ - y_min
    # dist_v: zero at bottom (y=y_min)
    dist_v = y_ - y_min
    # dist_s11: zero at left (x=0), right (x=x_max, y<cy), inner vertical (x=cx, y>cy)
    d_left = x_ - x_min
    d_right = np.where(y_ <= cy, x_max - x_, np.full_like(x_, 10.0))
    d_inner_v = np.where(y_ >= cy, np.abs(x_ - cx), np.full_like(x_, 10.0))
    dist_s11 = np.minimum(np.minimum(d_left, d_right), d_inner_v)
    # dist_s22: zero at top (y=y_max, x<cx), inner horizontal (y=cy, x>cx)
    d_top = np.where(x_ <= cx, y_max - y_, np.full_like(y_, 10.0))
    d_inner_h = np.where(x_ >= cx, np.abs(y_ - cy), np.full_like(y_, 10.0))
    dist_s22 = np.minimum(d_top, d_inner_h)
    # dist_s12: zero on all traction boundaries
    # min of distances to all edges
    d_bottom = y_ - y_min  # but s12 is not directly constrained at bottom (u=v=0)
    # Actually s12 = 0 on left, right, top, inner_h, inner_v edges (all traction-free or applied normal)
    dist_s12 = np.minimum(np.minimum(np.minimum(d_left, dist_s22), d_right), d_inner_v)

    DIST = np.concatenate([dist_u, dist_v, dist_s11, dist_s22, dist_s12], axis=1).astype(np.float32)

    # --- Particular solution (BC) training data ---
    n_bc = args.n_boundary

    # Bottom edge: u=0, v=0
    n_bot = n_bc
    x_bot = np.random.uniform(x_min, x_max, n_bot).astype(np.float32)
    y_bot = np.full(n_bot, y_min, dtype=np.float32)
    # For bottom, we only enforce u=0, v=0 through the particular solution
    xy_bot = np.stack([x_bot, y_bot], axis=1)
    # part targets at bottom: u=0, v=0, s11=free, s22=free, s12=free
    # We'll train with masks

    # Top edge (y=y_max, x in [x_min, cx]): σ_yy = -σ applied, σ_xy = 0
    n_top = n_bc
    x_top = np.random.uniform(x_min, cx, n_top).astype(np.float32)
    y_top = np.full(n_top, y_max, dtype=np.float32)
    xy_top = np.stack([x_top, y_top], axis=1)

    # Left edge (x=x_min): σ_xx = 0, σ_xy = 0
    n_lf = n_bc
    x_lf = np.full(n_lf, x_min, dtype=np.float32)
    y_lf = np.random.uniform(y_min, y_max, n_lf).astype(np.float32)
    xy_lf = np.stack([x_lf, y_lf], axis=1)

    # Right edge (x=x_max, y in [y_min, cy]): σ_xx = 0, σ_xy = 0
    n_rt = n_bc
    x_rt = np.full(n_rt, x_max, dtype=np.float32)
    y_rt = np.random.uniform(y_min, cy, n_rt).astype(np.float32)
    xy_rt = np.stack([x_rt, y_rt], axis=1)

    # Inner vertical (x=cx, y in [cy, y_max]): σ_xx = 0, σ_xy = 0
    n_iv = n_bc
    x_iv = np.full(n_iv, cx, dtype=np.float32)
    y_iv = np.random.uniform(cy, y_max, n_iv).astype(np.float32)
    xy_iv = np.stack([x_iv, y_iv], axis=1)

    # Inner horizontal (y=cy, x in [cx, x_max]): σ_yy = 0, σ_xy = 0
    n_ih = n_bc
    x_ih = np.random.uniform(cx, x_max, n_ih).astype(np.float32)
    y_ih = np.full(n_ih, cy, dtype=np.float32)
    xy_ih = np.stack([x_ih, y_ih], axis=1)

    # Hole surface points for traction-free BC in the physics loss
    # (points on the inner boundary of the L-bracket, near re-entrant corner)
    n_hole = n_bc * 2
    # Sample along inner edges
    xy_hole_v = np.stack([
        np.full(n_hole // 2, cx, dtype=np.float32),
        np.random.uniform(cy, y_max, n_hole // 2).astype(np.float32)
    ], axis=1)
    xy_hole_h = np.stack([
        np.random.uniform(cx, x_max, n_hole // 2).astype(np.float32),
        np.full(n_hole // 2, cy, dtype=np.float32)
    ], axis=1)
    # Normals: inner vertical has n=(1,0), inner horizontal has n=(0,1)
    n_hole_v = np.column_stack([np.ones(n_hole // 2), np.zeros(n_hole // 2)]).astype(np.float32)
    n_hole_h = np.column_stack([np.zeros(n_hole // 2), np.ones(n_hole // 2)]).astype(np.float32)
    xy_hole = np.concatenate([xy_hole_v, xy_hole_h], axis=0)
    normals_hole = np.concatenate([n_hole_v, n_hole_h], axis=0)

    # Also add outer traction boundaries for hole-like enforcement
    # Right outer edge: n=(1,0), traction-free
    xy_hole_rt = xy_rt.copy()
    n_hole_rt = np.column_stack([np.ones(n_rt), np.zeros(n_rt)]).astype(np.float32)
    # Left outer edge: n=(-1,0), traction-free
    xy_hole_lf = xy_lf.copy()
    n_hole_lf = np.column_stack([-np.ones(n_lf), np.zeros(n_lf)]).astype(np.float32)
    # Top loaded edge: n=(0,1), σ·n = (-σ, 0)
    xy_hole_top = xy_top.copy()
    n_hole_top = np.column_stack([np.zeros(n_top), np.ones(n_top)]).astype(np.float32)
    trac_top = np.column_stack([np.full(n_top, -sigma), np.zeros(n_top)]).astype(np.float32)

    # Combine all boundary points for hole/traction loss
    xy_surf = np.concatenate([xy_hole, xy_hole_rt, xy_hole_lf, xy_hole_top], axis=0)
    normals_surf = np.concatenate([normals_hole, n_hole_rt, n_hole_lf, n_hole_top], axis=0)
    trac_surf = np.concatenate([
        np.zeros((len(xy_hole), 2), dtype=np.float32),       # traction-free inner edges
        np.zeros((n_rt, 2), dtype=np.float32),                 # traction-free right
        np.zeros((n_lf, 2), dtype=np.float32),                 # traction-free left
        trac_top,                                               # loaded top
    ], axis=0)

    # Add some BC points into collocation set (as in reference)
    xy_coll = np.concatenate([
        xy_coll,
        xy_hole[::5],
        xy_lf[::5],
        xy_rt[::5],
        xy_top[::5],
        xy_bot[::5],
    ], axis=0)

    return {
        'xy_coll': xy_coll,
        'xy_dist': xy_dist,
        'DIST': DIST,
        'xy_bot': xy_bot,
        'xy_top': xy_top,
        'xy_lf': xy_lf,
        'xy_rt': xy_rt,
        'xy_iv': xy_iv,
        'xy_ih': xy_ih,
        'xy_surf': xy_surf,
        'normals_surf': normals_surf,
        'trac_surf': trac_surf,
        'sigma': sigma,
    }


class MixedPINNTrainer:
    """
    Train mixed-variable PINN with distance-particular decomposition.

    output = uv_model(xy) * dist_model(xy) + part_model(xy)
    outputs: [u, v, σ11, σ22, σ12]

    Losses:
        1. Constitutive: σ_predicted - C:ε(u) = 0
        2. Equilibrium: div(σ) = 0
        3. Traction BC: σ·n = t on boundaries
    """

    def __init__(self, uv_model, dist_model, part_model,
                 E=1.0, nu=0.3, lr=1e-4, epochs=20000):
        self.uv_model = uv_model
        self.dist_model = dist_model
        self.part_model = part_model
        self.E = E
        self.nu = nu
        self.C11 = E / (1.0 - nu ** 2)
        self.C12 = nu * E / (1.0 - nu ** 2)
        self.C66 = E / (2.0 * (1.0 + nu))

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr, decay_steps=epochs, alpha=1e-2)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.history = {
            'total': [], 'constitutive': [], 'equilibrium': [],
            'traction': [],
        }

    def _combined_output(self, xy):
        """Compute output = uv * dist + part."""
        y_uv = self.uv_model(xy, training=True)
        y_dist = self.dist_model(xy, training=False)  # frozen
        y_part = self.part_model(xy, training=False)   # frozen
        return y_uv * y_dist + y_part

    @tf.function
    def train_step(self, xy_coll, xy_surf, normals_surf, trac_surf):
        uv_vars = self.uv_model.trainable_variables
        with tf.GradientTape() as tape:
            # --- Physics loss at collocation points ---
            with tf.GradientTape(persistent=True) as tape_inner:
                tape_inner.watch(xy_coll)
                yh = self._combined_output(xy_coll)
                u = yh[:, 0]
                v = yh[:, 1]
                s11 = yh[:, 2]
                s22 = yh[:, 3]
                s12 = yh[:, 4]

            # Strain from displacement gradients
            grad_u = tape_inner.gradient(u, xy_coll)
            grad_v = tape_inner.gradient(v, xy_coll)
            du_dx = grad_u[:, 0]
            du_dy = grad_u[:, 1]
            dv_dx = grad_v[:, 0]
            dv_dy = grad_v[:, 1]

            # Constitutive: σ should match C:ε
            sp11 = self.C11 * du_dx + self.C12 * dv_dy
            sp22 = self.C12 * du_dx + self.C11 * dv_dy
            sp12 = self.C66 * (du_dy + dv_dx)

            loss_const = (tf.reduce_mean((s11 - sp11) ** 2) +
                          tf.reduce_mean((s22 - sp22) ** 2) +
                          tf.reduce_mean((s12 - sp12) ** 2))

            # Equilibrium: div(σ) = 0
            grad_s11 = tape_inner.gradient(s11, xy_coll)
            grad_s22 = tape_inner.gradient(s22, xy_coll)
            grad_s12 = tape_inner.gradient(s12, xy_coll)
            del tape_inner

            ds11_dx = grad_s11[:, 0] if grad_s11 is not None else tf.zeros_like(u)
            ds12_dy = grad_s12[:, 1] if grad_s12 is not None else tf.zeros_like(u)
            ds22_dy = grad_s22[:, 1] if grad_s22 is not None else tf.zeros_like(u)
            ds12_dx = grad_s12[:, 0] if grad_s12 is not None else tf.zeros_like(u)

            f_u = ds11_dx + ds12_dy
            f_v = ds12_dx + ds22_dy
            loss_eq = tf.reduce_mean(f_u ** 2 + f_v ** 2)

            # --- Traction BC on all boundary surfaces ---
            yh_surf = self._combined_output(xy_surf)
            s11_s = yh_surf[:, 2]
            s22_s = yh_surf[:, 3]
            s12_s = yh_surf[:, 4]
            nx = normals_surf[:, 0]
            ny = normals_surf[:, 1]
            tx_pred = s11_s * nx + s12_s * ny
            ty_pred = s12_s * nx + s22_s * ny
            tx_true = trac_surf[:, 0]
            ty_true = trac_surf[:, 1]
            loss_trac = tf.reduce_mean((tx_pred - tx_true) ** 2 +
                                       (ty_pred - ty_true) ** 2)

            total = loss_const + loss_eq + 10.0 * loss_trac

        grads = tape.gradient(total, uv_vars)
        safe_grads = []
        for g, v in zip(grads, uv_vars):
            if g is None:
                safe_grads.append(tf.zeros_like(v))
            else:
                safe_grads.append(tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)))
        safe_grads, _ = tf.clip_by_global_norm(safe_grads, 1.0)
        self.optimizer.apply_gradients(zip(safe_grads, uv_vars))

        return total, loss_const, loss_eq, loss_trac

    def train(self, data, epochs=20000, batch_size=256, print_every=500):
        xy_coll = tf.constant(data['xy_coll'], dtype=tf.float32)
        xy_surf = tf.constant(data['xy_surf'], dtype=tf.float32)
        normals_surf = tf.constant(data['normals_surf'], dtype=tf.float32)
        trac_surf = tf.constant(data['trac_surf'], dtype=tf.float32)

        n_coll = len(data['xy_coll'])
        n_surf = len(data['xy_surf'])

        for epoch in range(epochs):
            # Mini-batch: shuffle collocation and surface points
            idx_c = tf.random.shuffle(tf.range(n_coll))[:batch_size]
            idx_s = tf.random.shuffle(tf.range(n_surf))[:batch_size]

            total, lc, le, lt = self.train_step(
                tf.gather(xy_coll, idx_c),
                tf.gather(xy_surf, idx_s),
                tf.gather(normals_surf, idx_s),
                tf.gather(trac_surf, idx_s),
            )
            self.history['total'].append(float(total))
            self.history['constitutive'].append(float(lc))
            self.history['equilibrium'].append(float(le))
            self.history['traction'].append(float(lt))

            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Total: {total:.6e}, "
                      f"Const: {lc:.6e}, "
                      f"Equil: {le:.6e}, "
                      f"Trac: {lt:.6e}")

        return self.history


def train_dist_model(dist_model, xy_dist, DIST, epochs=2000, lr=1e-3):
    """Pre-train the distance model to fit distance targets."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    xy = tf.constant(xy_dist, dtype=tf.float32)
    dist_target = tf.constant(DIST, dtype=tf.float32)

    best_loss = 1e10
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            pred = dist_model(xy, training=True)
            loss = tf.reduce_mean((pred - dist_target) ** 2)
        grads = tape.gradient(loss, dist_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, dist_model.trainable_variables))
        if float(loss) < best_loss:
            best_loss = float(loss)
        if (epoch + 1) % 500 == 0:
            print(f"  Dist epoch {epoch+1}: loss = {loss:.6e}")
    print(f"  Dist model best loss: {best_loss:.6e}")


def train_part_model(part_model, data, epochs=2000, lr=1e-3):
    """Pre-train the particular solution model to satisfy BCs."""
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    sigma = data['sigma']

    xy_bot = tf.constant(data['xy_bot'], dtype=tf.float32)
    xy_top = tf.constant(data['xy_top'], dtype=tf.float32)
    xy_lf = tf.constant(data['xy_lf'], dtype=tf.float32)
    xy_rt = tf.constant(data['xy_rt'], dtype=tf.float32)
    xy_iv = tf.constant(data['xy_iv'], dtype=tf.float32)
    xy_ih = tf.constant(data['xy_ih'], dtype=tf.float32)

    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            # Bottom: u=0, v=0
            yh_bot = part_model(xy_bot, training=True)
            loss = tf.reduce_mean(yh_bot[:, 0] ** 2 + yh_bot[:, 1] ** 2)

            # Left: σ_xx=0, σ_xy=0
            yh_lf = part_model(xy_lf, training=True)
            loss = loss + tf.reduce_mean(yh_lf[:, 2] ** 2 + yh_lf[:, 4] ** 2)

            # Top: σ_yy=-σ, σ_xy=0
            yh_top = part_model(xy_top, training=True)
            loss = loss + tf.reduce_mean((yh_top[:, 3] - (-sigma)) ** 2 + yh_top[:, 4] ** 2)

            # Right: σ_xx=0, σ_xy=0
            yh_rt = part_model(xy_rt, training=True)
            loss = loss + tf.reduce_mean(yh_rt[:, 2] ** 2 + yh_rt[:, 4] ** 2)

            # Inner vertical (x=cx): σ_xx=0, σ_xy=0
            yh_iv = part_model(xy_iv, training=True)
            loss = loss + tf.reduce_mean(yh_iv[:, 2] ** 2 + yh_iv[:, 4] ** 2)

            # Inner horizontal (y=cy): σ_yy=0, σ_xy=0
            yh_ih = part_model(xy_ih, training=True)
            loss = loss + tf.reduce_mean(yh_ih[:, 3] ** 2 + yh_ih[:, 4] ** 2)

            loss = 1000.0 * loss

        grads = tape.gradient(loss, part_model.trainable_variables)
        optimizer.apply_gradients(zip(grads, part_model.trainable_variables))
        if (epoch + 1) % 500 == 0:
            print(f"  Part epoch {epoch+1}: loss = {loss:.6e}")


def predict_mixed(uv_model, dist_model, part_model, X, Y, layout):
    """
    Predict displacement and stress fields on the L-bracket grid.

    Returns: ux, uy, s11, s22, s12 (all shaped like X, masked by layout)
    """
    xy_flat = np.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)
    xy_tf = tf.constant(xy_flat)
    y_uv = uv_model(xy_tf, training=False).numpy()
    y_dist = dist_model(xy_tf, training=False).numpy()
    y_part = part_model(xy_tf, training=False).numpy()
    y_out = y_uv * y_dist + y_part

    ux = y_out[:, 0].reshape(X.shape) * layout
    uy = y_out[:, 1].reshape(X.shape) * layout
    s11 = y_out[:, 2].reshape(X.shape) * layout
    s22 = y_out[:, 3].reshape(X.shape) * layout
    s12 = y_out[:, 4].reshape(X.shape) * layout
    return ux, uy, s11, s22, s12


def main():
    parser = argparse.ArgumentParser(
        description='Train mixed-variable PINN for L-bracket elasticity')
    parser.add_argument('--problem', type=str, default='l_bracket')
    parser.add_argument('--epochs', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--n-domain', type=int, default=40000)
    parser.add_argument('--n-boundary', type=int, default=2000)
    parser.add_argument('--output-dir', type=str, default='./models')

    # Network architecture
    parser.add_argument('--uv-layers', type=int, nargs='+', default=[70] * 8)
    parser.add_argument('--dist-layers', type=int, nargs='+', default=[20] * 4)
    parser.add_argument('--part-layers', type=int, nargs='+', default=[20] * 4)
    parser.add_argument('--activation', type=str, default='tanh')

    # Pre-training epochs
    parser.add_argument('--dist-epochs', type=int, default=2000)
    parser.add_argument('--part-epochs', type=int, default=2000)

    # Material
    parser.add_argument('--E', type=float, default=1.0)
    parser.add_argument('--nu', type=float, default=0.3)
    parser.add_argument('--applied-stress', type=float, default=10.0)

    # Domain
    parser.add_argument('--x-min', type=float, default=0.0)
    parser.add_argument('--x-max', type=float, default=2.0)
    parser.add_argument('--y-min', type=float, default=0.0)
    parser.add_argument('--y-max', type=float, default=2.0)
    parser.add_argument('--corner-x', type=float, default=1.0)
    parser.add_argument('--corner-y', type=float, default=1.0)
    parser.add_argument('--fillet-radius', type=float, default=0.04)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("MIXED-VARIABLE PINN TRAINING: L-bracket")
    print("=" * 60)
    print(f"  Domain: [{args.x_min},{args.x_max}] x [{args.y_min},{args.y_max}]")
    print(f"  Corner: ({args.corner_x}, {args.corner_y})")
    print(f"  Applied stress: {args.applied_stress}")
    print(f"  E={args.E}, nu={args.nu}")
    print(f"  UV net: {args.uv_layers}, Dist net: {args.dist_layers}, Part net: {args.part_layers}")
    print()

    # Build three models (5 outputs each: u, v, σ11, σ22, σ12)
    uv_model = build_mlp(2, 5, args.uv_layers, args.activation, name='uv')
    dist_model = build_mlp(2, 5, args.dist_layers, args.activation, name='dist')
    part_model = build_mlp(2, 5, args.part_layers, args.activation, name='part')

    print("UV model:"); uv_model.summary()
    print("\nDist model:"); dist_model.summary()
    print("\nPart model:"); part_model.summary()

    # Generate training data
    print("\nGenerating training data...")
    data = create_l_bracket_data(args)
    print(f"  Collocation points: {len(data['xy_coll'])}")
    print(f"  Distance training points: {len(data['xy_dist'])}")
    print(f"  Surface/traction points: {len(data['xy_surf'])}")

    # --- Step 1: Pre-train distance model ---
    print("\n=== Pre-training distance model ===")
    train_dist_model(dist_model, data['xy_dist'], data['DIST'],
                     epochs=args.dist_epochs)

    # --- Step 2: Pre-train particular solution model ---
    print("\n=== Pre-training particular solution model ===")
    train_part_model(part_model, data, epochs=args.part_epochs)

    # Freeze dist and part models
    dist_model.trainable = False
    part_model.trainable = False

    # --- Step 3: Train main UV model with physics ---
    print("\n=== Training UV model (physics) ===")
    trainer = MixedPINNTrainer(
        uv_model, dist_model, part_model,
        E=args.E, nu=args.nu, lr=args.lr, epochs=args.epochs)
    history = trainer.train(data, epochs=args.epochs,
                            batch_size=args.batch_size)

    # Save all three models
    uv_path = os.path.join(args.output_dir, 'pinn_mixed_uv.weights.h5')
    dist_path = os.path.join(args.output_dir, 'pinn_mixed_dist.weights.h5')
    part_path = os.path.join(args.output_dir, 'pinn_mixed_part.weights.h5')
    uv_model.save_weights(uv_path)
    dist_model.save_weights(dist_path)
    part_model.save_weights(part_path)
    print(f"\nSaved models to {args.output_dir}")

    history_path = os.path.join(args.output_dir, 'pinn_mixed_history.npz')
    np.savez(history_path, **history)

    # --- Visualization ---
    try:
        import matplotlib.pyplot as plt
        try:
            import scienceplots
            plt.style.use(['science', 'no-latex'])
        except ImportError:
            pass

        from lib.domains import create_l_bracket

        X, Y, layout, *_ = create_l_bracket(
            Nx=200, Ny=200,
            x_domain=(args.x_min, args.x_max),
            y_domain=(args.y_min, args.y_max),
            corner_x=args.corner_x,
            corner_y=args.corner_y,
            applied_stress=args.applied_stress,
        )

        ux, uy, s11, s22, s12 = predict_mixed(
            uv_model, dist_model, part_model, X, Y, layout)

        vm = np.sqrt(s11**2 - s11*s22 + s22**2 + 3*s12**2) * layout

        # Displacement plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, field, title in zip(axes, [ux, uy, np.sqrt(ux**2 + uy**2)],
                                     ['$u_x$', '$u_y$', '$|u|$']):
            masked = np.ma.masked_where(layout == 0, field)
            cf = ax.contourf(X, Y, masked, levels=50, cmap='RdBu_r')
            plt.colorbar(cf, ax=ax)
            ax.set_aspect('equal')
            ax.set_title(title)
        plt.suptitle('Mixed-Variable PINN — Displacement', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'pinn_mixed_displacement.pdf'),
                    dpi=1200, bbox_inches='tight')

        # Stress plot
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        for ax, field, title in zip(axes,
                                     [s11, s22, s12, vm],
                                     [r'$\sigma_{11}$', r'$\sigma_{22}$',
                                      r'$\sigma_{12}$', r'$\sigma_{VM}$']):
            masked = np.ma.masked_where(layout == 0, field)
            cf = ax.contourf(X, Y, masked, levels=50, cmap='RdBu_r')
            plt.colorbar(cf, ax=ax)
            ax.set_aspect('equal')
            ax.set_title(title)
        plt.suptitle('Mixed-Variable PINN — Stress', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'pinn_mixed_stress.pdf'),
                    dpi=1200, bbox_inches='tight')

        # Loss plot
        fig, ax = plt.subplots(figsize=(8, 5))
        for key in ['total', 'constitutive', 'equilibrium', 'traction']:
            ax.semilogy(history[key], label=key.capitalize(), alpha=0.7)
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.set_title('Mixed-Variable PINN Training Loss')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, 'pinn_mixed_loss.pdf'), dpi=1200)
        plt.show()

        print("Saved plots.")
    except Exception as e:
        print(f"Plotting skipped: {e}")

    print("\nDone.")


if __name__ == '__main__':
    main()
