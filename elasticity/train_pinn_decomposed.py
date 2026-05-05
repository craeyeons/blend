"""
Train decomposed PINN for L-bracket: two overlapping rectangular sub-domains
with a coupling loss, then blend at inference.

Sub-domains:
    V-bar: [0, corner_x] x [0, y_max]  (vertical arm + overlap)
    H-bar: [0, x_max]    x [0, corner_y] (horizontal arm + overlap)
    Overlap: [0, corner_x] x [0, corner_y]

Usage:
    python train_pinn_decomposed.py --epochs 10000
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

from lib.network import Network


def sample_interior(x_range, y_range, n):
    """Uniformly sample interior points in a rectangle."""
    x = np.random.uniform(x_range[0], x_range[1], n).astype(np.float32)
    y = np.random.uniform(y_range[0], y_range[1], n).astype(np.float32)
    return np.stack([x, y], axis=-1)


def sample_edge(x_range, y_range, edge, n):
    """Sample n points on a specific edge of a rectangle.

    edge: 'bottom', 'top', 'left', 'right'
    """
    if edge == 'bottom':
        x = np.random.uniform(x_range[0], x_range[1], n).astype(np.float32)
        y = np.full(n, y_range[0], dtype=np.float32)
    elif edge == 'top':
        x = np.random.uniform(x_range[0], x_range[1], n).astype(np.float32)
        y = np.full(n, y_range[1], dtype=np.float32)
    elif edge == 'left':
        x = np.full(n, x_range[0], dtype=np.float32)
        y = np.random.uniform(y_range[0], y_range[1], n).astype(np.float32)
    elif edge == 'right':
        x = np.full(n, x_range[1], dtype=np.float32)
        y = np.random.uniform(y_range[0], y_range[1], n).astype(np.float32)
    return np.stack([x, y], axis=-1)


def sample_partial_edge(x_range, y_range, edge, coord_range, n):
    """Sample n points on a partial edge (subset of the full edge).

    edge: 'top' or 'right'
    coord_range: (lo, hi) range along the free coordinate
    """
    if edge == 'top':
        x = np.random.uniform(coord_range[0], coord_range[1], n).astype(np.float32)
        y = np.full(n, y_range[1], dtype=np.float32)
    elif edge == 'right':
        x = np.full(n, x_range[1], dtype=np.float32)
        y = np.random.uniform(coord_range[0], coord_range[1], n).astype(np.float32)
    elif edge == 'bottom':
        x = np.random.uniform(coord_range[0], coord_range[1], n).astype(np.float32)
        y = np.full(n, y_range[0], dtype=np.float32)
    elif edge == 'left':
        x = np.full(n, x_range[0], dtype=np.float32)
        y = np.random.uniform(coord_range[0], coord_range[1], n).astype(np.float32)
    return np.stack([x, y], axis=-1)


class DecomposedTrainer:
    """Train two sub-domain PINNs jointly with coupling loss."""

    def __init__(self, model_v, model_h, E=1.0, nu=0.3, lr=3e-4, epochs=10000,
                 w_pde=1.0, w_trac=1.0, w_coupling=10.0,
                 w_trac_match=10.0, grad_clip_norm=1.0):
        self.model_v = model_v
        self.model_h = model_h
        self.E = E
        self.nu = nu
        self.C11 = E / (1.0 - nu ** 2)
        self.C12 = nu * E / (1.0 - nu ** 2)
        self.C66 = E / (2.0 * (1.0 + nu))
        self.w_pde = w_pde
        self.w_trac = w_trac
        self.w_coupling = w_coupling
        self.w_trac_match = w_trac_match
        self.grad_clip_norm = grad_clip_norm

        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr, decay_steps=epochs, alpha=1e-2)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.history = {
            'total': [], 'pde_v': [], 'pde_h': [],
            'trac_v': [], 'trac_h': [], 'coupling': [],
            'trac_match': [],
        }

    def _pde_residual(self, model, xy):
        """Compute equilibrium residual for a model."""
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(xy)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(xy)
                uv = model(xy, training=True)
                ux = uv[:, 0]
                uy = uv[:, 1]
            grad_ux = tape1.gradient(ux, xy)
            grad_uy = tape1.gradient(uy, xy)
            dux_dx = grad_ux[:, 0]
            dux_dy = grad_ux[:, 1]
            duy_dx = grad_uy[:, 0]
            duy_dy = grad_uy[:, 1]
            sxx = self.C11 * dux_dx + self.C12 * duy_dy
            syy = self.C12 * dux_dx + self.C11 * duy_dy
            sxy = self.C66 * (dux_dy + duy_dx)
        grad_sxx = tape2.gradient(sxx, xy)
        grad_syy = tape2.gradient(syy, xy)
        grad_sxy = tape2.gradient(sxy, xy)
        dsxx_dx = grad_sxx[:, 0] if grad_sxx is not None else tf.zeros_like(ux)
        dsxy_dy = grad_sxy[:, 1] if grad_sxy is not None else tf.zeros_like(ux)
        dsxy_dx = grad_sxy[:, 0] if grad_sxy is not None else tf.zeros_like(ux)
        dsyy_dy = grad_syy[:, 1] if grad_syy is not None else tf.zeros_like(ux)
        del tape1, tape2
        eq_x = dsxx_dx + dsxy_dy
        eq_y = dsxy_dx + dsyy_dy
        return tf.reduce_mean(eq_x ** 2 + eq_y ** 2)

    def _trac_loss(self, model, xy, trac_vals, normals):
        """Traction BC loss: sigma . n = t."""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xy)
            uv = model(xy, training=True)
            ux = uv[:, 0]
            uy = uv[:, 1]
        grad_ux = tape.gradient(ux, xy)
        grad_uy = tape.gradient(uy, xy)
        del tape
        dux_dx = grad_ux[:, 0]
        dux_dy = grad_ux[:, 1]
        duy_dx = grad_uy[:, 0]
        duy_dy = grad_uy[:, 1]
        sxx = self.C11 * dux_dx + self.C12 * duy_dy
        syy = self.C12 * dux_dx + self.C11 * duy_dy
        sxy = self.C66 * (dux_dy + duy_dx)
        nx, ny = normals[:, 0], normals[:, 1]
        tx_pred = sxx * nx + sxy * ny
        ty_pred = sxy * nx + syy * ny
        tx_true, ty_true = trac_vals[:, 0], trac_vals[:, 1]
        return tf.reduce_mean((tx_pred - tx_true) ** 2 + (ty_pred - ty_true) ** 2)

    def _traction_at(self, model, xy, normals):
        """Compute traction vector sigma . n at given points."""
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(xy)
            uv = model(xy, training=True)
            ux = uv[:, 0]
            uy = uv[:, 1]
        grad_ux = tape.gradient(ux, xy)
        grad_uy = tape.gradient(uy, xy)
        del tape
        dux_dx = grad_ux[:, 0]
        dux_dy = grad_ux[:, 1]
        duy_dx = grad_uy[:, 0]
        duy_dy = grad_uy[:, 1]
        sxx = self.C11 * dux_dx + self.C12 * duy_dy
        syy = self.C12 * dux_dx + self.C11 * duy_dy
        sxy = self.C66 * (dux_dy + duy_dx)
        nx, ny = normals[:, 0], normals[:, 1]
        tx = sxx * nx + sxy * ny
        ty = sxy * nx + syy * ny
        return tx, ty

    def _traction_matching_loss(self, xy_iface, normals_iface):
        """Both PINNs must produce the same traction at the interface."""
        tx_v, ty_v = self._traction_at(self.model_v, xy_iface, normals_iface)
        tx_h, ty_h = self._traction_at(self.model_h, xy_iface, normals_iface)
        return tf.reduce_mean((tx_v - tx_h) ** 2 + (ty_v - ty_h) ** 2)

    @tf.function
    def train_step(self, xy_pde_v, xy_pde_h,
                   xy_trac_v, trac_vals_v, trac_normals_v,
                   xy_trac_h, trac_vals_h, trac_normals_h,
                   xy_coupling, xy_iface, normals_iface):
        all_vars = self.model_v.trainable_variables + self.model_h.trainable_variables
        with tf.GradientTape() as tape:
            pde_v = self._pde_residual(self.model_v, xy_pde_v)
            pde_h = self._pde_residual(self.model_h, xy_pde_h)
            trac_v = self._trac_loss(self.model_v, xy_trac_v,
                                     trac_vals_v, trac_normals_v)
            trac_h = self._trac_loss(self.model_h, xy_trac_h,
                                     trac_vals_h, trac_normals_h)
            # Coupling: both PINNs should agree in the overlap
            uv_v = self.model_v(xy_coupling, training=True)
            uv_h = self.model_h(xy_coupling, training=True)
            coupling = tf.reduce_mean((uv_v - uv_h) ** 2)

            # Traction matching: sigma_V . n = sigma_H . n at the interface
            trac_match = self._traction_matching_loss(xy_iface, normals_iface)

            total = (self.w_pde * (pde_v + pde_h)
                     + self.w_trac * (trac_v + trac_h)
                     + self.w_coupling * coupling
                     + self.w_trac_match * trac_match)

        grads = tape.gradient(total, all_vars)
        safe_grads = []
        for grad, var in zip(grads, all_vars):
            if grad is None:
                safe_grads.append(tf.zeros_like(var))
            else:
                safe_grads.append(tf.where(tf.math.is_finite(grad),
                                           grad, tf.zeros_like(grad)))
        if self.grad_clip_norm is not None:
            safe_grads, grad_norm = tf.clip_by_global_norm(
                safe_grads, self.grad_clip_norm)
        else:
            grad_norm = tf.linalg.global_norm(safe_grads)
        self.optimizer.apply_gradients(zip(safe_grads, all_vars))
        return total, pde_v, pde_h, trac_v, trac_h, coupling, trac_match

    def train(self, data, epochs=10000, print_every=500):
        (xy_pde_v, xy_pde_h,
         xy_trac_v, trac_vals_v, trac_normals_v,
         xy_trac_h, trac_vals_h, trac_normals_h,
         xy_coupling, xy_iface, normals_iface,
         ) = [tf.constant(d, dtype=tf.float32) for d in data]

        for epoch in range(epochs):
            (total, pde_v, pde_h, trac_v, trac_h,
             coupling, trac_match) = self.train_step(
                xy_pde_v, xy_pde_h,
                xy_trac_v, trac_vals_v, trac_normals_v,
                xy_trac_h, trac_vals_h, trac_normals_h,
                xy_coupling, xy_iface, normals_iface,
            )
            self.history['total'].append(float(total))
            self.history['pde_v'].append(float(pde_v))
            self.history['pde_h'].append(float(pde_h))
            self.history['trac_v'].append(float(trac_v))
            self.history['trac_h'].append(float(trac_h))
            self.history['coupling'].append(float(coupling))
            self.history['trac_match'].append(float(trac_match))

            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Total: {total:.6e}, "
                      f"PDE_v: {pde_v:.6e}, PDE_h: {pde_h:.6e}, "
                      f"Trac_v: {trac_v:.6e}, Trac_h: {trac_h:.6e}, "
                      f"Coupling: {coupling:.6e}, "
                      f"TracMatch: {trac_match:.6e}")
        return self.history


def load_decomposed_pinn(vbar_path, hbar_path, layers, activation,
                         x_min, x_max, y_min, y_max, corner_x, corner_y):
    """
    Load both sub-domain PINN models for the decomposed L-bracket.

    Returns (model_v, model_h).
    """
    network = Network()
    model_v = network.build(num_inputs=2, layers=layers,
                            activation=activation, num_outputs=2,
                            input_range=[(x_min, corner_x), (y_min, y_max)],
                            hard_bc='bottom_fixed')
    model_h = network.build(num_inputs=2, layers=layers,
                            activation=activation, num_outputs=2,
                            input_range=[(x_min, x_max), (y_min, corner_y)],
                            hard_bc='bottom_fixed')
    model_v.load_weights(vbar_path)
    model_h.load_weights(hbar_path)
    return model_v, model_h


def blend_solutions(model_v, model_h, X, Y, layout,
                    corner_x, corner_y, sharpness=10.0):
    """
    Blend V-bar and H-bar PINN solutions on the full L-bracket grid.

    In the overlap region [0, cx] x [0, cy], blends using distance-based
    weights: points nearer the V-bar exclusive region favor V-bar, etc.

    Parameters
    ----------
    model_v, model_h : tf.keras.Model
    X, Y : ndarray (Ny, Nx)
    layout : ndarray (Ny, Nx)
    corner_x, corner_y : float
    sharpness : float
        Controls blending transition sharpness.

    Returns
    -------
    ux, uy : ndarray (Ny, Nx)
    """
    cx, cy = corner_x, corner_y
    xy_flat = np.stack([X.flatten(), Y.flatten()], axis=-1).astype(np.float32)
    x_flat = xy_flat[:, 0]
    y_flat = xy_flat[:, 1]

    # Identify regions
    in_v = (x_flat <= cx)  # V-bar domain
    in_h = (y_flat <= cy)  # H-bar domain
    in_overlap = in_v & in_h
    in_v_only = in_v & ~in_h  # V-bar exclusive (y > cy, x < cx)
    in_h_only = in_h & ~in_v  # H-bar exclusive (x > cx, y < cy)

    ux_out = np.zeros(len(xy_flat), dtype=np.float32)
    uy_out = np.zeros(len(xy_flat), dtype=np.float32)

    # V-bar exclusive
    if np.any(in_v_only):
        uv = model_v.predict(xy_flat[in_v_only], batch_size=4096, verbose=0)
        ux_out[in_v_only] = uv[:, 0]
        uy_out[in_v_only] = uv[:, 1]

    # H-bar exclusive
    if np.any(in_h_only):
        uv = model_h.predict(xy_flat[in_h_only], batch_size=4096, verbose=0)
        ux_out[in_h_only] = uv[:, 0]
        uy_out[in_h_only] = uv[:, 1]

    # Overlap: smooth blend
    if np.any(in_overlap):
        pts = xy_flat[in_overlap]
        uv_v = model_v.predict(pts, batch_size=4096, verbose=0)
        uv_h = model_h.predict(pts, batch_size=4096, verbose=0)

        # Distance to V-bar exclusive region (y = cy boundary)
        d_v = cy - pts[:, 1]  # small when close to V-bar exclusive
        # Distance to H-bar exclusive region (x = cx boundary)
        d_h = cx - pts[:, 0]  # small when close to H-bar exclusive

        # w_v: weight for V-bar. High when close to V-bar exclusive (d_v small).
        w_v = d_h / (d_v + d_h + 1e-10)
        w_v = w_v[:, None]  # (N, 1) for broadcasting

        blended = w_v * uv_v + (1.0 - w_v) * uv_h
        ux_out[in_overlap] = blended[:, 0]
        uy_out[in_overlap] = blended[:, 1]

    ux = ux_out.reshape(X.shape) * layout
    uy = uy_out.reshape(X.shape) * layout
    return ux, uy


def resolve_load_vector(args):
    """Return (tx, ty) for the applied traction, from magnitude/angle if given.

    Falls back to legacy behavior: load_edge='top', tx=-applied_stress, ty=0.
    Angle is in degrees, measured CCW from +x.
    """
    mag = args.load_magnitude if args.load_magnitude is not None else args.applied_stress
    if args.load_angle is not None:
        theta = np.deg2rad(args.load_angle)
        tx = mag * np.cos(theta)
        ty = mag * np.sin(theta)
    else:
        if args.load_edge == 'top':
            tx, ty = -mag, 0.0
        else:  # 'right'
            tx, ty = mag, 0.0
    return float(tx), float(ty)


def create_training_data(args):
    """Create training data for both sub-domain PINNs."""
    cx, cy = args.corner_x, args.corner_y
    x_min, x_max = args.x_min, args.x_max
    y_min, y_max = args.y_min, args.y_max
    n_domain = args.n_domain
    n_bc = args.n_boundary
    tx_val, ty_val = resolve_load_vector(args)
    top_loaded = (args.load_edge == 'top')
    right_loaded = (args.load_edge == 'right')

    # --- V-bar: [x_min, cx] x [y_min, y_max] ---
    xy_pde_v = sample_interior((x_min, cx), (y_min, y_max), n_domain)

    # Traction BCs for V-bar:
    # Top (y=y_max): loaded iff load_edge=='top'. Normal = (0, 1).
    n_top = n_bc // 3
    xy_top = sample_edge((x_min, cx), (y_min, y_max), 'top', n_top)
    top_tx = tx_val if top_loaded else 0.0
    top_ty = ty_val if top_loaded else 0.0
    t_top = np.column_stack([np.full(n_top, top_tx),
                             np.full(n_top, top_ty)]).astype(np.float32)
    n_top_arr = np.column_stack([np.zeros(n_top), np.ones(n_top)]).astype(np.float32)

    # Left (x=x_min): traction-free, normal = (-1, 0)
    n_left = n_bc // 3
    xy_left = sample_edge((x_min, cx), (y_min, y_max), 'left', n_left)
    t_left = np.zeros((n_left, 2), dtype=np.float32)
    n_left_arr = np.column_stack([np.full(n_left, -1.0), np.zeros(n_left)]).astype(np.float32)

    # Right (x=cx, y > cy): traction-free inner edge, normal = (1, 0)
    n_right_inner = n_bc // 3
    xy_right_inner = sample_partial_edge((x_min, cx), (y_min, y_max),
                                         'right', (cy, y_max), n_right_inner)
    t_right_inner = np.zeros((n_right_inner, 2), dtype=np.float32)
    n_right_inner_arr = np.column_stack([np.ones(n_right_inner),
                                         np.zeros(n_right_inner)]).astype(np.float32)

    xy_trac_v = np.concatenate([xy_top, xy_left, xy_right_inner], axis=0)
    trac_vals_v = np.concatenate([t_top, t_left, t_right_inner], axis=0)
    trac_normals_v = np.concatenate([n_top_arr, n_left_arr, n_right_inner_arr], axis=0)

    # --- H-bar: [x_min, x_max] x [y_min, cy] ---
    xy_pde_h = sample_interior((x_min, x_max), (y_min, cy), n_domain)

    # Traction BCs for H-bar:
    # Left (x=x_min): traction-free, normal = (-1, 0)
    n_left_h = n_bc // 3
    xy_left_h = sample_edge((x_min, x_max), (y_min, cy), 'left', n_left_h)
    t_left_h = np.zeros((n_left_h, 2), dtype=np.float32)
    n_left_h_arr = np.column_stack([np.full(n_left_h, -1.0),
                                     np.zeros(n_left_h)]).astype(np.float32)

    # Right (x=x_max): loaded iff load_edge=='right'. Normal = (1, 0).
    n_right_h = n_bc // 3
    xy_right_h = sample_edge((x_min, x_max), (y_min, cy), 'right', n_right_h)
    right_tx = tx_val if right_loaded else 0.0
    right_ty = ty_val if right_loaded else 0.0
    t_right_h = np.column_stack([np.full(n_right_h, right_tx),
                                 np.full(n_right_h, right_ty)]).astype(np.float32)
    n_right_h_arr = np.column_stack([np.ones(n_right_h),
                                      np.zeros(n_right_h)]).astype(np.float32)

    # Top (y=cy, x > cx): traction-free inner edge, normal = (0, 1)
    n_top_inner = n_bc // 3
    xy_top_inner = sample_partial_edge((x_min, x_max), (y_min, cy),
                                       'top', (cx, x_max), n_top_inner)
    t_top_inner = np.zeros((n_top_inner, 2), dtype=np.float32)
    n_top_inner_arr = np.column_stack([np.zeros(n_top_inner),
                                       np.ones(n_top_inner)]).astype(np.float32)

    xy_trac_h = np.concatenate([xy_left_h, xy_right_h, xy_top_inner], axis=0)
    trac_vals_h = np.concatenate([t_left_h, t_right_h, t_top_inner], axis=0)
    trac_normals_h = np.concatenate([n_left_h_arr, n_right_h_arr,
                                     n_top_inner_arr], axis=0)

    # --- Coupling points in overlap [x_min, cx] x [y_min, cy] ---
    xy_coupling = sample_interior((x_min, cx), (y_min, cy), args.n_coupling)

    # --- Interface points for traction matching ---
    # Along x = cx, y in [y_min, cy]: V-bar right cut / H-bar interior
    n_iface_v = args.n_interface // 2
    xy_iface_v = sample_partial_edge((x_min, cx), (y_min, y_max),
                                     'right', (y_min, cy), n_iface_v)
    # Normal pointing right (+x) from V-bar perspective
    n_iface_v_arr = np.column_stack([np.ones(n_iface_v),
                                     np.zeros(n_iface_v)]).astype(np.float32)

    # Along y = cy, x in [x_min, cx]: H-bar top cut / V-bar interior
    n_iface_h = args.n_interface - n_iface_v
    xy_iface_h = sample_partial_edge((x_min, x_max), (y_min, cy),
                                     'top', (x_min, cx), n_iface_h)
    # Normal pointing up (+y) from H-bar perspective
    n_iface_h_arr = np.column_stack([np.zeros(n_iface_h),
                                     np.ones(n_iface_h)]).astype(np.float32)

    xy_iface = np.concatenate([xy_iface_v, xy_iface_h], axis=0)
    normals_iface = np.concatenate([n_iface_v_arr, n_iface_h_arr], axis=0)

    return (xy_pde_v, xy_pde_h,
            xy_trac_v, trac_vals_v, trac_normals_v,
            xy_trac_h, trac_vals_h, trac_normals_h,
            xy_coupling, xy_iface, normals_iface)


def main():
    parser = argparse.ArgumentParser(
        description='Train decomposed PINN for L-bracket')
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--n-domain', type=int, default=10000)
    parser.add_argument('--n-boundary', type=int, default=2000)
    parser.add_argument('--n-coupling', type=int, default=5000)
    parser.add_argument('--n-interface', type=int, default=2000)
    parser.add_argument('--layers', type=int, nargs='+', default=[128, 128, 128, 128])
    parser.add_argument('--activation', type=str, default='tanh')
    parser.add_argument('--output-dir', type=str, default='./models')

    parser.add_argument('--E', type=float, default=1.0)
    parser.add_argument('--nu', type=float, default=0.3)
    parser.add_argument('--applied-stress', type=float, default=10.0,
                        help='Legacy load magnitude; used when --load-magnitude absent')
    parser.add_argument('--load-edge', type=str, default='top',
                        choices=['top', 'right'],
                        help='Edge on which traction is applied')
    parser.add_argument('--load-magnitude', type=float, default=None,
                        help='Traction magnitude. Defaults to --applied-stress')
    parser.add_argument('--load-angle', type=float, default=None,
                        help='Traction direction (deg, CCW from +x). Omit for legacy default')
    parser.add_argument('--tag', type=str, default=None,
                        help='Extra suffix for output weights (e.g. "top_m10_a180")')

    parser.add_argument('--x-min', type=float, default=0.0)
    parser.add_argument('--x-max', type=float, default=2.0)
    parser.add_argument('--y-min', type=float, default=0.0)
    parser.add_argument('--y-max', type=float, default=2.0)
    parser.add_argument('--corner-x', type=float, default=1.0)
    parser.add_argument('--corner-y', type=float, default=1.0)

    parser.add_argument('--w-pde', type=float, default=1.0)
    parser.add_argument('--w-trac', type=float, default=1.0)
    parser.add_argument('--w-coupling', type=float, default=10.0)
    parser.add_argument('--w-trac-match', type=float, default=10.0)

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    cx, cy = args.corner_x, args.corner_y
    x_min, x_max = args.x_min, args.x_max
    y_min, y_max = args.y_min, args.y_max

    print("=" * 60)
    print("DECOMPOSED PINN TRAINING: L-bracket")
    print("=" * 60)
    print(f"  V-bar: [{x_min}, {cx}] x [{y_min}, {y_max}]")
    print(f"  H-bar: [{x_min}, {x_max}] x [{y_min}, {cy}]")
    print(f"  Overlap: [{x_min}, {cx}] x [{y_min}, {cy}]")
    print()

    # Build two models
    network = Network()
    input_range_v = [(x_min, cx), (y_min, y_max)]
    input_range_h = [(x_min, x_max), (y_min, cy)]

    model_v = network.build(num_inputs=2, layers=args.layers,
                            activation=args.activation, num_outputs=2,
                            input_range=input_range_v,
                            hard_bc='bottom_fixed')
    model_h = network.build(num_inputs=2, layers=args.layers,
                            activation=args.activation, num_outputs=2,
                            input_range=input_range_h,
                            hard_bc='bottom_fixed')

    print("V-bar model:")
    model_v.summary()
    print("\nH-bar model:")
    model_h.summary()

    # Training data
    print("\nGenerating training data...")
    data = create_training_data(args)
    print(f"  V-bar PDE points: {len(data[0])}")
    print(f"  H-bar PDE points: {len(data[1])}")
    print(f"  V-bar traction BC points: {len(data[2])}")
    print(f"  H-bar traction BC points: {len(data[5])}")
    print(f"  Coupling points: {len(data[8])}")
    print(f"  Interface traction matching points: {len(data[9])}")

    # Train
    trainer = DecomposedTrainer(
        model_v, model_h,
        E=args.E, nu=args.nu, lr=args.lr, epochs=args.epochs,
        w_pde=args.w_pde, w_trac=args.w_trac, w_coupling=args.w_coupling,
        w_trac_match=args.w_trac_match,
        grad_clip_norm=args.grad_clip if args.grad_clip > 0 else None,
    )
    print("\nTraining...")
    history = trainer.train(data, epochs=args.epochs)

    # Save. Encode load config in filename so different loads don't collide.
    tx_val, ty_val = resolve_load_vector(args)
    suffix = args.tag or f'{args.load_edge}_tx{tx_val:.3g}_ty{ty_val:.3g}'
    v_path = os.path.join(args.output_dir, f'pinn_l_bracket_vbar_{suffix}.weights.h5')
    h_path = os.path.join(args.output_dir, f'pinn_l_bracket_hbar_{suffix}.weights.h5')
    model_v.save_weights(v_path)
    model_h.save_weights(h_path)
    print(f"\nSaved V-bar model to {v_path}")
    print(f"Saved H-bar model to {h_path}")

    history_path = os.path.join(args.output_dir,
                                f'pinn_l_bracket_decomposed_history_{suffix}.npz')
    np.savez(history_path, **history)

    # Quick visualization: blended solution on L-bracket grid
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
            x_domain=(x_min, x_max), y_domain=(y_min, y_max),
            corner_x=cx, corner_y=cy,
            applied_stress=args.applied_stress,
            load_edge=args.load_edge,
            load_tx=tx_val, load_ty=ty_val,
        )

        ux, uy = blend_solutions(model_v, model_h, X, Y, layout, cx, cy)
        disp = np.sqrt(ux ** 2 + uy ** 2)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for ax, field, title in zip(axes, [ux, uy, disp],
                                     ['$u_x$', '$u_y$', '$|u|$']):
            masked = np.ma.masked_where(layout == 0, field)
            cf = ax.contourf(X, Y, masked, levels=50, cmap='RdBu_r')
            plt.colorbar(cf, ax=ax)
            ax.set_aspect('equal')
            ax.set_title(title)
            ax.set_xlabel('x'); ax.set_ylabel('y')
        plt.suptitle('Decomposed PINN — Blended Solution', fontsize=14)
        plt.tight_layout()
        fig_path = os.path.join(args.output_dir, 'pinn_l_bracket_decomposed.pdf')
        plt.savefig(fig_path, dpi=1200, bbox_inches='tight')
        print(f"Saved solution plot to {fig_path}")

        # Loss plot
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(history['total'], label='Total')
        ax.semilogy(history['pde_v'], label='PDE V-bar')
        ax.semilogy(history['pde_h'], label='PDE H-bar')
        ax.semilogy(history['trac_v'], label='Trac V-bar')
        ax.semilogy(history['trac_h'], label='Trac H-bar')
        ax.semilogy(history['coupling'], label='Coupling')
        ax.semilogy(history['trac_match'], label='Trac Match')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.set_title('Decomposed PINN Training Loss')
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig_path = os.path.join(args.output_dir, 'pinn_l_bracket_decomposed_loss.pdf')
        plt.savefig(fig_path, dpi=1200)
        print(f"Saved loss plot to {fig_path}")
        plt.show()
    except Exception as e:
        print(f"Plotting skipped: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
