"""
One-shot Schwarz PINN+FEM hybrid for 2D Helmholtz.

The router produces a per-cell logit on the regular grid: positive = reject
PINN, negative = accept PINN. Accepted cells are mapped to their nearest
FEM mesh vertex, and the PINN's value there is pinned as an additional
Dirichlet constraint. The remaining (rejected) interior DOFs are solved
via a single FEM `spsolve`, on a system that is smaller than the full-FEM
baseline by the number of accepted vertices.

Speedup story: more accepted cells -> fewer interior DOFs -> smaller
linear system -> faster solve.
"""

import time
import numpy as np
import tensorflow as tf
from scipy.ndimage import binary_opening


def _predict_pinn_on_points(pinn_model, pts_xy, batch_size=65536):
    """Vectorized PINN forward on (N, 2) numpy. Returns (N,) float32."""
    pts = np.asarray(pts_xy, dtype=np.float32)
    out = np.empty(pts.shape[0], dtype=np.float32)
    for i in range(0, pts.shape[0], batch_size):
        chunk = pts[i:i + batch_size]
        pred = pinn_model(tf.constant(chunk), training=False).numpy()
        out[i:i + batch_size] = pred[:, 0]
    return out


def threshold_for_coverage(logits, layout, coverage_frac):
    """Pick the threshold that gives the requested rejection fraction
    (fraction of solid cells where logit >= threshold). coverage_frac
    in [0, 1]; 0 = accept all PINN, 1 = reject all (full FEM)."""
    solid = layout > 0
    lv = logits[solid]
    if coverage_frac <= 0.0:
        return float(lv.max()) + 1e-6  # nothing gets rejected
    if coverage_frac >= 1.0:
        return float(lv.min()) - 1e-6  # everything gets rejected
    # We want the top `coverage_frac` of lv values to be >= threshold.
    k = int(np.ceil(coverage_frac * lv.size))
    k = max(1, min(k, lv.size))
    sorted_desc = np.sort(lv)[::-1]
    return float(sorted_desc[k - 1])


def solve_hybrid_schwarz(solver, pinn_model, router_model,
                         f_callable, g_callable,
                         X, Y, layout,
                         f_grid, pinn_u_grid, residual_grid,
                         ete_grid=None,
                         threshold=0.0,
                         reuse_logits=None):
    """Run one hybrid solve.

    Parameters
    ----------
    solver : HelmholtzSolver
    pinn_model : keras.Model  (maps (N, 2) -> (N, 1))
    router_model : RouterCNN
    f_callable, g_callable : FEM forcing / original Dirichlet data callables
    X, Y : (Ny, Nx) grid meshgrid
    layout : (Ny, Nx) binary (1 = solid, 0 = hole)
    f_grid, pinn_u_grid, residual_grid : precomputed channels for the
        router input
    threshold : float
        Router logit threshold. Cells with logit < threshold are accepted
        (PINN pinned). Logit >= threshold -> rejected (FEM solves).
    reuse_logits : optional (Ny, Nx) precomputed logit field. If provided,
        skips the router forward pass (useful in threshold sweeps).

    Returns
    -------
    dict with keys: u_grid, u_dof, logits, accept_mask, coverage_pct,
    router_time_s, pinn_pin_time_s, solve_time_s.
    """
    from .router import create_router_input

    t0 = time.perf_counter()
    if reuse_logits is None:
        inputs = create_router_input(layout, f_grid, pinn_u_grid,
                                     residual_grid, ete=ete_grid)
        logits = router_model(tf.constant(inputs, dtype=tf.float32),
                              training=False)[0, :, :, 0].numpy()
    else:
        logits = reuse_logits
    router_time_s = time.perf_counter() - t0

    # Binary decision per cell. Only solid cells can be accepted.
    # Apply morphological opening to the reject mask to remove isolated
    # speckle — small scattered FEM cells produce thin rings of pinned
    # PINN BCs around 1-cell interiors, which reproduces PINN's error
    # instead of correcting it. Opening (erode then dilate) with a 3x3
    # cross absorbs these back into the PINN-accepted region.
    solid = layout > 0
    reject_raw = (logits >= threshold) & solid
    reject_opened = binary_opening(reject_raw, structure=np.ones((3, 3)),
                                   iterations=1)
    accept_mask = solid & ~reject_opened

    # Map accepted grid cells -> nearest FEM vertex ids.
    t1 = time.perf_counter()
    vertex_ids_grid = solver.vertex_ids_for_grid(X, Y)
    accepted_vertex_ids = np.unique(vertex_ids_grid[accept_mask])
    # Drop any vertex already in the original fixed set; we don't override
    # the genuine Dirichlet DOFs with PINN values.
    base_fixed = set(solver.fixed_dof_ids.tolist())
    accepted_vertex_ids = np.array(
        [v for v in accepted_vertex_ids if v not in base_fixed],
        dtype=np.int64)

    if accepted_vertex_ids.size > 0:
        vxy = solver.mesh.p[:, accepted_vertex_ids].T  # (M, 2)
        pinn_vals = _predict_pinn_on_points(pinn_model, vxy)
    else:
        pinn_vals = np.zeros((0,), dtype=np.float32)
    pinn_pin_time_s = time.perf_counter() - t1

    u_dof, solve_time_s = solver.solve_with_extra_dirichlet(
        f_callable, g_callable, accepted_vertex_ids, pinn_vals)

    u_grid = solver.interp_to_grid(u_dof, X, Y)

    n_solid = int(solid.sum())
    coverage_pct = 100.0 * float(reject_opened[solid].sum()) / max(n_solid, 1)

    return {
        'u_grid': u_grid,
        'u_dof': u_dof,
        'logits': logits,
        'accept_mask': accept_mask.astype(np.int32),
        'coverage_pct': coverage_pct,
        'router_time_s': router_time_s,
        'pinn_pin_time_s': pinn_pin_time_s,
        'solve_time_s': solve_time_s,
        'n_accepted_dofs': int(accepted_vertex_ids.size),
    }


def rmse(pred, ref):
    return float(np.sqrt(np.mean((pred - ref) ** 2)))
