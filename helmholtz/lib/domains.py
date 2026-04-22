"""
Domain setup for 2D Helmholtz benchmarks.

Experiment 1 uses only `create_square` (full square, Dirichlet on all 4 edges).
Experiment 2 will add `create_square_with_hole` following the same return
convention so downstream scripts stay uniform.
"""

import numpy as np


def create_square(Nx=201, Ny=201, x_domain=(0.0, 1.0), y_domain=(0.0, 1.0)):
    """Regular-grid square domain with Dirichlet BC on the outer boundary.

    Returns
    -------
    X, Y : (Ny, Nx) meshgrid
    layout : (Ny, Nx) binary; 1 = interior cell (all 1s for the full square)
    dirichlet_mask : (Ny, Nx) binary; 1 on the 4 outer edges
    bc_u : (Ny, Nx) float32; Dirichlet values (zeros here; caller fills in
        u_exact for the manufactured-solution sanity check)
    """
    x = np.linspace(x_domain[0], x_domain[1], Nx, dtype=np.float32)
    y = np.linspace(y_domain[0], y_domain[1], Ny, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    layout = np.ones((Ny, Nx), dtype=np.float32)

    dirichlet_mask = np.zeros((Ny, Nx), dtype=np.float32)
    dirichlet_mask[0, :] = 1.0
    dirichlet_mask[-1, :] = 1.0
    dirichlet_mask[:, 0] = 1.0
    dirichlet_mask[:, -1] = 1.0

    bc_u = np.zeros((Ny, Nx), dtype=np.float32)
    return X, Y, layout, dirichlet_mask, bc_u


def manufactured_solution(X, Y, k):
    """Exact solution and corresponding forcing for `-Delta u - k^2 u = f`.

    u*(x, y) = sin(k*x) sin(k*y)
    => Delta u* = -2 k^2 u*
    => -Delta u* - k^2 u* = 2 k^2 u* - k^2 u* = k^2 u*

    So f(x, y) = k^2 sin(k*x) sin(k*y).
    """
    u_star = np.sin(k * X) * np.sin(k * Y)
    f_source = (k ** 2) * u_star
    return u_star.astype(np.float32), f_source.astype(np.float32)
