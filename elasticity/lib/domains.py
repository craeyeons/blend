"""
Domain setups for 2D linear elasticity problems.

Each setup returns coordinate grids, layout mask, boundary condition masks
and values in the same format used by the router.

Conventions
-----------
- layout: 1 = material, 0 = void/hole
- disp_bc_mask: 1 where displacement BCs are prescribed
- trac_bc_mask: 1 where traction BCs are prescribed
- bc_ux, bc_uy: prescribed displacement values (nonzero only where disp_bc_mask=1)
- bc_tx, bc_ty: prescribed traction values (nonzero only where trac_bc_mask=1)
"""

import numpy as np
from scipy import ndimage


def create_plate_with_hole(Nx=200, Ny=200,
                           x_domain=(-2.0, 2.0), y_domain=(-2.0, 2.0),
                           hole_center=(0.0, 0.0), hole_radius=0.5,
                           applied_stress=1.0):
    """
    Rectangular plate with a central circular hole under uniaxial tension.

    Boundary conditions:
        - Left edge (x=x_min): ux = 0 (roller), free in y
        - Bottom edge (y=y_min): uy = 0 (roller), free in x
        - Right edge (x=x_max): uniform traction tx = applied_stress
        - Top edge (y=y_max): traction-free (ty = 0)
        - Hole surface: traction-free

    Using symmetry: only model quarter-plate with rollers on symmetry planes.

    Parameters
    ----------
    Nx, Ny : int
        Grid dimensions.
    x_domain, y_domain : tuple
        Physical domain bounds.
    hole_center : tuple
        (cx, cy) center of hole.
    hole_radius : float
        Hole radius.
    applied_stress : float
        Applied tensile stress on right boundary.

    Returns
    -------
    X, Y, layout, disp_bc_mask, trac_bc_mask, bc_ux, bc_uy, bc_tx, bc_ty
    """
    x_min, x_max = x_domain
    y_min, y_max = y_domain
    cx, cy = hole_center

    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    X, Y = np.meshgrid(x, y)  # (Ny, Nx)

    # Layout: 1 = material, 0 = hole
    dist_sq = (X - cx) ** 2 + (Y - cy) ** 2
    layout = (dist_sq > hole_radius ** 2).astype(np.float32)

    disp_bc_mask = np.zeros((Ny, Nx), dtype=np.float32)
    trac_bc_mask = np.zeros((Ny, Nx), dtype=np.float32)
    bc_ux = np.zeros((Ny, Nx), dtype=np.float32)
    bc_uy = np.zeros((Ny, Nx), dtype=np.float32)
    bc_tx = np.zeros((Ny, Nx), dtype=np.float32)
    bc_ty = np.zeros((Ny, Nx), dtype=np.float32)

    # Left edge: ux = 0 (symmetry roller)
    disp_bc_mask[:, 0] = 1.0
    bc_ux[:, 0] = 0.0

    # Bottom edge: uy = 0 (symmetry roller)
    disp_bc_mask[0, :] = 1.0
    bc_uy[0, :] = 0.0

    # Right edge: uniform traction tx = applied_stress
    trac_bc_mask[:, -1] = 1.0
    bc_tx[:, -1] = applied_stress
    bc_ty[:, -1] = 0.0

    # Top edge: traction-free
    trac_bc_mask[-1, :] = 1.0
    bc_tx[-1, :] = 0.0
    bc_ty[-1, :] = 0.0

    # Hole boundary: traction-free (mark adjacent material points)
    hole_mask = dist_sq <= hole_radius ** 2
    dilated = ndimage.binary_dilation(hole_mask)
    hole_boundary = dilated & ~hole_mask
    trac_bc_mask[hole_boundary] = 1.0
    bc_tx[hole_boundary] = 0.0
    bc_ty[hole_boundary] = 0.0

    return X, Y, layout, disp_bc_mask, trac_bc_mask, bc_ux, bc_uy, bc_tx, bc_ty


def create_l_bracket(Nx=200, Ny=200,
                     x_domain=(0.0, 2.0), y_domain=(0.0, 2.0),
                     corner_x=1.0, corner_y=1.0,
                     applied_stress=1.0):
    """
    L-shaped bracket under load.

    Geometry (origin at bottom-left):
        Full square domain [0, 2] x [0, 2] with upper-right quadrant
        [corner_x, x_max] x [corner_y, y_max] removed.

    Boundary conditions:
        - Bottom edge (y=0): fully fixed (ux=0, uy=0)
        - Top edge of upper arm (y=y_max, x < corner_x): traction tx = -applied_stress (pull left)
        - All other outer edges and re-entrant corner: traction-free

    Parameters
    ----------
    Nx, Ny : int
        Grid dimensions.
    x_domain, y_domain : tuple
        Physical domain bounds.
    corner_x, corner_y : float
        Coordinates of the re-entrant corner.
    applied_stress : float
        Applied traction on the loaded edge.

    Returns
    -------
    X, Y, layout, disp_bc_mask, trac_bc_mask, bc_ux, bc_uy, bc_tx, bc_ty
    """
    x_min, x_max = x_domain
    y_min, y_max = y_domain

    x = np.linspace(x_min, x_max, Nx)
    y = np.linspace(y_min, y_max, Ny)
    X, Y = np.meshgrid(x, y)  # (Ny, Nx)

    # Layout: L-shape (remove upper-right block)
    layout = np.ones((Ny, Nx), dtype=np.float32)
    layout[(Y > corner_y) & (X > corner_x)] = 0.0

    disp_bc_mask = np.zeros((Ny, Nx), dtype=np.float32)
    trac_bc_mask = np.zeros((Ny, Nx), dtype=np.float32)
    bc_ux = np.zeros((Ny, Nx), dtype=np.float32)
    bc_uy = np.zeros((Ny, Nx), dtype=np.float32)
    bc_tx = np.zeros((Ny, Nx), dtype=np.float32)
    bc_ty = np.zeros((Ny, Nx), dtype=np.float32)

    # Bottom edge: fully fixed
    disp_bc_mask[0, :] = layout[0, :]
    bc_ux[0, :] = 0.0
    bc_uy[0, :] = 0.0

    # Right edge of lower arm (x=x_max, y < corner_y): traction-free
    lower_arm_right = (np.abs(X - x_max) < (x[1] - x[0]) / 2) & (Y <= corner_y)
    trac_bc_mask[lower_arm_right & (layout > 0)] = 1.0

    # Left edge: traction-free
    left_edge = np.abs(X - x_min) < (x[1] - x[0]) / 2
    trac_bc_mask[left_edge & (layout > 0)] = 1.0

    # Top edge of upper arm (y=y_max, x < corner_x): pull left
    top_edge = (np.abs(Y - y_max) < (y[1] - y[0]) / 2) & (X <= corner_x)
    trac_bc_mask[top_edge & (layout > 0)] = 1.0
    bc_tx[top_edge & (layout > 0)] = -applied_stress

    # Inner edges of L (re-entrant corner region): traction-free
    # Horizontal inner edge: y ~ corner_y, x > corner_x
    h_inner = (np.abs(Y - corner_y) < (y[1] - y[0]) / 2) & (X >= corner_x)
    trac_bc_mask[h_inner & (layout > 0)] = 1.0
    # Vertical inner edge: x ~ corner_x, y > corner_y
    v_inner = (np.abs(X - corner_x) < (x[1] - x[0]) / 2) & (Y >= corner_y)
    trac_bc_mask[v_inner & (layout > 0)] = 1.0

    # Top of right arm (y ~ corner_y where layout transitions)
    # Already covered above

    # Don't double-count: displacement BC takes priority
    trac_bc_mask[disp_bc_mask > 0] = 0.0

    return X, Y, layout, disp_bc_mask, trac_bc_mask, bc_ux, bc_uy, bc_tx, bc_ty
