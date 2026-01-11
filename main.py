import jax
import jax.numpy as jnp
from jax import jit, lax
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import time
from scipy import ndimage
import tensorflow as tf
from lib.network import Network

# Enable 64-bit precision for better accuracy
jax.config.update("jax_enable_x64", True)

def solve_navier_stokes_jax(Re=100, N=100, max_iter=20000, tol=1e-6):
    """
    Solve 2D lid-driven cavity using JAX with finite differences (SIMPLE-like method)
    
    Parameters:
    -----------
    Re : float
        Reynolds number
    N : int
        Number of grid points per side
    max_iter : int
        Maximum iterations
    tol : float
        Convergence tolerance
    """
    print(f"JAX devices: {jax.devices()}")
    print(f"Using device: {jax.devices()[0]}")
    
    # Grid setup
    L = 1.0
    dx = L / (N - 1)
    dy = dx
    dt = 0.001 * min(dx, dy)  # CFL-based time step
    
    nu = 1.0 / Re  # Kinematic viscosity
    
    # Initialize fields on GPU
    u = jnp.zeros((N, N))   # x-velocity
    v = jnp.zeros((N, N))   # y-velocity
    p = jnp.zeros((N, N))   # pressure
    
    # Boundary conditions: lid moves with u=1 at y=1 (top)
    u = u.at[-1, :].set(1.0)  # Top lid moves right
    
    @jit
    def apply_boundary_conditions(u, v):
        """Apply velocity boundary conditions"""
        # Top lid (y=1): u=1, v=0
        u = u.at[-1, :].set(1.0)
        v = v.at[-1, :].set(0.0)
        
        # Bottom wall (y=0): u=0, v=0
        u = u.at[0, :].set(0.0)
        v = v.at[0, :].set(0.0)
        
        # Left wall (x=0): u=0, v=0
        u = u.at[:, 0].set(0.0)
        v = v.at[:, 0].set(0.0)
        
        # Right wall (x=1): u=0, v=0
        u = u.at[:, -1].set(0.0)
        v = v.at[:, -1].set(0.0)
        
        return u, v
    
    @jit
    def laplacian(f, dx, dy):
        """Compute Laplacian using central differences"""
        lap = jnp.zeros_like(f)
        lap = lap.at[1:-1, 1:-1].set(
            (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, :-2]) / dx**2 +
            (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1]) / dy**2
        )
        return lap
    
    @jit
    def divergence(u, v, dx, dy):
        """Compute divergence of velocity field"""
        div = jnp.zeros_like(u)
        div = div.at[1:-1, 1:-1].set(
            (u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx) +
            (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dy)
        )
        return div
    
    @jit
    def convection(u, v, f, dx, dy):
        """Compute convective term (u·∇)f using upwind scheme"""
        # Central differences for derivatives
        dfdx = jnp.zeros_like(f)
        dfdy = jnp.zeros_like(f)
        
        dfdx = dfdx.at[1:-1, 1:-1].set(
            (f[1:-1, 2:] - f[1:-1, :-2]) / (2*dx)
        )
        dfdy = dfdy.at[1:-1, 1:-1].set(
            (f[2:, 1:-1] - f[:-2, 1:-1]) / (2*dy)
        )
        
        return u * dfdx + v * dfdy
    
    @jit
    def pressure_poisson_iteration(p, rhs, dx, dy):
        """Single Jacobi iteration for pressure Poisson equation"""
        p_new = jnp.zeros_like(p)
        p_new = p_new.at[1:-1, 1:-1].set(
            0.25 * (
                p[1:-1, 2:] + p[1:-1, :-2] +
                p[2:, 1:-1] + p[:-2, 1:-1] -
                dx**2 * rhs[1:-1, 1:-1]
            )
        )
        # Neumann BC: dp/dn = 0 on walls
        p_new = p_new.at[0, :].set(p_new[1, :])
        p_new = p_new.at[-1, :].set(p_new[-2, :])
        p_new = p_new.at[:, 0].set(p_new[:, 1])
        p_new = p_new.at[:, -1].set(p_new[:, -2])
        return p_new
    
    @jit
    def solve_pressure_poisson(p, rhs, dx, dy, n_iter=50):
        """Solve pressure Poisson equation with Jacobi iterations"""
        def body_fn(i, p):
            return pressure_poisson_iteration(p, rhs, dx, dy)
        return lax.fori_loop(0, n_iter, body_fn, p)
    
    @jit
    def step(u, v, p, dt, dx, dy, nu):
        """Single time step using fractional step method"""
        # Step 1: Compute intermediate velocity (without pressure)
        conv_u = convection(u, v, u, dx, dy)
        conv_v = convection(u, v, v, dx, dy)
        lap_u = laplacian(u, dx, dy)
        lap_v = laplacian(v, dx, dy)
        
        u_star = u + dt * (-conv_u + nu * lap_u)
        v_star = v + dt * (-conv_v + nu * lap_v)
        
        # Apply BCs to intermediate velocity
        u_star, v_star = apply_boundary_conditions(u_star, v_star)
        
        # Step 2: Solve pressure Poisson equation
        div_star = divergence(u_star, v_star, dx, dy)
        rhs = div_star / dt
        p = solve_pressure_poisson(p, rhs, dx, dy, n_iter=100)
        
        # Step 3: Correct velocity
        dpdx = jnp.zeros_like(p)
        dpdy = jnp.zeros_like(p)
        dpdx = dpdx.at[1:-1, 1:-1].set((p[1:-1, 2:] - p[1:-1, :-2]) / (2*dx))
        dpdy = dpdy.at[1:-1, 1:-1].set((p[2:, 1:-1] - p[:-2, 1:-1]) / (2*dy))
        
        u_new = u_star - dt * dpdx
        v_new = v_star - dt * dpdy
        
        # Apply BCs to corrected velocity
        u_new, v_new = apply_boundary_conditions(u_new, v_new)
        
        return u_new, v_new, p
    
    @jit
    def compute_residual(u_new, u_old, v_new, v_old):
        """Compute convergence residual"""
        diff_u = jnp.sum((u_new - u_old)**2)
        diff_v = jnp.sum((v_new - v_old)**2)
        return jnp.sqrt(diff_u + diff_v)
    
    # Apply initial BCs
    u, v = apply_boundary_conditions(u, v)
    
    print("Solving Navier-Stokes equations with JAX (Fractional Step method)...")
    print(f"Reynolds number: {Re}")
    print(f"Grid size: {N}x{N}")
    print(f"Time step: {dt:.6e}")
    print("-" * 50)
    
    # Time stepping loop
    for n in range(max_iter):
        u_old, v_old = u, v
        u, v, p = step(u, v, p, dt, dx, dy, nu)
        
        # Check convergence every 50 iterations
        if n % 50 == 0:
            residual = compute_residual(u, u_old, v, v_old)
            residual_val = float(residual)
            print(f"Iteration {n}, Residual: {residual_val:.6e}")
            
            if residual_val < tol:
                print(f"\nConverged at iteration {n}")
                break
    else:
        print(f"\nReached maximum iterations ({max_iter})")
    
    # Convert to numpy for plotting
    return np.array(u), np.array(v), np.array(p)


def plot_solution(u, v, p):
    """Plot velocity and pressure fields"""
    N = u.shape[0]
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    
    vel_mag = np.sqrt(u**2 + v**2)
    
    def contour(ax, x, y, z, title):
        cf = ax.contourf(x, y, z, levels=50, cmap='jet')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax)
    
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    contour(ax1, X, Y, vel_mag, '|u| (velocity magnitude)')
    
    ax2 = fig.add_subplot(gs[0, 1])
    contour(ax2, X, Y, p, 'p (pressure)')
    
    ax3 = fig.add_subplot(gs[1, 0])
    contour(ax3, X, Y, u, 'u (x-velocity)')
    
    ax4 = fig.add_subplot(gs[1, 1])
    contour(ax4, X, Y, v, 'v (y-velocity)')
    
    plt.tight_layout()
    fig.savefig('cavity_flow_jax.png', dpi=300)
    plt.show()


def solve_hybrid_pinn_cfd(network, uv_func, mask, Re=100, N=100, max_iter=20000, tol=1e-6):
    """
    Hybrid solver that uses PINN in certain regions and CFD for the rest.
    
    The PINN output at the interface serves as boundary conditions for the CFD solver.
    The mask indicates where to use CFD (mask=1) vs PINN (mask=0).
    
    Parameters:
    -----------
    network : 
        PINN model that outputs (psi, p) given (x, y) coordinates.
        Must have a predict() method and be callable for gradient computation.
    uv_func : callable
        Function to compute (u, v) from the network: uv_func(network, xy) -> (u, v)
    mask : ndarray of shape (N, N)
        Binary mask where 1 indicates CFD region, 0 indicates PINN region.
        The interface between regions will use PINN values as Dirichlet BCs.
    Re : float
        Reynolds number
    N : int
        Number of grid points per side (must match mask dimensions)
    max_iter : int
        Maximum iterations for CFD solver
    tol : float
        Convergence tolerance
        
    Returns:
    --------
    u, v, p : ndarray
        Velocity and pressure fields combining PINN and CFD solutions
    """
    print(f"JAX devices: {jax.devices()}")
    print(f"Using device: {jax.devices()[0]}")
    
    # Validate mask dimensions
    assert mask.shape == (N, N), f"Mask shape {mask.shape} must match grid size ({N}, {N})"
    
    # Grid setup
    L = 1.0
    dx = L / (N - 1)
    dy = dx
    dt = 0.001 * min(dx, dy)
    nu = 1.0 / Re
    
    # Create coordinate grid
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    xy = np.stack([X.flatten(), Y.flatten()], axis=-1)
    
    # Query PINN for the entire domain
    print("Querying PINN for initial field values...")
    psi_p = network.predict(xy, batch_size=len(xy))
    psi_pinn = psi_p[..., 0].reshape(X.shape)
    p_pinn = psi_p[..., 1].reshape(X.shape)
    
    # Compute velocities from PINN
    u_pinn, v_pinn = uv_func(network, xy)
    u_pinn = u_pinn.reshape(X.shape)
    v_pinn = v_pinn.reshape(X.shape)
    
    # Find interface: boundary between PINN and CFD regions
    # Interface points are CFD cells that have at least one PINN neighbor
    mask_float = mask.astype(float)
    
    # Dilate the PINN region (mask=0) to find interface
    pinn_region = (mask == 0).astype(float)
    dilated_pinn = ndimage.binary_dilation(pinn_region, iterations=1).astype(float)
    
    # Interface is where CFD region meets dilated PINN region
    interface_mask = (mask == 1) & (dilated_pinn == 1)
    
    # Also include domain boundaries where they intersect CFD region
    domain_boundary = np.zeros_like(mask, dtype=bool)
    domain_boundary[0, :] = True   # Bottom
    domain_boundary[-1, :] = True  # Top
    domain_boundary[:, 0] = True   # Left
    domain_boundary[:, -1] = True  # Right
    
    # Combined boundary for CFD: interface + domain boundaries in CFD region
    cfd_boundary = interface_mask | (domain_boundary & (mask == 1))
    
    # Convert to JAX arrays
    mask_jax = jnp.array(mask, dtype=jnp.float64)
    interface_jax = jnp.array(interface_mask, dtype=jnp.float64)
    cfd_boundary_jax = jnp.array(cfd_boundary, dtype=jnp.float64)
    
    # Initialize CFD fields with PINN values
    u = jnp.array(u_pinn)
    v = jnp.array(v_pinn)
    p = jnp.array(p_pinn)
    
    # Store PINN values for boundary conditions
    u_pinn_jax = jnp.array(u_pinn)
    v_pinn_jax = jnp.array(v_pinn)
    p_pinn_jax = jnp.array(p_pinn)
    
    @jit
    def apply_hybrid_boundary_conditions(u, v, mask, cfd_boundary, u_pinn, v_pinn):
        """
        Apply boundary conditions:
        - At domain walls: standard no-slip (except lid)
        - At PINN-CFD interface: use PINN values as Dirichlet BC
        - In PINN region: keep PINN values
        """
        # First, apply standard lid-driven cavity BCs at domain boundaries
        # Top lid (y=1): u=1, v=0 (only if in CFD region)
        u = jnp.where(
            (jnp.arange(u.shape[0])[:, None] == u.shape[0] - 1) & (mask == 1),
            1.0, u
        )
        v = jnp.where(
            (jnp.arange(v.shape[0])[:, None] == v.shape[0] - 1) & (mask == 1),
            0.0, v
        )
        
        # Bottom wall (y=0): u=0, v=0 (only if in CFD region)
        u = jnp.where(
            (jnp.arange(u.shape[0])[:, None] == 0) & (mask == 1),
            0.0, u
        )
        v = jnp.where(
            (jnp.arange(v.shape[0])[:, None] == 0) & (mask == 1),
            0.0, v
        )
        
        # Left wall (x=0): u=0, v=0 (only if in CFD region)
        u = jnp.where(
            (jnp.arange(u.shape[1])[None, :] == 0) & (mask == 1),
            0.0, u
        )
        v = jnp.where(
            (jnp.arange(v.shape[1])[None, :] == 0) & (mask == 1),
            0.0, v
        )
        
        # Right wall (x=1): u=0, v=0 (only if in CFD region)
        u = jnp.where(
            (jnp.arange(u.shape[1])[None, :] == u.shape[1] - 1) & (mask == 1),
            0.0, u
        )
        v = jnp.where(
            (jnp.arange(v.shape[1])[None, :] == v.shape[1] - 1) & (mask == 1),
            0.0, v
        )
        
        # At interface (CFD cells adjacent to PINN region): use PINN values
        # This creates a smooth transition
        u = jnp.where(cfd_boundary & ~(
            (jnp.arange(u.shape[0])[:, None] == 0) |
            (jnp.arange(u.shape[0])[:, None] == u.shape[0] - 1) |
            (jnp.arange(u.shape[1])[None, :] == 0) |
            (jnp.arange(u.shape[1])[None, :] == u.shape[1] - 1)
        ), u_pinn, u)
        v = jnp.where(cfd_boundary & ~(
            (jnp.arange(v.shape[0])[:, None] == 0) |
            (jnp.arange(v.shape[0])[:, None] == v.shape[0] - 1) |
            (jnp.arange(v.shape[1])[None, :] == 0) |
            (jnp.arange(v.shape[1])[None, :] == v.shape[1] - 1)
        ), v_pinn, v)
        
        # In PINN region (mask=0): always use PINN values
        u = jnp.where(mask == 0, u_pinn, u)
        v = jnp.where(mask == 0, v_pinn, v)
        
        return u, v
    
    @jit
    def laplacian(f, dx, dy):
        """Compute Laplacian using central differences"""
        lap = jnp.zeros_like(f)
        lap = lap.at[1:-1, 1:-1].set(
            (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, :-2]) / dx**2 +
            (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1]) / dy**2
        )
        return lap
    
    @jit
    def divergence(u, v, dx, dy):
        """Compute divergence of velocity field"""
        div = jnp.zeros_like(u)
        div = div.at[1:-1, 1:-1].set(
            (u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx) +
            (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dy)
        )
        return div
    
    @jit
    def convection(u, v, f, dx, dy):
        """Compute convective term (u·∇)f"""
        dfdx = jnp.zeros_like(f)
        dfdy = jnp.zeros_like(f)
        
        dfdx = dfdx.at[1:-1, 1:-1].set(
            (f[1:-1, 2:] - f[1:-1, :-2]) / (2*dx)
        )
        dfdy = dfdy.at[1:-1, 1:-1].set(
            (f[2:, 1:-1] - f[:-2, 1:-1]) / (2*dy)
        )
        
        return u * dfdx + v * dfdy
    
    @jit
    def pressure_poisson_iteration_hybrid(p, rhs, dx, dy, mask, p_pinn):
        """Single Jacobi iteration for pressure Poisson equation (hybrid version)"""
        p_new = jnp.zeros_like(p)
        p_new = p_new.at[1:-1, 1:-1].set(
            0.25 * (
                p[1:-1, 2:] + p[1:-1, :-2] +
                p[2:, 1:-1] + p[:-2, 1:-1] -
                dx**2 * rhs[1:-1, 1:-1]
            )
        )
        # Neumann BC at domain walls: dp/dn = 0
        p_new = p_new.at[0, :].set(p_new[1, :])
        p_new = p_new.at[-1, :].set(p_new[-2, :])
        p_new = p_new.at[:, 0].set(p_new[:, 1])
        p_new = p_new.at[:, -1].set(p_new[:, -2])
        
        # In PINN region: use PINN pressure
        p_new = jnp.where(mask == 0, p_pinn, p_new)
        
        return p_new
    
    @jit
    def solve_pressure_poisson_hybrid(p, rhs, dx, dy, mask, p_pinn, n_iter=50):
        """Solve pressure Poisson equation with Jacobi iterations (hybrid version)"""
        def body_fn(i, p):
            return pressure_poisson_iteration_hybrid(p, rhs, dx, dy, mask, p_pinn)
        return lax.fori_loop(0, n_iter, body_fn, p)
    
    @jit
    def step_hybrid(u, v, p, dt, dx, dy, nu, mask, cfd_boundary, u_pinn, v_pinn, p_pinn):
        """Single time step using fractional step method (hybrid version)"""
        # Step 1: Compute intermediate velocity (without pressure)
        conv_u = convection(u, v, u, dx, dy)
        conv_v = convection(u, v, v, dx, dy)
        lap_u = laplacian(u, dx, dy)
        lap_v = laplacian(v, dx, dy)
        
        u_star = u + dt * (-conv_u + nu * lap_u)
        v_star = v + dt * (-conv_v + nu * lap_v)
        
        # Only update in CFD region
        u_star = jnp.where(mask == 1, u_star, u_pinn)
        v_star = jnp.where(mask == 1, v_star, v_pinn)
        
        # Apply BCs to intermediate velocity
        u_star, v_star = apply_hybrid_boundary_conditions(
            u_star, v_star, mask, cfd_boundary, u_pinn, v_pinn
        )
        
        # Step 2: Solve pressure Poisson equation
        div_star = divergence(u_star, v_star, dx, dy)
        rhs = div_star / dt
        p = solve_pressure_poisson_hybrid(p, rhs, dx, dy, mask, p_pinn, n_iter=100)
        
        # Step 3: Correct velocity
        dpdx = jnp.zeros_like(p)
        dpdy = jnp.zeros_like(p)
        dpdx = dpdx.at[1:-1, 1:-1].set((p[1:-1, 2:] - p[1:-1, :-2]) / (2*dx))
        dpdy = dpdy.at[1:-1, 1:-1].set((p[2:, 1:-1] - p[:-2, 1:-1]) / (2*dy))
        
        u_new = u_star - dt * dpdx
        v_new = v_star - dt * dpdy
        
        # Only update in CFD region
        u_new = jnp.where(mask == 1, u_new, u_pinn)
        v_new = jnp.where(mask == 1, v_new, v_pinn)
        
        # Apply BCs to corrected velocity
        u_new, v_new = apply_hybrid_boundary_conditions(
            u_new, v_new, mask, cfd_boundary, u_pinn, v_pinn
        )
        
        return u_new, v_new, p
    
    @jit
    def compute_residual_hybrid(u_new, u_old, v_new, v_old, mask):
        """Compute convergence residual only in CFD region"""
        diff_u = jnp.sum(((u_new - u_old) * mask)**2)
        diff_v = jnp.sum(((v_new - v_old) * mask)**2)
        return jnp.sqrt(diff_u + diff_v)
    
    # Apply initial boundary conditions
    u, v = apply_hybrid_boundary_conditions(u, v, mask_jax, cfd_boundary_jax, u_pinn_jax, v_pinn_jax)
    
    # Ensure PINN region keeps PINN values
    p = jnp.where(mask_jax == 0, p_pinn_jax, p)
    
    print("Solving Navier-Stokes equations with hybrid PINN-CFD method...")
    print(f"Reynolds number: {Re}")
    print(f"Grid size: {N}x{N}")
    print(f"CFD region: {np.sum(mask)} cells ({100*np.sum(mask)/(N*N):.1f}%)")
    print(f"PINN region: {N*N - np.sum(mask)} cells ({100*(N*N - np.sum(mask))/(N*N):.1f}%)")
    print(f"Interface cells: {np.sum(interface_mask)}")
    print(f"Time step: {dt:.6e}")
    print("-" * 50)
    
    # Time stepping loop
    for n in range(max_iter):
        u_old, v_old = u, v
        u, v, p = step_hybrid(
            u, v, p, dt, dx, dy, nu, 
            mask_jax, cfd_boundary_jax, u_pinn_jax, v_pinn_jax, p_pinn_jax
        )
        
        # Check convergence every 50 iterations
        if n % 50 == 0:
            residual = compute_residual_hybrid(u, u_old, v, v_old, mask_jax)
            residual_val = float(residual)
            print(f"Iteration {n}, Residual: {residual_val:.6e}")
            
            if residual_val < tol:
                print(f"\nConverged at iteration {n}")
                break
    else:
        print(f"\nReached maximum iterations ({max_iter})")
    
    # Convert to numpy and return
    return np.array(u), np.array(v), np.array(p)


def create_center_pinn_mask(N, border_width):
    """
    Create a mask where PINN is used in the center (mask=0) 
    and CFD is used near the boundaries (mask=1).
    
    This is useful because PINN typically performs better away from boundaries.
    
    Parameters:
    -----------
    N : int
        Grid size
    border_width : int
        Width of the CFD region at each boundary
        
    Returns:
    --------
    mask : ndarray of shape (N, N)
        Binary mask (1 = CFD region, 0 = PINN region)
    """
    mask = np.zeros((N, N), dtype=np.int32)
    mask[:border_width, :] = 1   # Bottom border
    mask[-border_width:, :] = 1  # Top border
    mask[:, :border_width] = 1   # Left border
    mask[:, -border_width:] = 1  # Right border
    return mask


def create_boundary_pinn_mask(N, center_width):
    """
    Create a mask where PINN is used near the boundaries (mask=0)
    and CFD is used in the center (mask=1).
    
    This is the inverse of create_center_pinn_mask.
    
    Parameters:
    -----------
    N : int
        Grid size
    center_width : int
        Half-width of the center CFD region
        
    Returns:
    --------
    mask : ndarray of shape (N, N)
        Binary mask (1 = CFD region, 0 = PINN region)
    """
    mask = np.zeros((N, N), dtype=np.int32)
    center = N // 2
    mask[center-center_width:center+center_width, 
         center-center_width:center+center_width] = 1
    return mask


def create_custom_mask(N, pinn_regions):
    """
    Create a custom mask with specified PINN regions.
    
    Parameters:
    -----------
    N : int
        Grid size
    pinn_regions : list of tuples
        List of (y_start, y_end, x_start, x_end) tuples defining PINN regions
        
    Returns:
    --------
    mask : ndarray of shape (N, N)
        Binary mask (1 = CFD region, 0 = PINN region)
    """
    mask = np.ones((N, N), dtype=np.int32)  # Default to CFD
    for y_start, y_end, x_start, x_end in pinn_regions:
        mask[y_start:y_end, x_start:x_end] = 0  # PINN regions
    return mask


def plot_hybrid_solution(u, v, p, mask):
    """Plot velocity and pressure fields with mask overlay"""
    N = u.shape[0]
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y)
    
    vel_mag = np.sqrt(u**2 + v**2)
    
    def contour_with_mask(ax, x, y, z, title, mask):
        cf = ax.contourf(x, y, z, levels=50, cmap='jet')
        # Overlay mask boundary
        ax.contour(x, y, mask, levels=[0.5], colors='white', linewidths=2, linestyles='--')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(cf, ax=ax)
    
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(2, 2)
    
    ax1 = fig.add_subplot(gs[0, 0])
    contour_with_mask(ax1, X, Y, vel_mag, '|u| (velocity magnitude)', mask)
    
    ax2 = fig.add_subplot(gs[0, 1])
    contour_with_mask(ax2, X, Y, p, 'p (pressure)', mask)
    
    ax3 = fig.add_subplot(gs[1, 0])
    contour_with_mask(ax3, X, Y, u, 'u (x-velocity)', mask)
    
    ax4 = fig.add_subplot(gs[1, 1])
    contour_with_mask(ax4, X, Y, v, 'v (y-velocity)', mask)
    
    # Add legend
    fig.text(0.5, 0.02, 'White dashed line: PINN/CFD interface', 
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    fig.savefig('cavity_flow_hybrid.png', dpi=300)
    plt.show()

def uv(network, xy):
    """
    Compute flow velocities (u, v) for the network with output (psi, p).

    Args:
        xy: network input variables as ndarray.

    Returns:
        (u, v) as ndarray.
    """

    xy = tf.constant(xy)
    with tf.GradientTape() as g:
        g.watch(xy)
        psi_p = network(xy)
    psi_p_j = g.batch_jacobian(psi_p, xy)
    u =  psi_p_j[..., 0, 1]
    v = -psi_p_j[..., 0, 0]
    return u.numpy(), v.numpy()

if __name__ == "__main__":
    u0 = 1.0
    L = 1.0
    nu = 0.01
    rho = 1.0
    
    Re = rho * u0 * L / nu
    N = 100  # Grid points
    network = Network().build()
    network.load_weights('models/pinn_cavity_model.h5')
    mask = create_center_pinn_mask(N, border_width=20)
    start_time = time.time()
    u, v, p = solve_hybrid_pinn_cfd(
        network=network,      # Your PINN model
        uv_func=uv,           # Your uv() function
        mask=mask,
        Re=100,
        N=N
    )
    end_time = time.time()
    
    print(f"\nSimulation time: {end_time - start_time:.4f} seconds")
    plot_solution(u, v, p)
