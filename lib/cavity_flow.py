"""
Lid-driven cavity flow simulation using PINN-CFD hybrid solver.
"""

import jax.numpy as jnp
from jax import jit
import numpy as np
from scipy import ndimage

from lib.base_simulation import BaseSimulation
from lib.complexity_scoring import ComplexityScorer, create_dynamic_mask, compute_mask_statistics


class CavityFlowSimulation(BaseSimulation):
    """
    Lid-driven cavity flow simulation.
    
    The top lid moves with velocity u=1, while all other walls are no-slip.
    This creates a recirculating flow pattern inside the cavity.
    """
    
    def __init__(self, Re=100, N=100, max_iter=200000, tol=1e-6):
        """
        Initialize cavity flow simulation.
        
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
        super().__init__(Re, N, max_iter, tol)
    
    def get_simulation_name(self):
        return "Lid-Driven Cavity Flow"
    
    def get_output_filename(self):
        return "cavity_flow"
    
    def apply_boundary_conditions(self, u, v):
        """
        Apply velocity boundary conditions for lid-driven cavity.
        
        - Top lid (y=1): u=1, v=0 (lid moves right)
        - Bottom wall (y=0): u=0, v=0 (no-slip)
        - Left wall (x=0): u=0, v=0 (no-slip)
        - Right wall (x=1): u=0, v=0 (no-slip)
        """
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


class CavityFlowHybridSimulation(CavityFlowSimulation):
    """
    Hybrid PINN-CFD solver for lid-driven cavity flow.
    
    Uses PINN in specified regions and CFD for the rest, with PINN values
    serving as boundary conditions at the interface.
    """
    
    def __init__(self, network, uv_func, mask, Re=100, N=100, max_iter=200000, tol=1e-6):
        """
        Initialize hybrid cavity flow simulation.
        
        Parameters:
        -----------
        network : 
            PINN model that outputs (psi, p) given (x, y) coordinates.
        uv_func : callable
            Function to compute (u, v) from the network: uv_func(network, xy) -> (u, v)
        mask : ndarray of shape (N, N)
            Binary mask where 1 indicates CFD region, 0 indicates PINN region.
        Re : float
            Reynolds number
        N : int
            Number of grid points per side
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        """
        super().__init__(Re, N, max_iter, tol)
        self.network = network
        self.uv_func = uv_func
        self.mask = mask
        
        # Validate mask dimensions
        assert mask.shape == (N, N), f"Mask shape {mask.shape} must match grid size ({N}, {N})"
        
        # Compute PINN values and interface
        self._setup_pinn_interface()
    
    def _setup_pinn_interface(self):
        """Compute PINN values and find interface between PINN and CFD regions."""
        N = self.N
        
        # Query PINN for the entire domain
        print("Querying PINN for initial field values...")
        psi_p = self.network.predict(self.xy, batch_size=len(self.xy))
        self.psi_pinn = psi_p[..., 0].reshape(self.X.shape)
        self.p_pinn = psi_p[..., 1].reshape(self.X.shape)
        
        # Compute velocities from PINN
        u_pinn, v_pinn = self.uv_func(self.network, self.xy)
        self.u_pinn = u_pinn.reshape(self.X.shape)
        self.v_pinn = v_pinn.reshape(self.X.shape)
        
        # Find interface: boundary between PINN and CFD regions
        pinn_region = (self.mask == 0).astype(float)
        dilated_pinn = ndimage.binary_dilation(pinn_region, iterations=1).astype(float)
        
        # Interface is where CFD region meets dilated PINN region
        self.interface_mask = (self.mask == 1) & (dilated_pinn == 1)
        
        # Domain boundaries
        domain_boundary = np.zeros_like(self.mask, dtype=bool)
        domain_boundary[0, :] = True   # Bottom
        domain_boundary[-1, :] = True  # Top
        domain_boundary[:, 0] = True   # Left
        domain_boundary[:, -1] = True  # Right
        
        # Combined boundary for CFD
        self.cfd_boundary = self.interface_mask | (domain_boundary & (self.mask == 1))
        
        # Convert to JAX arrays
        self.mask_jax = jnp.array(self.mask, dtype=jnp.float64)
        self.interface_jax = jnp.array(self.interface_mask, dtype=bool)
        self.cfd_boundary_jax = jnp.array(self.cfd_boundary, dtype=bool)
        self.u_pinn_jax = jnp.array(self.u_pinn)
        self.v_pinn_jax = jnp.array(self.v_pinn)
        self.p_pinn_jax = jnp.array(self.p_pinn)
        
        print(f"CFD region: {np.sum(self.mask)} cells ({100*np.sum(self.mask)/(N*N):.1f}%)")
        print(f"PINN region: {N*N - np.sum(self.mask)} cells ({100*(N*N - np.sum(self.mask))/(N*N):.1f}%)")
        print(f"Interface cells: {np.sum(self.interface_mask)}")
    
    def apply_hybrid_boundary_conditions(self, u, v):
        """
        Apply boundary conditions for hybrid solver:
        - At domain walls: standard no-slip (except lid)
        - At PINN-CFD interface: use PINN values as Dirichlet BC
        - In PINN region: keep PINN values
        """
        mask = self.mask_jax
        cfd_boundary = self.cfd_boundary_jax
        u_pinn = self.u_pinn_jax
        v_pinn = self.v_pinn_jax
        
        # Standard lid-driven cavity BCs at domain boundaries (only in CFD region)
        # Top lid
        u = jnp.where(
            (jnp.arange(u.shape[0])[:, None] == u.shape[0] - 1) & (mask == 1),
            1.0, u
        )
        v = jnp.where(
            (jnp.arange(v.shape[0])[:, None] == v.shape[0] - 1) & (mask == 1),
            0.0, v
        )
        
        # Bottom wall
        u = jnp.where((jnp.arange(u.shape[0])[:, None] == 0) & (mask == 1), 0.0, u)
        v = jnp.where((jnp.arange(v.shape[0])[:, None] == 0) & (mask == 1), 0.0, v)
        
        # Left wall
        u = jnp.where((jnp.arange(u.shape[1])[None, :] == 0) & (mask == 1), 0.0, u)
        v = jnp.where((jnp.arange(v.shape[1])[None, :] == 0) & (mask == 1), 0.0, v)
        
        # Right wall
        u = jnp.where((jnp.arange(u.shape[1])[None, :] == u.shape[1] - 1) & (mask == 1), 0.0, u)
        v = jnp.where((jnp.arange(v.shape[1])[None, :] == v.shape[1] - 1) & (mask == 1), 0.0, v)
        
        # At interface: use PINN values
        is_boundary = (
            (jnp.arange(u.shape[0])[:, None] == 0) |
            (jnp.arange(u.shape[0])[:, None] == u.shape[0] - 1) |
            (jnp.arange(u.shape[1])[None, :] == 0) |
            (jnp.arange(u.shape[1])[None, :] == u.shape[1] - 1)
        )
        u = jnp.where(cfd_boundary & ~is_boundary, u_pinn, u)
        v = jnp.where(cfd_boundary & ~is_boundary, v_pinn, v)
        
        # In PINN region: always use PINN values
        u = jnp.where(mask == 0, u_pinn, u)
        v = jnp.where(mask == 0, v_pinn, v)
        
        return u, v
    
    def solve(self):
        """
        Solve Navier-Stokes equations using hybrid PINN-CFD method.
        
        Returns:
        --------
        u, v, p : ndarray
            Velocity and pressure fields combining PINN and CFD solutions
        """
        from jax import lax
        
        N = self.N
        dx, dy, dt = self.dx, self.dy, self.dt
        nu = self.nu
        
        # Initialize with PINN values
        u = jnp.array(self.u_pinn)
        v = jnp.array(self.v_pinn)
        p = jnp.array(self.p_pinn)
        
        # Get helper functions
        laplacian, divergence, convection, pressure_gradient = self._create_jit_functions()
        
        mask_jax = self.mask_jax
        u_pinn_jax = self.u_pinn_jax
        v_pinn_jax = self.v_pinn_jax
        p_pinn_jax = self.p_pinn_jax
        cfd_boundary_jax = self.cfd_boundary_jax
        
        apply_hybrid_bc = jit(self.apply_hybrid_boundary_conditions)
        
        @jit
        def pressure_poisson_iteration_hybrid(p, rhs):
            """Single Jacobi iteration for pressure Poisson equation (hybrid version)"""
            p_new = jnp.zeros_like(p)
            p_new = p_new.at[1:-1, 1:-1].set(
                0.25 * (
                    p[1:-1, 2:] + p[1:-1, :-2] +
                    p[2:, 1:-1] + p[:-2, 1:-1] -
                    dx**2 * rhs[1:-1, 1:-1]
                )
            )
            # Neumann BC at domain walls
            p_new = p_new.at[0, :].set(p_new[1, :])
            p_new = p_new.at[-1, :].set(p_new[-2, :])
            p_new = p_new.at[:, 0].set(p_new[:, 1])
            p_new = p_new.at[:, -1].set(p_new[:, -2])
            
            # In PINN region: use PINN pressure
            p_new = jnp.where(mask_jax == 0, p_pinn_jax, p_new)
            return p_new
        
        @jit
        def solve_pressure_poisson_hybrid(p, rhs, n_iter=100):
            """Solve pressure Poisson equation with Jacobi iterations"""
            def body_fn(i, p):
                return pressure_poisson_iteration_hybrid(p, rhs)
            return lax.fori_loop(0, n_iter, body_fn, p)
        
        @jit
        def step_hybrid(u, v, p):
            """Single time step using fractional step method (hybrid version)"""
            # Step 1: Compute intermediate velocity
            conv_u = convection(u, v, u)
            conv_v = convection(u, v, v)
            lap_u = laplacian(u)
            lap_v = laplacian(v)
            
            u_star = u + dt * (-conv_u + nu * lap_u)
            v_star = v + dt * (-conv_v + nu * lap_v)
            
            # Only update in CFD region
            u_star = jnp.where(mask_jax == 1, u_star, u_pinn_jax)
            v_star = jnp.where(mask_jax == 1, v_star, v_pinn_jax)
            
            u_star, v_star = apply_hybrid_bc(u_star, v_star)
            
            # Step 2: Solve pressure Poisson equation
            div_star = divergence(u_star, v_star)
            rhs = div_star / dt
            p_new = solve_pressure_poisson_hybrid(p, rhs)
            
            # Step 3: Correct velocity
            dpdx, dpdy = pressure_gradient(p_new)
            u_new = u_star - dt * dpdx
            v_new = v_star - dt * dpdy
            
            # Only update in CFD region
            u_new = jnp.where(mask_jax == 1, u_new, u_pinn_jax)
            v_new = jnp.where(mask_jax == 1, v_new, v_pinn_jax)
            
            u_new, v_new = apply_hybrid_bc(u_new, v_new)
            
            return u_new, v_new, p_new
        
        @jit
        def compute_residual_hybrid(u_new, u_old, v_new, v_old):
            """Compute convergence residual only in CFD region"""
            diff_u = jnp.sum(((u_new - u_old) * mask_jax)**2)
            diff_v = jnp.sum(((v_new - v_old) * mask_jax)**2)
            return jnp.sqrt(diff_u + diff_v)
        
        # Apply initial boundary conditions
        u, v = apply_hybrid_bc(u, v)
        p = jnp.where(mask_jax == 0, p_pinn_jax, p)
        
        print(f"Solving {self.get_simulation_name()} with hybrid PINN-CFD method...")
        print(f"Reynolds number: {self.Re}")
        print(f"Grid size: {N}x{N}")
        print(f"Time step: {dt:.6e}")
        print("-" * 50)
        
        # Time stepping loop
        for n in range(self.max_iter):
            u_old, v_old = u, v
            u, v, p = step_hybrid(u, v, p)
            
            if n % 50 == 0:
                residual = compute_residual_hybrid(u, u_old, v, v_old)
                residual_val = float(residual)
                print(f"Iteration {n}, Residual: {residual_val:.6e}")
                
                if residual_val < self.tol:
                    print(f"\nConverged at iteration {n}")
                    break
        else:
            print(f"\nReached maximum iterations ({self.max_iter})")
        
        return np.array(u), np.array(v), np.array(p)


# Mask creation utilities for cavity flow
def create_center_pinn_mask(N, border_width):
    """
    Create a mask where PINN is used in the center (mask=0) 
    and CFD is used near the boundaries (mask=1).
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
    """
    mask = np.ones((N, N), dtype=np.int32)  # Default to CFD
    for y_start, y_end, x_start, x_end in pinn_regions:
        mask[y_start:y_end, x_start:x_end] = 0  # PINN regions
    return mask


class CavityFlowDynamicHybridSimulation(CavityFlowSimulation):
    """
    Hybrid PINN-CFD solver with dynamic segregation based on complexity scoring.
    
    Uses complexity score to automatically segregate the domain into CFD-dominated
    and PINN-dominated regions based on local flow diagnostics.
    """
    
    def __init__(self, network, uv_func, u_init, v_init, p_init, Re=100, N=100,
                 max_iter=200000, tol=1e-6, complexity_threshold=1.0,
                 complexity_weights=None, normalization='mean'):
        """
        Initialize dynamic hybrid cavity flow simulation.
        
        Parameters:
        -----------
        network : tf.keras.Model
            PINN model that outputs (psi, p) or (u, v, p) given (x, y) coordinates.
        uv_func : callable
            Function to compute (u, v) from the network: uv_func(network, xy) -> (u, v)
        u_init, v_init, p_init : ndarray of shape (N, N)
            Initial velocity and pressure fields
        Re : float
            Reynolds number
        N : int
            Number of grid points per side
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        complexity_threshold : float
            Complexity score threshold for CFD assignment
        complexity_weights : dict, optional
            Weights for diagnostic components
        normalization : str
            Normalization method for complexity scoring ('mean', 'max', 'percentile')
        """
        super().__init__(Re, N, max_iter, tol)
        self.network = network
        self.uv_func = uv_func
        self.complexity_threshold = complexity_threshold
        self.complexity_weights = complexity_weights
        self.normalization = normalization
        
        # Store initial fields
        self.u_init = np.array(u_init)
        self.v_init = np.array(v_init)
        self.p_init = np.array(p_init)
        
        # Compute complexity-based mask
        self._compute_dynamic_mask()
        
        # Setup PINN interface
        self._setup_pinn_interface()
    
    def _compute_dynamic_mask(self):
        """
        Compute binary mask based on complexity score of initial fields.
        
        High complexity regions (boundary layers, shear layers) → CFD
        Low complexity regions (smooth, diffusion-dominated) → PINN
        """
        print("Computing dynamic segregation mask based on complexity scoring...")
        
        # Compute complexity score
        self.mask, self.complexity_score = create_dynamic_mask(
            self.u_init, self.v_init, self.p_init,
            nu=self.nu, dx=self.dx, dy=self.dy,
            threshold=self.complexity_threshold,
            weights=self.complexity_weights,
            normalization=self.normalization,
            rho=1.0
        )
        
        # Compute and display statistics
        stats = compute_mask_statistics(self.mask, self.complexity_score)
        print(f"  CFD region: {stats['cfd_count']} cells ({stats['cfd_percentage']:.1f}%)")
        print(f"  PINN region: {stats['pinn_count']} cells ({stats['pinn_percentage']:.1f}%)")
        print(f"  Complexity score range: [{stats['complexity_min']:.4f}, {stats['complexity_max']:.4f}]")
        print(f"  Mean complexity: {stats['complexity_mean']:.4f} ± {stats['complexity_std']:.4f}")
        print(f"  CFD avg complexity: {stats['cfd_avg_complexity']:.4f}")
        print(f"  PINN avg complexity: {stats['pinn_avg_complexity']:.4f}")
    
    def _setup_pinn_interface(self):
        """Compute PINN values and find interface between PINN and CFD regions."""
        N = self.N
        
        # Query PINN for the entire domain
        print("Querying PINN for initial field values...")
        psi_p = self.network.predict(self.xy, batch_size=len(self.xy))
        self.psi_pinn = psi_p[..., 0].reshape(self.X.shape)
        self.p_pinn = psi_p[..., 1].reshape(self.X.shape)
        
        # Compute velocities from PINN
        u_pinn, v_pinn = self.uv_func(self.network, self.xy)
        self.u_pinn = u_pinn.reshape(self.X.shape)
        self.v_pinn = v_pinn.reshape(self.X.shape)
        
        # Find interface: boundary between PINN and CFD regions
        pinn_region = (self.mask == 0).astype(float)
        dilated_pinn = ndimage.binary_dilation(pinn_region, iterations=1).astype(float)
        
        # Interface is where CFD region meets dilated PINN region
        self.interface_mask = (self.mask == 1) & (dilated_pinn == 1)
        
        # Domain boundaries
        domain_boundary = np.zeros_like(self.mask, dtype=bool)
        domain_boundary[0, :] = True   # Bottom
        domain_boundary[-1, :] = True  # Top
        domain_boundary[:, 0] = True   # Left
        domain_boundary[:, -1] = True  # Right
        
        # Combined boundary for CFD
        self.cfd_boundary = self.interface_mask | (domain_boundary & (self.mask == 1))
        
        # Convert to JAX arrays
        self.mask_jax = jnp.array(self.mask, dtype=jnp.float64)
        self.interface_jax = jnp.array(self.interface_mask, dtype=bool)
        self.cfd_boundary_jax = jnp.array(self.cfd_boundary, dtype=bool)
        self.u_pinn_jax = jnp.array(self.u_pinn)
        self.v_pinn_jax = jnp.array(self.v_pinn)
        self.p_pinn_jax = jnp.array(self.p_pinn)
        
        print(f"CFD region: {np.sum(self.mask)} cells ({100*np.sum(self.mask)/(N*N):.1f}%)")
        print(f"PINN region: {N*N - np.sum(self.mask)} cells ({100*(N*N - np.sum(self.mask))/(N*N):.1f}%)")
        print(f"Interface cells: {np.sum(self.interface_mask)}")
    
    def apply_hybrid_boundary_conditions(self, u, v):
        """
        Apply boundary conditions for hybrid solver with dynamic segregation.
        
        - At domain walls: standard no-slip (except lid)
        - At PINN-CFD interface: use PINN values as Dirichlet BC
        - In PINN region: keep PINN values
        """
        mask = self.mask_jax
        cfd_boundary = self.cfd_boundary_jax
        u_pinn = self.u_pinn_jax
        v_pinn = self.v_pinn_jax
        
        # Standard lid-driven cavity BCs at domain boundaries (only in CFD region)
        # Top lid
        u = jnp.where(
            (jnp.arange(u.shape[0])[:, None] == u.shape[0] - 1) & (mask == 1),
            1.0, u
        )
        v = jnp.where(
            (jnp.arange(v.shape[0])[:, None] == v.shape[0] - 1) & (mask == 1),
            0.0, v
        )
        
        # Bottom wall
        u = jnp.where((jnp.arange(u.shape[0])[:, None] == 0) & (mask == 1), 0.0, u)
        v = jnp.where((jnp.arange(v.shape[0])[:, None] == 0) & (mask == 1), 0.0, v)
        
        # Left wall
        u = jnp.where((jnp.arange(u.shape[1])[None, :] == 0) & (mask == 1), 0.0, u)
        v = jnp.where((jnp.arange(v.shape[1])[None, :] == 0) & (mask == 1), 0.0, v)
        
        # Right wall
        u = jnp.where((jnp.arange(u.shape[1])[None, :] == u.shape[1] - 1) & (mask == 1), 0.0, u)
        v = jnp.where((jnp.arange(v.shape[1])[None, :] == u.shape[1] - 1) & (mask == 1), 0.0, v)
        
        # At interface: use PINN values
        is_boundary = (
            (jnp.arange(u.shape[0])[:, None] == 0) |
            (jnp.arange(u.shape[0])[:, None] == u.shape[0] - 1) |
            (jnp.arange(u.shape[1])[None, :] == 0) |
            (jnp.arange(u.shape[1])[None, :] == u.shape[1] - 1)
        )
        u = jnp.where(cfd_boundary & ~is_boundary, u_pinn, u)
        v = jnp.where(cfd_boundary & ~is_boundary, v_pinn, v)
        
        # In PINN region: always use PINN values
        u = jnp.where(mask == 0, u_pinn, u)
        v = jnp.where(mask == 0, v_pinn, v)
        
        return u, v
    
    def solve(self):
        """
        Solve Navier-Stokes equations using dynamic hybrid PINN-CFD method.
        
        Returns:
        --------
        u, v, p : ndarray
            Velocity and pressure fields combining PINN and CFD solutions
        """
        from jax import lax
        
        N = self.N
        dx, dy, dt = self.dx, self.dy, self.dt
        nu = self.nu
        
        # Initialize with PINN values
        u = jnp.array(self.u_pinn)
        v = jnp.array(self.v_pinn)
        p = jnp.array(self.p_pinn)
        
        # Get helper functions
        laplacian, divergence, convection, pressure_gradient = self._create_jit_functions()
        
        mask_jax = self.mask_jax
        u_pinn_jax = self.u_pinn_jax
        v_pinn_jax = self.v_pinn_jax
        p_pinn_jax = self.p_pinn_jax
        cfd_boundary_jax = self.cfd_boundary_jax
        
        apply_hybrid_bc = jit(self.apply_hybrid_boundary_conditions)
        
        @jit
        def pressure_poisson_iteration_hybrid(p, rhs):
            """Single Jacobi iteration for pressure Poisson equation (hybrid version)"""
            p_new = jnp.zeros_like(p)
            p_new = p_new.at[1:-1, 1:-1].set(
                0.25 * (
                    p[1:-1, 2:] + p[1:-1, :-2] +
                    p[2:, 1:-1] + p[:-2, 1:-1] -
                    dx**2 * rhs[1:-1, 1:-1]
                )
            )
            # Neumann BC at domain walls
            p_new = p_new.at[0, :].set(p_new[1, :])
            p_new = p_new.at[-1, :].set(p_new[-2, :])
            p_new = p_new.at[:, 0].set(p_new[:, 1])
            p_new = p_new.at[:, -1].set(p_new[:, -2])
            return p_new
        
        @jit
        def step_hybrid(u, v, p):
            """Single time step with hybrid segregation"""
            # Compute convection and viscous terms
            conv_u = convection(u, v)
            conv_v = convection(v, v)
            lap_u = laplacian(u)
            lap_v = laplacian(v)
            
            # Predict velocities
            u_star = u - dt * conv_u - dt * convection(u, u) + dt * nu * lap_u
            v_star = v - dt * conv_v - dt * convection(v, v) + dt * nu * lap_v
            
            # Pressure Poisson equation right-hand side
            rhs = divergence(u_star, v_star) / dt
            
            # Solve pressure Poisson iteratively (use 20 iterations per time step)
            p_new = p
            for _ in range(20):
                p_new = pressure_poisson_iteration_hybrid(p_new, rhs)
            
            # Pressure correction and boundary conditions
            dpdy, dpdx = pressure_gradient(p_new)
            u_new = u_star - dt * dpdx
            v_new = v_star - dt * dpdy
            
            # Apply hybrid BCs
            u_new, v_new = apply_hybrid_bc(u_new, v_new)
            
            # Ensure PINN region maintains PINN solution
            u_new = jnp.where(mask_jax == 0, u_pinn_jax, u_new)
            v_new = jnp.where(mask_jax == 0, v_pinn_jax, v_new)
            p_new = jnp.where(mask_jax == 0, p_pinn_jax, p_new)
            
            return u_new, v_new, p_new
        
        @jit
        def compute_residual_hybrid(u_new, u_old, v_new, v_old):
            """Compute convergence residual only in CFD region"""
            diff_u = jnp.sum(((u_new - u_old) * mask_jax)**2)
            diff_v = jnp.sum(((v_new - v_old) * mask_jax)**2)
            return jnp.sqrt(diff_u + diff_v)
        
        # Apply initial boundary conditions
        u, v = apply_hybrid_bc(u, v)
        p = jnp.where(mask_jax == 0, p_pinn_jax, p)
        
        print(f"Solving {self.get_simulation_name()} with dynamic hybrid PINN-CFD method...")
        print(f"Reynolds number: {self.Re}")
        print(f"Grid size: {N}x{N}")
        print(f"Time step: {dt:.6e}")
        print(f"Complexity threshold: {self.complexity_threshold}")
        print("-" * 50)
        
        # Time stepping loop
        for n in range(self.max_iter):
            u_old, v_old = u, v
            u, v, p = step_hybrid(u, v, p)
            
            if n % 50 == 0:
                residual = compute_residual_hybrid(u, u_old, v, v_old)
                residual_val = float(residual)
                print(f"Iteration {n}, Residual: {residual_val:.6e}")
                
                if residual_val < self.tol:
                    print(f"\nConverged at iteration {n}")
                    break
        else:
            print(f"\nReached maximum iterations ({self.max_iter})")
        
        return np.array(u), np.array(v), np.array(p)
