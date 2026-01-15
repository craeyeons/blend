"""
Flow around cylinder simulation using PINN-CFD hybrid solver.
"""

import jax
import jax.numpy as jnp
from jax import jit, lax
import numpy as np
import tensorflow as tf
from scipy import ndimage
# Enable 64-bit precision for better numerical accuracy
jax.config.update("jax_enable_x64", True)
from lib.base_simulation import BaseSimulation


class CylinderFlowSimulation(BaseSimulation):
    """
    Flow around a cylinder simulation.
    
    Uniform inlet flow from the left, cylinder obstacle in the domain,
    with no-slip on cylinder surface and channel walls.
    
    Uses Nx × Ny grid where Nx/Ny = Lx/Ly to maintain uniform grid spacing.
    """
    
    def __init__(self, Re=100, N=100, max_iter=200000, tol=1e-6,
                 x_domain=(0, 2), y_domain=(0, 1),
                 cylinder_center=(0.5, 0.5), cylinder_radius=0.1,
                 inlet_velocity=1.0):
        """
        Initialize cylinder flow simulation.
        
        Parameters:
        -----------
        Re : float
            Reynolds number
        N : int
            Base number of grid points (Ny = N, Nx = N * aspect_ratio)
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        x_domain : tuple
            (x_min, x_max) domain bounds
        y_domain : tuple
            (y_min, y_max) domain bounds
        cylinder_center : tuple
            (Cx, Cy) cylinder center coordinates
        cylinder_radius : float
            Cylinder radius
        inlet_velocity : float
            Inlet velocity magnitude
        """
        # Store domain parameters before calling parent init
        self.x_ini, self.x_f = x_domain
        self.y_ini, self.y_f = y_domain
        self.Cx, self.Cy = cylinder_center
        self.radius = cylinder_radius
        self.u_inlet = inlet_velocity
        
        # Override parent grid setup
        self.Re = Re
        self.N = N  # Base resolution
        self.max_iter = max_iter
        self.tol = tol
        self.nu = 1.0 / Re
        
        # Grid setup with aspect ratio - ensure uniform spacing
        self.Lx = self.x_f - self.x_ini
        self.Ly = self.y_f - self.y_ini
        aspect_ratio = self.Lx / self.Ly
        
        # Ny = N, Nx = N * aspect_ratio (rounded to int)
        self.Ny = N
        self.Nx = int(N * aspect_ratio)
        
        # Uniform grid spacing
        self.dx = self.Lx / (self.Nx - 1)
        self.dy = self.Ly / (self.Ny - 1)
        self.dt = 0.005 * min(self.dx, self.dy)
        
        # Create coordinate grid (Ny rows, Nx columns)
        x = np.linspace(self.x_ini, self.x_f, self.Nx)
        y = np.linspace(self.y_ini, self.y_f, self.Ny)
        self.X, self.Y = np.meshgrid(x, y)  # Shape: (Ny, Nx)
        self.xy = np.stack([self.X.flatten(), self.Y.flatten()], axis=-1)
        
        # Create cylinder mask (1 = inside cylinder, 0 = fluid)
        self.cylinder_mask = self._create_cylinder_mask()
        self.cylinder_mask_jax = jnp.array(self.cylinder_mask)
        
        print(f"JAX devices: {jax.devices()}")
        print(f"Using device: {jax.devices()[0]}")
        print(f"Grid: {self.Nx} x {self.Ny} (dx={self.dx:.4f}, dy={self.dy:.4f})")
    
    def _create_cylinder_mask(self):
        """Create mask identifying points inside the cylinder."""
        dist_sq = (self.X - self.Cx)**2 + (self.Y - self.Cy)**2
        return (dist_sq <= self.radius**2).astype(np.int32)
    
    def get_simulation_name(self):
        return "Flow Around Cylinder"
    
    def get_output_filename(self):
        return "cylinder_flow"
    
    def inlet_profile(self, y):
        """
        Parabolic inlet velocity profile.
        u = 4 * u_max * y * (H - y) / H^2 for channel flow
        """
        H = self.y_f - self.y_ini
        return 4 * self.u_inlet * (y - self.y_ini) * (self.y_f - y) / (H**2)
    
    def apply_boundary_conditions(self, u, v):
        """
        Apply velocity boundary conditions for cylinder flow.
        
        - Inlet (x=x_ini): parabolic profile u=u(y), v=0
        - Outlet (x=x_f): zero gradient (Neumann)
        - Top/Bottom walls (y=y_ini, y=y_f): no-slip u=0, v=0
        - Cylinder surface: no-slip u=0, v=0
        """
        Ny, Nx = self.Ny, self.Nx
        
        # Inlet (left boundary): parabolic profile
        y_inlet = jnp.linspace(self.y_ini, self.y_f, Ny)
        u_inlet_profile = 4 * self.u_inlet * (y_inlet - self.y_ini) * (self.y_f - y_inlet) / (self.Ly**2)
        u = u.at[:, 0].set(u_inlet_profile)
        v = v.at[:, 0].set(0.0)
        
        # Outlet (right boundary): zero gradient
        u = u.at[:, -1].set(u[:, -2])
        v = v.at[:, -1].set(v[:, -2])
        
        # Top wall (y=y_f): no-slip
        u = u.at[-1, :].set(0.0)
        v = v.at[-1, :].set(0.0)
        
        # Bottom wall (y=y_ini): no-slip
        u = u.at[0, :].set(0.0)
        v = v.at[0, :].set(0.0)
        
        # Cylinder surface: no-slip
        u = jnp.where(self.cylinder_mask_jax == 1, 0.0, u)
        v = jnp.where(self.cylinder_mask_jax == 1, 0.0, v)
        
        return u, v
    
    def solve(self):
        """
        Solve Navier-Stokes equations for cylinder flow.
        
        Returns:
        --------
        u, v, p : ndarray
            Velocity and pressure fields (shape: Ny x Nx)
        """
        Nx, Ny = self.Nx, self.Ny
        dx, dy, dt = self.dx, self.dy, self.dt
        nu = self.nu
        
        # Initialize fields (shape: Ny x Nx)
        u = jnp.zeros((Ny, Nx))
        v = jnp.zeros((Ny, Nx))
        p = jnp.zeros((Ny, Nx))
        
        # Set initial velocity field (parabolic profile)
        y_grid = jnp.linspace(self.y_ini, self.y_f, Ny)
        u_init = 4 * self.u_inlet * (y_grid[:, None] - self.y_ini) * (self.y_f - y_grid[:, None]) / (self.Ly**2)
        u = jnp.where(self.cylinder_mask_jax == 0, u_init, 0.0)
        
        # Apply initial boundary conditions
        apply_bc = jit(self.apply_boundary_conditions)
        u, v = apply_bc(u, v)
        
        cylinder_mask = self.cylinder_mask_jax
        
        @jit
        def laplacian(f):
            """Compute Laplacian using central differences"""
            lap = jnp.zeros_like(f)
            lap = lap.at[1:-1, 1:-1].set(
                (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, :-2]) / dx**2 +
                (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1]) / dy**2
            )
            return lap
        
        @jit
        def divergence(u, v):
            """Compute divergence of velocity field"""
            div = jnp.zeros_like(u)
            div = div.at[1:-1, 1:-1].set(
                (u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx) +
                (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dy)
            )
            return div
        
        @jit
        def convection(u, v, f):
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
        def pressure_poisson_iteration(p, rhs):
            """Single Jacobi iteration for pressure Poisson equation"""
            p_new = jnp.zeros_like(p)
            # For uniform grid (dx ≈ dy), use standard formula
            p_new = p_new.at[1:-1, 1:-1].set(
                0.25 * (
                    p[1:-1, 2:] + p[1:-1, :-2] +
                    p[2:, 1:-1] + p[:-2, 1:-1] -
                    dx**2 * rhs[1:-1, 1:-1]
                )
            )
            # Boundary conditions for pressure
            p_new = p_new.at[0, :].set(p_new[1, :])      # Bottom
            p_new = p_new.at[-1, :].set(p_new[-2, :])    # Top
            p_new = p_new.at[:, 0].set(p_new[:, 1])      # Inlet
            p_new = p_new.at[:, -1].set(0.0)             # Outlet (reference pressure)
            
            # Inside cylinder: set to zero
            p_new = jnp.where(cylinder_mask == 1, 0.0, p_new)
            
            return p_new
        
        @jit
        def solve_pressure_poisson(p, rhs, n_iter=100):
            """Solve pressure Poisson equation with Jacobi iterations"""
            def body_fn(i, p):
                return pressure_poisson_iteration(p, rhs)
            return lax.fori_loop(0, n_iter, body_fn, p)
        
        @jit
        def step(u, v, p):
            """Single time step using fractional step method"""
            # Step 1: Compute intermediate velocity
            conv_u = convection(u, v, u)
            conv_v = convection(u, v, v)
            lap_u = laplacian(u)
            lap_v = laplacian(v)
            
            u_star = u + dt * (-conv_u + nu * lap_u)
            v_star = v + dt * (-conv_v + nu * lap_v)
            
            # Zero velocity inside cylinder
            u_star = jnp.where(cylinder_mask == 1, 0.0, u_star)
            v_star = jnp.where(cylinder_mask == 1, 0.0, v_star)
            
            u_star, v_star = apply_bc(u_star, v_star)
            
            # Step 2: Solve pressure Poisson equation
            div_star = divergence(u_star, v_star)
            rhs = div_star / dt
            p_new = solve_pressure_poisson(p, rhs)
            
            # Step 3: Correct velocity
            dpdx = jnp.zeros_like(p_new)
            dpdy = jnp.zeros_like(p_new)
            dpdx = dpdx.at[1:-1, 1:-1].set((p_new[1:-1, 2:] - p_new[1:-1, :-2]) / (2*dx))
            dpdy = dpdy.at[1:-1, 1:-1].set((p_new[2:, 1:-1] - p_new[:-2, 1:-1]) / (2*dy))
            
            u_new = u_star - dt * dpdx
            v_new = v_star - dt * dpdy
            
            # Zero velocity inside cylinder
            u_new = jnp.where(cylinder_mask == 1, 0.0, u_new)
            v_new = jnp.where(cylinder_mask == 1, 0.0, v_new)
            
            u_new, v_new = apply_bc(u_new, v_new)
            
            return u_new, v_new, p_new
        
        @jit
        def compute_residual(u_new, u_old, v_new, v_old):
            """Compute convergence residual (excluding cylinder interior)"""
            fluid_mask = 1 - cylinder_mask
            diff_u = jnp.sum(((u_new - u_old) * fluid_mask)**2)
            diff_v = jnp.sum(((v_new - v_old) * fluid_mask)**2)
            return jnp.sqrt(diff_u + diff_v)
        
        print(f"Solving {self.get_simulation_name()} with JAX (Fractional Step method)...")
        print(f"Reynolds number: {self.Re}")
        print(f"Grid size: {Nx}x{Ny}")
        print(f"Domain: x=[{self.x_ini}, {self.x_f}], y=[{self.y_ini}, {self.y_f}]")
        print(f"Cylinder: center=({self.Cx}, {self.Cy}), radius={self.radius}")
        print(f"Time step: {dt:.6e}")
        print("-" * 50)
        
        # Time stepping loop
        for n in range(self.max_iter):
            u_old, v_old = u, v
            u, v, p = step(u, v, p)
            
            if n % 50 == 0:
                residual = compute_residual(u, u_old, v, v_old)
                residual_val = float(residual)
                print(f"Iteration {n}, Residual: {residual_val:.6e}")
                
                if residual_val < self.tol:
                    print(f"\nConverged at iteration {n}")
                    break
        else:
            print(f"\nReached maximum iterations ({self.max_iter})")
        
        return np.array(u), np.array(v), np.array(p)


class CylinderFlowHybridSimulation(CylinderFlowSimulation):
    """
    Hybrid PINN-CFD solver for flow around cylinder.
    
    Uses PINN in specified regions and CFD for the rest, with PINN values
    serving as boundary conditions at the interface. Supports multiple
    CFD regions via a user-provided mask.
    
    Uses Nx × Ny grid where Nx/Ny = Lx/Ly to maintain uniform grid spacing.
    """
    
    def __init__(self, network, uv_func, mask, Re=100, N=100, max_iter=200000, tol=1e-6,
                 x_domain=(0, 2), y_domain=(0, 1),
                 cylinder_center=(0.5, 0.5), cylinder_radius=0.1,
                 inlet_velocity=1.0):
        """
        Initialize hybrid cylinder flow simulation.
        
        Parameters:
        -----------
        network : tf.keras.Model
            PINN model that outputs (u, v, p) or (psi, p) given (x, y) coordinates.
        uv_func : callable
            Function to compute (u, v) from the network: uv_func(network, xy) -> (u, v)
        mask : ndarray of shape (Ny, Nx)
            Binary mask where 1 indicates CFD region, 0 indicates PINN region.
            Can have multiple disconnected CFD regions.
        Re : float
            Reynolds number
        N : int
            Base number of grid points (Ny = N, Nx = N * aspect_ratio)
        max_iter : int
            Maximum iterations
        tol : float
            Convergence tolerance
        x_domain : tuple
            (x_min, x_max) domain bounds
        y_domain : tuple
            (y_min, y_max) domain bounds
        cylinder_center : tuple
            (Cx, Cy) cylinder center coordinates
        cylinder_radius : float
            Cylinder radius
        inlet_velocity : float
            Inlet velocity magnitude
        """
        # Initialize parent class (sets up grid, cylinder mask, etc.)
        super().__init__(Re, N, max_iter, tol, x_domain, y_domain,
                        cylinder_center, cylinder_radius, inlet_velocity)
        
        self.network = network
        self.uv_func = uv_func
        self.cfd_mask = mask  # User-provided CFD/PINN mask
        
        # Validate mask dimensions (Ny x Nx)
        expected_shape = (self.Ny, self.Nx)
        assert mask.shape == expected_shape, f"Mask shape {mask.shape} must match grid size {expected_shape}"
        
        # Compute PINN values and interface
        self._setup_pinn_interface()
    
    def _setup_pinn_interface(self):
        """Compute PINN values and find interface between PINN and CFD regions."""
        Ny, Nx = self.Ny, self.Nx
        
        # Query PINN for the entire domain
        print("Querying PINN for initial field values...")
        psi_p = self.network.predict(self.xy, batch_size=len(self.xy))
        
        # Handle different network output formats
        if psi_p.shape[-1] == 2:
            # Network outputs (psi, p) - compute u, v from stream function
            self.psi_pinn = psi_p[..., 0].reshape(self.X.shape)
            self.p_pinn = psi_p[..., 1].reshape(self.X.shape)
        else:
            # Network outputs (u, v, p) directly
            self.p_pinn = psi_p[..., -1].reshape(self.X.shape)
        
        # Compute velocities from PINN
        u_pinn, v_pinn = self.uv_func(self.network, self.xy)
        self.u_pinn = u_pinn.reshape(self.X.shape)
        self.v_pinn = v_pinn.reshape(self.X.shape)
        
        # Ensure cylinder region has zero velocity in PINN output
        self.u_pinn = np.where(self.cylinder_mask == 1, 0.0, self.u_pinn)
        self.v_pinn = np.where(self.cylinder_mask == 1, 0.0, self.v_pinn)
        
        # Find interface: boundary between PINN and CFD regions
        # Interface points are CFD cells that have at least one PINN neighbor
        pinn_region = (self.cfd_mask == 0).astype(float)
        dilated_pinn = ndimage.binary_dilation(pinn_region, iterations=1).astype(float)
        
        # Interface is where CFD region meets dilated PINN region
        self.interface_mask = (self.cfd_mask == 1) & (dilated_pinn == 1)
        
        # Domain boundaries
        domain_boundary = np.zeros_like(self.cfd_mask, dtype=bool)
        domain_boundary[0, :] = True   # Bottom
        domain_boundary[-1, :] = True  # Top
        domain_boundary[:, 0] = True   # Left (inlet)
        domain_boundary[:, -1] = True  # Right (outlet)
        
        # Combined boundary for CFD: interface + domain boundaries in CFD region
        self.cfd_boundary = self.interface_mask | (domain_boundary & (self.cfd_mask == 1))
        
        # Convert to JAX arrays
        self.cfd_mask_jax = jnp.array(self.cfd_mask, dtype=jnp.float64)
        self.interface_jax = jnp.array(self.interface_mask, dtype=bool)
        self.cfd_boundary_jax = jnp.array(self.cfd_boundary, dtype=bool)
        self.u_pinn_jax = jnp.array(self.u_pinn)
        self.v_pinn_jax = jnp.array(self.v_pinn)
        self.p_pinn_jax = jnp.array(self.p_pinn)
        
        # Count regions
        from scipy import ndimage as ndi
        labeled, num_regions = ndi.label(self.cfd_mask)
        total_cells = Ny * Nx
        cfd_cells = np.sum(self.cfd_mask)
        
        print(f"CFD region: {cfd_cells} cells ({100*cfd_cells/total_cells:.1f}%)")
        print(f"PINN region: {total_cells - cfd_cells} cells ({100*(total_cells - cfd_cells)/total_cells:.1f}%)")
        print(f"Number of CFD regions: {num_regions}")
        print(f"Interface cells: {np.sum(self.interface_mask)}")
    
    def apply_hybrid_boundary_conditions(self, u, v):
        """
        Apply boundary conditions for hybrid solver (correct order):
        1. Domain boundary conditions ALWAYS enforced (physical constraints)
        2. PINN values in interior PINN region (not on domain boundaries)
        3. Interface cells get PINN values as coupling BC
        4. Cylinder surface: always no-slip
        """
        Ny, Nx = self.Ny, self.Nx
        cfd_mask = self.cfd_mask_jax
        interface_mask = self.interface_jax
        u_pinn = self.u_pinn_jax
        v_pinn = self.v_pinn_jax
        cylinder_mask = self.cylinder_mask_jax
        
        # Step 1: Apply domain boundary conditions (always, regardless of PINN/CFD region)
        # Inlet (left boundary, x=0): parabolic profile
        y_inlet = jnp.linspace(self.y_ini, self.y_f, Ny)
        u_inlet_profile = 4 * self.u_inlet * (y_inlet - self.y_ini) * (self.y_f - y_inlet) / (self.Ly**2)
        u = u.at[:, 0].set(u_inlet_profile)
        v = v.at[:, 0].set(0.0)
        
        # Outlet (right boundary): zero gradient
        u = u.at[:, -1].set(u[:, -2])
        v = v.at[:, -1].set(v[:, -2])
        
        # Top wall (y=y_f): no-slip
        u = u.at[-1, :].set(0.0)
        v = v.at[-1, :].set(0.0)
        
        # Bottom wall (y=y_ini): no-slip
        u = u.at[0, :].set(0.0)
        v = v.at[0, :].set(0.0)
        
        # Step 2: Set PINN values in interior PINN region (away from domain boundaries)
        # Create a mask for interior points: PINN region but not on domain boundaries
        interior_pinn = (cfd_mask == 0) & ~(
            (jnp.arange(Ny)[:, None] == 0) |      # bottom wall
            (jnp.arange(Ny)[:, None] == Ny - 1) | # top wall
            (jnp.arange(Nx)[None, :] == 0) |      # inlet (left)
            (jnp.arange(Nx)[None, :] == Nx - 1)   # outlet (right)
        )
        u = jnp.where(interior_pinn, u_pinn, u)
        v = jnp.where(interior_pinn, v_pinn, v)
        
        # Step 3: At interface (CFD cells adjacent to PINN region): use PINN values as BC
        u = jnp.where(interface_mask, u_pinn, u)
        v = jnp.where(interface_mask, v_pinn, v)
        
        # Step 4: Cylinder surface: always no-slip
        u = jnp.where(cylinder_mask == 1, 0.0, u)
        v = jnp.where(cylinder_mask == 1, 0.0, v)
        
        return u, v
    
    def solve(self):
        """
        Solve Navier-Stokes equations using hybrid PINN-CFD method.
        
        Returns:
        --------
        u, v, p : ndarray
            Velocity and pressure fields combining PINN and CFD solutions (shape: Ny x Nx)
        """
        Nx, Ny = self.Nx, self.Ny
        dx, dy, dt = self.dx, self.dy, self.dt
        nu = self.nu
        
        # Initialize with PINN values
        u = jnp.array(self.u_pinn)
        v = jnp.array(self.v_pinn)
        p = jnp.array(self.p_pinn)
        
        cfd_mask_jax = self.cfd_mask_jax
        u_pinn_jax = self.u_pinn_jax
        v_pinn_jax = self.v_pinn_jax
        p_pinn_jax = self.p_pinn_jax
        cylinder_mask = self.cylinder_mask_jax
        
        apply_hybrid_bc = jit(self.apply_hybrid_boundary_conditions)
        
        @jit
        def laplacian(f):
            """Compute Laplacian using central differences"""
            lap = jnp.zeros_like(f)
            lap = lap.at[1:-1, 1:-1].set(
                (f[1:-1, 2:] - 2*f[1:-1, 1:-1] + f[1:-1, :-2]) / dx**2 +
                (f[2:, 1:-1] - 2*f[1:-1, 1:-1] + f[:-2, 1:-1]) / dy**2
            )
            return lap
        
        @jit
        def divergence(u, v):
            """Compute divergence of velocity field"""
            div = jnp.zeros_like(u)
            div = div.at[1:-1, 1:-1].set(
                (u[1:-1, 2:] - u[1:-1, :-2]) / (2*dx) +
                (v[2:, 1:-1] - v[:-2, 1:-1]) / (2*dy)
            )
            return div
        
        @jit
        def convection(u, v, f):
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
        def pressure_poisson_iteration_hybrid(p, rhs):
            """Single Jacobi iteration for pressure Poisson equation (hybrid version)"""
            p_new = jnp.zeros_like(p)
            # For uniform grid (dx ≈ dy), use standard formula
            p_new = p_new.at[1:-1, 1:-1].set(
                0.25 * (
                    p[1:-1, 2:] + p[1:-1, :-2] +
                    p[2:, 1:-1] + p[:-2, 1:-1] -
                    dx**2 * rhs[1:-1, 1:-1]
                )
            )
            # Boundary conditions for pressure
            p_new = p_new.at[0, :].set(p_new[1, :])      # Bottom
            p_new = p_new.at[-1, :].set(p_new[-2, :])    # Top
            p_new = p_new.at[:, 0].set(p_new[:, 1])      # Inlet
            p_new = p_new.at[:, -1].set(0.0)             # Outlet (reference pressure)
            
            # Inside cylinder: zero pressure
            p_new = jnp.where(cylinder_mask == 1, 0.0, p_new)
            
            # In PINN region: use PINN pressure
            p_new = jnp.where(cfd_mask_jax == 0, p_pinn_jax, p_new)
            
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
            
            # Apply boundary conditions (this sets PINN values, interface, and wall BCs)
            u_star, v_star = apply_hybrid_bc(u_star, v_star)
            
            # Step 2: Solve pressure Poisson equation
            div_star = divergence(u_star, v_star)
            rhs = div_star / dt
            p_new = solve_pressure_poisson_hybrid(p, rhs)
            
            # Step 3: Correct velocity (only in interior CFD cells, not at interface)
            dpdx = jnp.zeros_like(p_new)
            dpdy = jnp.zeros_like(p_new)
            dpdx = dpdx.at[1:-1, 1:-1].set((p_new[1:-1, 2:] - p_new[1:-1, :-2]) / (2*dx))
            dpdy = dpdy.at[1:-1, 1:-1].set((p_new[2:, 1:-1] - p_new[:-2, 1:-1]) / (2*dy))
            
            u_new = u_star - dt * dpdx
            v_new = v_star - dt * dpdy
            
            # Apply boundary conditions again after pressure correction
            u_new, v_new = apply_hybrid_bc(u_new, v_new)
            
            return u_new, v_new, p_new
        
        @jit
        def compute_residual_hybrid(u_new, u_old, v_new, v_old):
            """Compute convergence residual only in CFD region"""
            fluid_mask = (1 - cylinder_mask) * cfd_mask_jax
            diff_u = jnp.sum(((u_new - u_old) * fluid_mask)**2)
            diff_v = jnp.sum(((v_new - v_old) * fluid_mask)**2)
            return jnp.sqrt(diff_u + diff_v)
        
        # Apply initial boundary conditions
        u, v = apply_hybrid_bc(u, v)
        p = jnp.where(cfd_mask_jax == 0, p_pinn_jax, p)
        
        print(f"Solving {self.get_simulation_name()} with hybrid PINN-CFD method...")
        print(f"Reynolds number: {self.Re}")
        print(f"Grid size: {Nx}x{Ny}")
        print(f"Domain: x=[{self.x_ini}, {self.x_f}], y=[{self.y_ini}, {self.y_f}]")
        print(f"Cylinder: center=({self.Cx}, {self.Cy}), radius={self.radius}")
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


# Mask creation utilities for cylinder flow
def create_cylinder_boundary_mask(N, x_domain, y_domain, cylinder_center, cylinder_radius, 
                                   border_width=10, cfd_radius_factor=3.0, include_wake=False, 
                                   wake_length=1.0):
    """
    Create a mask where CFD is used near the cylinder, at domain boundaries,
    and PINN is used in the middle region (far from both cylinder and edges).
    
    Parameters:
    -----------
    N : int
        Base grid size (Ny = N, Nx = N * aspect_ratio)
    x_domain, y_domain : tuple
        Domain bounds (x_min, x_max), (y_min, y_max)
    cylinder_center : tuple
        (Cx, Cy) cylinder center
    cylinder_radius : float
        Cylinder radius
    border_width : int
        Width of the CFD region at each boundary (in grid points)
    cfd_radius_factor : float
        CFD region extends to cfd_radius_factor * cylinder_radius from cylinder center
    include_wake : bool
        Whether to include the wake region in CFD
    wake_length : float
        Length of the wake region behind the cylinder (if include_wake=True)
        
    Returns:
    --------
    mask : ndarray of shape (Ny, Nx)
        Binary mask (1 = CFD region, 0 = PINN region)
    """
    x_ini, x_f = x_domain
    y_ini, y_f = y_domain
    Lx = x_f - x_ini
    Ly = y_f - y_ini
    aspect_ratio = Lx / Ly
    
    Ny = N
    Nx = int(N * aspect_ratio)
    Cx, Cy = cylinder_center
    
    # Create coordinate grid
    x = np.linspace(x_ini, x_f, Nx)
    y = np.linspace(y_ini, y_f, Ny)
    X, Y = np.meshgrid(x, y)
    
    mask = np.zeros((Ny, Nx), dtype=np.int32)
    
    # 1. CFD near the cylinder (within cfd_radius_factor * cylinder_radius)
    cfd_radius = cfd_radius_factor * cylinder_radius
    dist_from_cylinder = np.sqrt((X - Cx)**2 + (Y - Cy)**2)
    mask[dist_from_cylinder <= cfd_radius] = 1
    
    # 2. CFD at domain boundaries
    mask[:border_width, :] = 1   # Bottom border
    mask[-border_width:, :] = 1  # Top border
    mask[:, :border_width] = 1   # Left border (inlet)
    mask[:, -border_width:] = 1  # Right border (outlet)
    
    # 3. Optionally include wake region
    if include_wake:
        wake_start = Cx + cylinder_radius
        wake_end = min(Cx + cylinder_radius + wake_length, x_f)
        wake_width = 2 * cylinder_radius
        
        wake_region = (
            (X >= wake_start) & (X <= wake_end) &
            (Y >= Cy - wake_width) & (Y <= Cy + wake_width)
        )
        mask[wake_region] = 1
    
    return mask


def create_cylinder_wake_mask(N, x_domain, y_domain, cylinder_center, cylinder_radius,
                               wake_length=1.0, wake_width_factor=2.0):
    """
    Create a mask where CFD is used ONLY in the wake region behind the cylinder.
    PINN is used everywhere else.
    
    Parameters:
    -----------
    N : int
        Base grid size (Ny = N, Nx = N * aspect_ratio)
    x_domain, y_domain : tuple
        Domain bounds
    cylinder_center : tuple
        (Cx, Cy) cylinder center
    cylinder_radius : float
        Cylinder radius
    wake_length : float
        Length of the wake region behind the cylinder
    wake_width_factor : float
        Width of wake = wake_width_factor * cylinder_radius on each side
        
    Returns:
    --------
    mask : ndarray of shape (Ny, Nx)
        Binary mask (1 = CFD region, 0 = PINN region)
    """
    x_ini, x_f = x_domain
    y_ini, y_f = y_domain
    Lx = x_f - x_ini
    Ly = y_f - y_ini
    aspect_ratio = Lx / Ly
    
    Ny = N
    Nx = int(N * aspect_ratio)
    Cx, Cy = cylinder_center
    
    x = np.linspace(x_ini, x_f, Nx)
    y = np.linspace(y_ini, y_f, Ny)
    X, Y = np.meshgrid(x, y)
    
    # Wake region: rectangular region behind cylinder
    wake_start = Cx + cylinder_radius
    wake_end = min(Cx + cylinder_radius + wake_length, x_f)
    wake_width = wake_width_factor * cylinder_radius
    
    mask = np.zeros((Ny, Nx), dtype=np.int32)
    wake_region = (
        (X >= wake_start) & (X <= wake_end) &
        (Y >= Cy - wake_width) & (Y <= Cy + wake_width)
    )
    mask[wake_region] = 1
    
    return mask


class CylinderFlowPINNSimulation:
    """
    PINN-based solver for flow around cylinder.
    
    This class handles PINN training and prediction for cylinder flow problems.
    """
    
    def __init__(self, network, pinn_model, optimizer,
                 Re=100, num_train_samples=5000, num_test_samples=200,
                 x_domain=(0, 2), y_domain=(0, 1),
                 cylinder_center=(0.5, 0.5), cylinder_radius=0.1,
                 inlet_velocity=1.0):
        """
        Initialize PINN cylinder flow simulation.
        
        Parameters:
        -----------
        network : tf.keras.Model
            Neural network model
        pinn_model : PINN
            Physics-informed neural network wrapper
        optimizer : 
            Optimizer for training (e.g., L-BFGS-B)
        Re : float
            Reynolds number
        num_train_samples : int
            Number of training samples
        num_test_samples : int
            Number of test samples for visualization
        x_domain, y_domain : tuple
            Domain bounds
        cylinder_center : tuple
            Cylinder center (Cx, Cy)
        cylinder_radius : float
            Cylinder radius
        inlet_velocity : float
            Inlet velocity
        """
        self.network = network
        self.pinn_model = pinn_model
        self.optimizer = optimizer
        self.Re = Re
        self.num_train_samples = num_train_samples
        self.num_test_samples = num_test_samples
        
        self.x_ini, self.x_f = x_domain
        self.y_ini, self.y_f = y_domain
        self.Cx, self.Cy = cylinder_center
        self.a = cylinder_radius  # Semi-axis a (for ellipse generalization)
        self.b = cylinder_radius  # Semi-axis b
        self.u_inlet = inlet_velocity
    
    def inlet_profile(self, xy):
        """Parabolic inlet velocity profile."""
        y = xy[..., 1, None]
        H = self.y_f - self.y_ini
        return 4 * self.u_inlet * (y - self.y_ini) * (self.y_f - y) / (H**2)
    
    def generate_training_data(self):
        """Generate training points for PINN."""
        N = self.num_train_samples
        
        # Domain points (excluding cylinder interior)
        xy_eqn = np.random.rand(N, 2)
        xy_eqn[..., 0] = (self.x_f - self.x_ini) * xy_eqn[..., 0] + self.x_ini
        xy_eqn[..., 1] = (self.y_f - self.y_ini) * xy_eqn[..., 1] + self.y_ini
        
        # Remove points inside cylinder
        for i in range(N):
            while (xy_eqn[i, 0] - self.Cx)**2/self.a**2 + (xy_eqn[i, 1] - self.Cy)**2/self.b**2 < 1:
                xy_eqn[i, 0] = (self.x_f - self.x_ini) * np.random.rand() + self.x_ini
                xy_eqn[i, 1] = (self.y_f - self.y_ini) * np.random.rand() + self.y_ini
        
        # Cylinder surface points
        xy_circle = np.random.rand(N, 2)
        xy_circle[..., 0] = 2 * self.a * xy_circle[..., 0] + (self.Cx - self.a)
        xy_circle[:N//2, 1] = self.b * np.sqrt(1 - ((xy_circle[:N//2, 0] - self.Cx) / self.a)**2) + self.Cy
        xy_circle[N//2:, 1] = -self.b * np.sqrt(1 - ((xy_circle[N//2:, 0] - self.Cx) / self.a)**2) + self.Cy
        
        # Bottom wall
        xy_bottom = np.random.rand(N, 2)
        xy_bottom[..., 0] = (self.x_f - self.x_ini) * xy_bottom[..., 0] + self.x_ini
        xy_bottom[..., 1] = self.y_ini
        
        # Top wall
        xy_top = np.random.rand(N, 2)
        xy_top[..., 0] = (self.x_f - self.x_ini) * xy_top[..., 0] + self.x_ini
        xy_top[..., 1] = self.y_f
        
        # Outlet
        xy_outlet = np.random.rand(N, 2)
        xy_outlet[..., 0] = self.x_f
        xy_outlet[..., 1] = (self.y_f - self.y_ini) * xy_outlet[..., 1] + self.y_ini
        
        # Inlet
        xy_inlet = np.random.rand(N, 2)
        xy_inlet[..., 0] = self.x_ini
        xy_inlet[..., 1] = (self.y_f - self.y_ini) * xy_inlet[..., 1] + self.y_ini
        
        x_train = [xy_eqn, xy_bottom, xy_top, xy_outlet, xy_inlet, xy_circle]
        
        # Training outputs
        zeros = np.zeros((N, 3))
        
        # Inlet condition: parabolic profile
        u_in = self.inlet_profile(tf.constant(xy_inlet)).numpy()
        v_in = np.zeros((N, 1))
        p_in = u_in  # Pressure reference
        inlet_cond = np.concatenate([u_in, v_in, p_in], axis=-1)
        
        y_train = [zeros, inlet_cond, zeros, zeros, zeros, zeros]
        
        return x_train, y_train
    
    def train(self):
        """Train the PINN model."""
        x_train, y_train = self.generate_training_data()
        self.optimizer.fit(x_train, y_train)
    
    def predict(self, xy):
        """Predict velocity and pressure fields."""
        return self.network.predict(xy, batch_size=len(xy))
    
    def compute_velocities(self, xy):
        """Compute u, v from network output."""
        xy_tf = tf.constant(xy)
        x, y = [xy_tf[..., i, tf.newaxis] for i in range(xy_tf.shape[-1])]
        
        with tf.GradientTape(persistent=True) as g:
            g.watch(x)
            g.watch(y)
            uvp = self.network(tf.concat([x, y], axis=-1))
            u = uvp[..., 0, tf.newaxis]
            v = uvp[..., 1, tf.newaxis]
        
        return u.numpy().squeeze(), v.numpy().squeeze()


# Import jax at module level for use in methods
import jax
