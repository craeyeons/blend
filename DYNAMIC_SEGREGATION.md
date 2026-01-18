# Dynamic Segregation Based on Complexity Scoring

This document explains the new dynamic segregation approach for hybrid PINN-CFD solutions. Instead of using hard-coded masks defining CFD and PINN regions, this method automatically segregates the domain based on local flow complexity.

## Overview

The segregation is based on computing local diagnostic quantities at every point in the domain:

1. **Strain-rate magnitude** - Characterizes local shear stress
2. **Vorticity magnitude** - Characterizes local rotation
3. **Momentum equation residual** - Indicates where the PINN struggles to satisfy the momentum equations
4. **Continuity residual** - Indicates mass conservation violations

These quantities are combined into a **local complexity score** using:

$$D(x,y) = w_1 \frac{\|\mathbf{S}\|}{S_{ref}} + w_2 \frac{|\omega|}{\omega_{ref}} + w_3 \frac{R_m}{R_{m,ref}} + w_4 \frac{R_c}{R_{c,ref}}$$

where:
- $\|\mathbf{S}\|$ = strain-rate magnitude
- $|\omega|$ = vorticity magnitude
- $R_m$ = momentum equation residual
- $R_c$ = continuity equation residual
- $w_i$ = weights (default: strain=1.0, vorticity=1.0, momentum=2.0, continuity=2.0)

## Usage

### For Cavity Flow

```python
from lib.cavity_flow import CavityFlowDynamicHybridSimulation
from lib.network import Network

# Load PINN model
network = Network().build()
network.load_weights('path/to/model.h5')

# Create initial flow field (e.g., from uniform flow or previous simulation)
u_init = np.ones((N, N)) * 0.1  # Initial guess
v_init = np.zeros((N, N))
p_init = np.zeros((N, N))

# Create dynamic hybrid simulation
sim = CavityFlowDynamicHybridSimulation(
    network=network,
    uv_func=compute_uv_from_psi,  # Function to extract (u,v) from network
    u_init=u_init,
    v_init=v_init,
    p_init=p_init,
    Re=100,
    N=100,
    max_iter=200000,
    tol=1e-6,
    complexity_threshold=1.0,  # Adjust this to control CFD/PINN split
    complexity_weights={
        'strain': 1.0,
        'vorticity': 1.0,
        'momentum': 2.0,  # Residuals weighted higher
        'continuity': 2.0
    },
    normalization='mean'  # or 'max', 'percentile'
)

# Solve
u, v, p = sim.solve()
```

### For Cylinder Flow

```python
from lib.cylinder_flow import CylinderFlowDynamicHybridSimulation
from cylinder_network import Network as CylinderNetwork

# Load PINN model
network = CylinderNetwork().build(num_inputs=2, layers=[48,48,48,48], 
                                  activation='tanh', num_outputs=3)
network.load_weights('path/to/model.h5')

# Create initial flow field
u_init = np.ones((Ny, Nx)) * 1.0  # Inlet velocity
v_init = np.zeros((Ny, Nx))
p_init = np.zeros((Ny, Nx))

# Create dynamic hybrid simulation
sim = CylinderFlowDynamicHybridSimulation(
    network=network,
    uv_func=extract_uv_from_network,
    u_init=u_init,
    v_init=v_init,
    p_init=p_init,
    Re=100,
    N=100,
    complexity_threshold=1.5,  # Higher threshold = more PINN, less CFD
    x_domain=(0, 2),
    y_domain=(0, 1),
    cylinder_center=(0.5, 0.5),
    cylinder_radius=0.1
)

# Solve
u, v, p = sim.solve()
```

## Key Parameters

### `complexity_threshold`
- **Type**: float
- **Default**: 1.0
- **Effect**: 
  - Low values (< 0.5): More points assigned to CFD
  - Mid values (0.5-1.5): Balanced split
  - High values (> 1.5): More points assigned to PINN
- **Interpretation**: Points where $D(x,y) > \tau$ are assigned to CFD

### `complexity_weights`
Dictionary controlling the relative importance of diagnostic quantities:
```python
{
    'strain': 1.0,        # Strain-rate magnitude weight
    'vorticity': 1.0,     # Vorticity magnitude weight
    'momentum': 2.0,      # Momentum residual weight (higher priority)
    'continuity': 2.0     # Continuity residual weight (higher priority)
}
```

**Recommendation**: Keep residual weights ≥ 2.0 since PDE residuals directly indicate where PINN fails.

### `normalization`
- **'mean'**: Normalize by domain-mean value (default)
- **'max'**: Normalize by maximum value
- **'percentile'**: Normalize by 95th percentile

## Diagnostic Quantities

### 1. Strain-Rate Magnitude

$$\|\mathbf{S}\| = \sqrt{2 S_{ij} S_{ij}}$$

where the strain-rate tensor is:
$$S_{ij} = \frac{1}{2}\left(\frac{\partial u_i}{\partial x_j} + \frac{\partial u_j}{\partial x_i}\right)$$

**Indicates**: High-shear regions (boundary layers, shear layers)

### 2. Vorticity Magnitude

$$|\omega| = \left|\frac{\partial v}{\partial x} - \frac{\partial u}{\partial y}\right|$$

**Indicates**: Rotating flows (vortex cores, recirculation zones)

### 3. Momentum Equation Residual

$$R_m = \sqrt{r_u^2 + r_v^2}$$

where:
$$r_u = \rho\left(u\frac{\partial u}{\partial x} + v\frac{\partial u}{\partial y}\right) + \frac{\partial p}{\partial x} - \nu\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right)$$

**Indicates**: Where momentum equations are difficult to satisfy (non-smooth regions)

### 4. Continuity Equation Residual

$$R_c = \left|\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y}\right|$$

**Indicates**: Mass conservation violations

## Output and Statistics

After setting up the segregation, the code prints:
```
Computing dynamic segregation mask based on complexity scoring...
  CFD region: 3450 cells (34.5%)
  PINN region: 6550 cells (65.5%)
  Complexity score range: [0.0234, 5.3421]
  Mean complexity: 1.2345 ± 0.8765
  CFD avg complexity: 2.8901
  PINN avg complexity: 0.4561
```

This shows:
- Percentage split between CFD and PINN
- Range of complexity scores
- Statistics about segregation quality

## Interpretation Guidelines

### Good Segregation
- **CFD regions** have significantly higher average complexity than PINN regions
- Boundary layers and sharp gradients are assigned to CFD
- Smooth, far-field regions are assigned to PINN
- CFD percentage typically 20-50% of domain

### Too Much CFD
- `CFD avg complexity` and `PINN avg complexity` are similar
- **Solution**: Increase `complexity_threshold`

### Too Much PINN
- `complexity_threshold` was too high
- Many high-complexity regions assigned to PINN
- **Solution**: Decrease `complexity_threshold`

## Choosing Threshold Values

### Initial Run Strategy

1. Start with `complexity_threshold = 1.0` (default)
2. Run and examine the statistics
3. Adjust based on results:
   - If CFD% too high: increase threshold
   - If CFD% too low: decrease threshold

### Problem-Specific Guidance

**Cavity Flow** (Re=100):
- `threshold ≈ 0.8-1.2` works well
- Concentrates CFD near lid and corners

**Cylinder Flow** (Re=100):
- `threshold ≈ 1.0-1.5` works well
- CFD naturally concentrates around cylinder and wake

**High Reynolds** (Re > 1000):
- May need `threshold ≈ 0.5-0.8`
- More complex flow requires more CFD

## Advanced: Custom Weights

Modify weights to emphasize different physics:

```python
# Conservative: emphasize PDE residuals
weights = {
    'strain': 0.5,
    'vorticity': 0.5,
    'momentum': 3.0,     # High priority
    'continuity': 3.0     # High priority
}

# Physics-focused: emphasize flow features
weights = {
    'strain': 2.0,        # Emphasize shear
    'vorticity': 2.0,     # Emphasize rotation
    'momentum': 1.0,
    'continuity': 1.0
}
```

## Performance Considerations

1. **Complexity score computation** is done once at setup (negligible cost)
2. **Dynamic segregation** may require more CFD iterations due to different mask
3. **Total runtime** typically similar to hard-coded masks, but quality may improve

## Comparison with Hard-Coded Masks

| Aspect | Hard-Coded | Dynamic |
|--------|-----------|---------|
| CFD region | Fixed shape | Adapted to flow |
| Boundary layer handling | Manual | Automatic |
| Parameter tuning | Border width | Threshold |
| Adaptability | Low | High |
| Physics information used | None | Diagnostic quantities |

## References

The complexity scoring approach is based on the principle that:
- **High complexity regions** (steep gradients, nonlinear coupling) → Use CFD
- **Low complexity regions** (smooth, diffusion-dominated) → Use PINN

This ensures physics-informed allocation of computational resources between expensive CFD (accurate, slow) and learned PINN (faster, needs smooth regions).
