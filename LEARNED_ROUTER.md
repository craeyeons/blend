# Learned Router for Hybrid PINN-CFD Domain Segregation

## Overview

This module implements a neural network-based router for intelligently segregating the computational domain into PINN and CFD regions. Unlike threshold-based approaches that produce patchy, disconnected masks, the learned router generates **smooth, connected regions** while being aware of **boundary condition violations** and **upstream error propagation**.

## Problem with Threshold-Based Approach

The original complexity-based threshold approach has several limitations:

1. **Patchy Masks**: Binary thresholding on local complexity scores creates disconnected islands
2. **No Boundary Awareness**: Doesn't learn that PINN fails at inlet despite low complexity
3. **No Upstream Propagation**: Doesn't account for convective error transport
4. **Arbitrary Threshold**: Requires manual tuning per case

## Learned Router Solution

### Architecture

```
Input Features (per grid point) → CNN → Rejection Probability r(x) ∈ [0,1]
```

The router is a 2D Convolutional Neural Network with:
- Multi-scale dilated convolutions (dilation rates: 1, 2, 4, 1)
- Batch normalization and dropout for regularization
- Sigmoid output for soft decisions

### Input Features (13 channels)

| Feature | Description | Purpose |
|---------|-------------|---------|
| `x_norm, y_norm` | Normalized coordinates | Spatial context |
| `d_inlet` | Distance to inlet | Boundary awareness |
| `d_cyl` | Distance to cylinder | Obstacle awareness |
| `ξ, η` | Flow-aligned coordinates | Convection direction |
| `u, v, p` | PINN velocity/pressure | Flow state |
| `‖∇u‖, ‖∇v‖` | Velocity gradients | Shear detection |
| `D` | Complexity score | PDE difficulty |
| `D_BC` | Boundary violation | BC mismatch detection |

### Loss Function

```
L_total = L_base + λ_spatial·L_spatial + λ_upstream·L_upstream + λ_connect·L_connect
```

#### 1. Base Loss (Local Complexity)
```python
L_base = mean[(1 - r(x)) · D(x) + r(x) · τ]
```
- Penalize accepting PINN where complexity is high
- Penalize excessive rejection (CFD has fixed cost τ)

#### 2. Spatial Regularization (Smoothness)
```python
L_spatial = Σ w_ij · |r(x_i) - r(x_j)|
w_ij = exp(-‖x_i - x_j‖²/h²) · (1 + α·flow_alignment)
```
- Encourage smooth mask transitions
- Stronger coupling along flow direction (α=2)

#### 3. Upstream Penalty (Boundary Awareness)
```python
L_upstream = Σ (1 - r(x)) · [D_BC(x) + D(x)] · exp(-d_inlet/δ)
```
- Strongly penalize accepting PINN near inlet where BC violations occur
- Exponential decay of penalty with distance from inlet

#### 4. Connectivity Penalty (Region Coherence)
```python
L_connect = mean(|∇²r|)
```
- Penalize isolated islands via Laplacian regularization
- Encourages connected PINN regions

## Usage

### Quick Start

```python
from lib.router import create_router, LearnedRouter
from lib.cylinder_flow import CylinderFlowLearnedRouterSimulation

# Create router system
learned_router, trainer = create_router(
    x_domain=(0, 2),
    y_domain=(0, 1),
    cylinder_center=(0.5, 0.5),
    cylinder_radius=0.1,
    nu=0.01,
    # Loss weights
    lambda_spatial=0.1,
    lambda_upstream=1.0,
    lambda_connect=0.5
)

# Prepare training data (requires CFD ground truth)
training_data = trainer.prepare_training_data(
    u_pinn, v_pinn, p_pinn,
    X, Y, dx, dy,
    u_cfd=u_cfd, v_cfd=v_cfd,
    error_threshold=0.1
)

# Train router
history = trainer.train([training_data], epochs=100)

# Use trained router in simulation
sim = CylinderFlowLearnedRouterSimulation(
    network=pinn_network,
    uv_func=compute_uv,
    learned_router=learned_router,
    u_init=u_pinn,
    v_init=v_pinn,
    p_init=p_pinn,
    Re=100, N=100
)
u, v, p = sim.solve()
```

### Inference Only (Pre-trained Router)

```python
# Load pre-trained router
learned_router.load('./models/router_cylinder.h5')

# Predict mask
mask, r_prob = learned_router.predict_mask(
    u_pinn, v_pinn, p_pinn,
    X, Y, dx, dy,
    cylinder_mask=cylinder_mask
)
```

## Hyperparameters

### Loss Weights (Recommended Starting Values)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `threshold` (τ) | 1.0 | Target complexity for PINN regions |
| `lambda_spatial` | 0.1 | Weight for mask smoothness |
| `lambda_upstream` | 1.0 | Weight for inlet/BC awareness |
| `lambda_connect` | 0.5 | Weight for region connectivity |

### Physical Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `delta_influence` | 0.2 | Influence length (fraction of Lx) |
| `xi_threshold` | 0.3 | Upstream region cutoff |
| `alpha_flow` | 2.0 | Flow-directional coupling strength |

### Network Architecture

| Parameter | Default | Description |
|-----------|---------|-------------|
| `filters` | [32, 64, 64, 32] | Conv layer filter counts |
| `kernel_size` | 3 | Convolution kernel size |
| `dropout_rate` | 0.1 | Dropout for regularization |

## Expected Behavior

After training, the router should:

1. **Reject PINN (r ≈ 1)** in:
   - Upstream/inlet region (BC violations)
   - Near cylinder surface (boundary layer)
   - Domain boundaries

2. **Accept PINN (r ≈ 0)** in:
   - Wake/downstream region
   - Far-field where flow is smooth

3. **Produce smooth transitions** between regions

4. **Generate connected CFD domain** from inlet to ensure proper upstream physics

## Files

| File | Description |
|------|-------------|
| `lib/router.py` | Core router implementation |
| `lib/cylinder_flow.py` | `CylinderFlowLearnedRouterSimulation` class |
| `example_learned_router.py` | Complete training and evaluation example |
| `models/router_cylinder.h5` | Saved router weights (after training) |

## Comparison: Threshold vs Learned Router

| Aspect | Threshold-Based | Learned Router |
|--------|-----------------|----------------|
| Mask smoothness | Patchy | Smooth, connected |
| Boundary awareness | None | Learned from data |
| Upstream propagation | Ignored | Penalized via L_upstream |
| Parameter tuning | Per-case threshold | Trained once, generalizes |
| Training data needed | None | CFD ground truth |

## References

The learned router approach is inspired by:
- Meta-learning for adaptive mesh refinement
- Attention mechanisms in physics-informed neural networks
- Hybrid solver domain decomposition methods
