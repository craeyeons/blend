# Data Format for Learned Router Training

## Overview

The router is trained on 25 cases: 5 Reynolds numbers × 5 cylinder positions.

**Training requires:**
- **CFD solutions** (`.npz` files) — Computed once and saved to disk
- **PINN models** (`.h5` files) — Queried directly during training (no pre-saved PINN predictions needed)

## File Structure

```
data/
├── cases.json                    # Metadata for all cases
├── Re_1/
│   ├── cyl_pos_0.5_0.3/
│   │   └── cfd_solution.npz      # CFD ground truth
│   ├── cyl_pos_0.5_0.4/
│   │   └── cfd_solution.npz
│   └── ...
├── Re_10/
├── Re_40/
├── Re_100/
└── Re_200/

models/
├── pinn_cylinder_1.0.h5          # PINN model for Re=1
├── pinn_cylinder_10.0.h5         # PINN model for Re=10
├── pinn_cylinder_40.0.h5         # PINN model for Re=40
├── pinn_cylinder_100.0.h5        # PINN model for Re=100
└── pinn_cylinder_200.0.h5        # PINN model for Re=200
```

## Quick Start

### Step 1: Generate CFD Data

```bash
# Compute all 25 CFD solutions
python generate_cfd_data.py

# Or compute only specific Reynolds numbers
python generate_cfd_data.py --Re 100 200

# Check existing data
python generate_cfd_data.py --check

# Compute only missing cases
python generate_cfd_data.py --missing
```

### Step 2: Train Router

```bash
# Train on all cases
python example_learned_router.py train
```

## File Formats

### 1. cases.json (Metadata)

```json
{
    "description": "Training cases for learned router",
    "note": "PINN models loaded from ./models/pinn_cylinder_{Re}.h5",
    "domain": {
        "x_domain": [0.0, 2.0],
        "y_domain": [0.0, 1.0],
        "N": 100
    },
    "cases": [
        {
            "id": "Re100_cyl_0.5_0.5",
            "Re": 100.0,
            "cylinder_center": [0.5, 0.5],
            "cylinder_radius": 0.1,
            "inlet_velocity": 1.0,
            "cfd_path": "Re_100/cyl_pos_0.5_0.5/cfd_solution.npz"
        }
    ]
}
```

### 2. cfd_solution.npz (CFD Ground Truth)

NumPy compressed archive with the following arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `u` | (Ny, Nx) | x-velocity field |
| `v` | (Ny, Nx) | y-velocity field |
| `p` | (Ny, Nx) | pressure field |
| `X` | (Ny, Nx) | x-coordinate grid |
| `Y` | (Ny, Nx) | y-coordinate grid |
| `cylinder_mask` | (Ny, Nx) | 1 inside cylinder, 0 outside |

**Python code to save:**
```python
import numpy as np

np.savez_compressed(
    'cfd_solution.npz',
    u=u_cfd,           # shape: (Ny, Nx), dtype: float64
    v=v_cfd,           # shape: (Ny, Nx), dtype: float64
    p=p_cfd,           # shape: (Ny, Nx), dtype: float64
    X=X,               # shape: (Ny, Nx), dtype: float64
    Y=Y,               # shape: (Ny, Nx), dtype: float64
    cylinder_mask=cylinder_mask  # shape: (Ny, Nx), dtype: int32
)
```

### 3. PINN Models (pinn_cylinder_{Re}.h5)

PINN models are loaded directly from `.h5` files during training.

**Expected naming convention:**
- `models/pinn_cylinder_1.0.h5` for Re=1
- `models/pinn_cylinder_10.0.h5` for Re=10
- etc.

**During training, PINN predictions are computed on-the-fly:**
```python
from lib.network import Network

network = Network().build(num_inputs=2, num_outputs=3)
network.load_weights(f'models/pinn_cylinder_{Re}.h5')

# Query predictions on grid
xy = np.stack([X.flatten(), Y.flatten()], axis=-1)
uvp = network.predict(xy, batch_size=len(xy))
u_pinn = uvp[..., 0].reshape(X.shape)
v_pinn = uvp[..., 1].reshape(X.shape)
p_pinn = uvp[..., 2].reshape(X.shape)
```

## Recommended Cases

### Reynolds Numbers (5 values)
```python
Re_values = [1.0, 10.0, 40.0, 100.0, 200.0]
```

### Cylinder Positions (5 values)
```python
cylinder_positions = [
    (0.5, 0.3),  # Near bottom wall
    (0.5, 0.4),
    (0.5, 0.5),  # Center
    (0.5, 0.6),
    (0.5, 0.7),  # Near top wall
]
```

## Workflow Summary

```
┌─────────────────────────────────────────────────────────────┐
│  1. Generate CFD Data                                       │
│     python generate_cfd_data.py                             │
│     → Creates data/Re_*/cyl_pos_*/cfd_solution.npz          │
│     → Creates data/cases.json                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  2. Ensure PINN Models Exist                                │
│     models/pinn_cylinder_{Re}.h5                            │
│     (Pre-trained PINN models for each Reynolds number)      │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  3. Train Router                                            │
│     python example_learned_router.py train                  │
│     → Loads CFD from .npz files                             │
│     → Queries PINN models for predictions                   │
│     → Saves router to models/router_multi.h5                │
└─────────────────────────────────────────────────────────────┘
```

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `generate_cfd_data.py` | Compute and save all CFD solutions |
| `example_learned_router.py train` | Train router on saved data |
| `example_learned_router.py template` | Generate empty data structure |
| `example_learned_router.py demo` | Run single-case demo |
