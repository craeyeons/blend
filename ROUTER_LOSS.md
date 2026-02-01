# Router Loss Function Documentation

## Overview

The CNN router learns to partition a fluid domain into **CFD regions** (r=1) and **PINN regions** (r=0). The goal is to minimize CFD usage while maintaining solution accuracy by only using CFD where the PINN has high residual errors.

## Total Loss

$$\mathcal{L} = \mathcal{L}_{\text{CFD}} + \mathcal{L}_{\text{residual}} + \mathcal{L}_{\text{TV}} - \lambda_H \cdot H(r) - \lambda_V \cdot \text{Var}(r)$$

---

## Loss Components

### 1. CFD Cost ($\mathcal{L}_{\text{CFD}}$)

$$\mathcal{L}_{\text{CFD}} = \beta \cdot \frac{1}{N} \sum_{i \in \Omega_{\text{fluid}}} r_i$$

- **Range:** $[0, \beta]$
- **Meaning:** Mean router output × β (CFD fraction scaled by β)
- **Effect:** Penalizes assigning regions to CFD

| β value | Interpretation |
|---------|----------------|
| 0.1 | Low CFD penalty, prioritize accuracy |
| 1.0 | Equal weight to CFD cost and residual |
| 2.0 | Strong CFD penalty, tolerate higher residual |

---

### 2. PINN Residual Loss ($\mathcal{L}_{\text{residual}}$)

$$\mathcal{L}_{\text{residual}} = \frac{1}{N} \sum_{i \in \Omega_{\text{fluid}}} (1 - r_i) \cdot \tilde{R}_i$$

Where $\tilde{R}_i$ is the **normalized residual** at point $i$.

- **Range:** $[0, 1]$
- **Meaning:** Mean residual weighted by PINN assignment
- **Effect:** Penalizes assigning high-residual regions to PINN

#### Residual Normalization (Normalize-then-Clip)

Each residual component is normalized by the **95th percentile**, then clipped:

```
R_norm = clip(R / p95, 0, 1.5)
```

This ensures:
- Values normalized relative to the 95th percentile
- Outliers can contribute up to 1.5× (penalized more than typical points)
- Extreme outliers (>1.5× p95) are capped at 1.5
- Robust to extreme values while preserving outlier signal

#### Residual Components

$$R = \frac{w_c \cdot R_{\text{continuity}} + w_m \cdot R_{\text{momentum}} + w_b \cdot R_{\text{BC}} + w_p \cdot R_{\text{propagated}}}{w_c + w_m + w_b + w_p}$$

| Component | Description | Default Weight |
|-----------|-------------|----------------|
| $R_{\text{continuity}}$ | $\|\nabla \cdot \mathbf{u}\|$ | 1.0 |
| $R_{\text{momentum}}$ | $\sqrt{r_u^2 + r_v^2}$ (Navier-Stokes residual) | 1.0 |
| $R_{\text{BC}}$ | Boundary condition error | 2.0 |
| $R_{\text{propagated}}$ | Upstream error propagation | 1.5 |

---

### 3. Total Variation ($\mathcal{L}_{\text{TV}}$)

$$\mathcal{L}_{\text{TV}} = \lambda_{\text{TV}} \cdot \left( \frac{1}{N} \sum_{i,j} |r_{i,j} - r_{i,j+1}| + |r_{i,j} - r_{i+1,j}| \right)$$

- **Range:** $[0, \lambda_{\text{TV}}]$
- **Meaning:** Mean absolute difference between adjacent pixels
- **Effect:** Encourages spatially smooth masks (no salt-and-pepper noise)
- **Typical value:** $\lambda_{\text{TV}} = 0.01$

---

### 4. Entropy Regularization ($-\lambda_H \cdot H(r)$)

$$H(r) = -\frac{1}{N} \sum_{i} \left[ r_i \log r_i + (1-r_i) \log(1-r_i) \right]$$

- **Range:** $H(r) \in [0, \log 2] \approx [0, 0.693]$
- **Maximum:** At $r = 0.5$ (uniform uncertainty)
- **Minimum:** At $r = 0$ or $r = 1$ (full certainty)
- **Effect:** **Maximized** to encourage non-extreme outputs (prevents all-0 or all-1)
- **Typical value:** $\lambda_H = 0.1$ to $5.0$

---

### 5. Variance Regularization ($-\lambda_V \cdot \text{Var}(r)$)

$$\text{Var}(r) = \frac{1}{N} \sum_{i} (r_i - \bar{r})^2$$

- **Range:** $[0, 0.25]$ (max at $r = 0$ for half, $r = 1$ for half)
- **Effect:** **Maximized** to encourage diverse outputs (not all same value)
- **Typical value:** $\lambda_V = 0.05$ to $2.0$

---

## Temperature-Scaled Sigmoid

The router outputs logits $z$ which are passed through a temperature-scaled sigmoid:

$$r = \sigma(z / T) = \frac{1}{1 + e^{-z/T}}$$

| Temperature | Effect |
|-------------|--------|
| T = 0.2 | Very soft, outputs cluster around 0.5 |
| T = 0.5 | Soft, gradual transitions |
| T = 1.0 | Standard sigmoid |
| T = 2.0 | Sharp, outputs pushed toward 0 or 1 |

**Recommendation:** Start with $T = 0.3-0.5$ for stable training.

---

## Gradient Clipping

Gradients are clipped by global norm to prevent exploding gradients:

```python
gradients, _ = tf.clip_by_global_norm(gradients, grad_clip_norm)
```

**Typical value:** `grad_clip_norm = 1.0`

---

## Recommended Hyperparameters

### Balanced Training (start here)
```bash
python train_router.py \
    --beta 1.0 \
    --lambda-tv 0.01 \
    --lambda-entropy 1.0 \
    --lambda-variance 0.5 \
    --temperature 0.5 \
    --lr 5e-5 \
    --grad-clip 1.0
```

### If router collapses to all-CFD or all-PINN
```bash
python train_router.py \
    --beta 1.0 \
    --lambda-entropy 5.0 \
    --lambda-variance 2.0 \
    --temperature 0.3 \
    --lr 1e-4
```

### If you want more CFD
- Increase `--beta` (e.g., 0.5 → more CFD allowed)
- Decrease residual weights

### If you want less CFD
- Decrease `--beta` (e.g., 2.0 → penalize CFD more)

---

## Monitoring Training

| Metric | Good Range | Problem if... |
|--------|------------|---------------|
| CFD% | 20-60% | 0% or 100% (collapsed) |
| Entropy | > 0.3 | 0 (saturated outputs) |
| Variance | > 0.05 | 0 (uniform outputs) |
| Residual | Decreasing | Stuck at 1.0 |

---

## Loss Scales Summary

| Component | Range | Notes |
|-----------|-------|-------|
| CFD cost | $[0, \beta]$ | β scales this term |
| Residual | $[0, 1.5]$ | Normalized by p95, clipped at 1.5 |
| TV | $[0, \lambda_{TV}]$ | Usually small |
| Entropy | $[0, 0.693]$ | Maximized (subtracted) |
| Variance | $[0, 0.25]$ | Maximized (subtracted) |
