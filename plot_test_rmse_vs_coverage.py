"""Plot RMSE vs coverage for the held-out cylinder test."""

import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

plt.style.use(['science', 'no-latex'])

# Sweep + optimum in one array, sorted by coverage.
COV = np.array([0.000, 0.097, 0.198, 0.298, 0.399,
                0.4766,                                  # optimum
                0.500, 0.600, 0.700, 0.799, 0.900, 1.000])
RMSE = np.array([0.1212, 0.124339, 0.099376, 0.078327, 0.057381,
                 0.022706,                               # optimum
                 0.032783, 0.032102, 0.072165, 0.084395, 0.334774, 0.000])

OPT_IDX = 5  # index of the optimum in the arrays above

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(COV * 100, RMSE, '-', linewidth=1.5, color='tab:blue', zorder=2)
mask = np.ones_like(COV, dtype=bool); mask[OPT_IDX] = False
ax.plot(COV[mask] * 100, RMSE[mask], 'o', markersize=4.5,
        color='tab:blue', zorder=3)
ax.plot(COV[OPT_IDX] * 100, RMSE[OPT_IDX], '*',
        markersize=16, color='tab:green', markeredgecolor='black',
        markeredgewidth=0.6, zorder=4,
        label=f'Optimal (cov={COV[OPT_IDX]*100:.1f}%)')

ax.set_xlabel('Coverage (% solved by CFD)')
ax.set_ylabel('Hybrid RMSE (vs CFD)')
ax.set_title('RMSE vs Coverage  (held-out test, β = 1.1)')
ax.set_xlim(-3, 103)
ax.legend(loc='upper left', frameon=False)
fig.tight_layout()
fig.savefig('rmse_vs_coverage_test.pdf', dpi=1200, bbox_inches='tight')
print('Saved rmse_vs_coverage_test.pdf')
