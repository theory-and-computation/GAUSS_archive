"""gaussian_potential.py — the benchmark model potential (single source of truth).

Defines the double-Gaussian model potential used throughout the benchmark:

  * the Langevin integrators in main.py call these functions to obtain the
    potential energy and (unbiased) forces at the particle position;
  * gpr_model.sum_gaus1D() is a thin wrapper around gaussian_potential_1D()
    so the analytic reference (plotted against the GP reconstruction and used
    for the RMSE logged in hp.txt) shares these exact parameters.

Potential shape
---------------
A sum of two isotropic Gaussian "hills", inverted and scaled:

    V(x) = -SCALE * [ C1 exp(-(x-mu1)^2 / (2 sigma1^2))
                      + C2 exp(-(x-mu2)^2 / (2 sigma2^2)) ]

with C_i = 1 / (sigma_i sqrt(2 pi)). The leading minus sign turns the hills
into a *double-well* potential with minima near x = mu1 = 7 (narrow,
sigma1 = 1) and x = mu2 = 12 (broad, sigma2 = 2). The minima are slightly
shifted from mu1/mu2 by the overlap of the two Gaussians.

The 2D version is the product-Gaussian analogue with wells on the diagonal at
(7, 7) and (12, 12); the same potential is used for the 1D and 2D runs.

Sign convention
---------------
Each function returns (energy, force) with force = -dV/dx consistent with the
returned energy. The inversion happens once, here — do not "fix" the sign in
callers without changing the benchmark.

The GP kernel and noise model are NOT in this file — see gpr_model.py
([KERNEL x/3] / [NOISE x/3] markers).
"""
import numpy as np

# ── Single authoritative potential parameters ─────────────────────────────────
# Well 1: centre / width (narrow)
_MU_1, _SIGMA_1 = 7, 1
# Well 2: centre / width (broad)
_MU_2, _SIGMA_2 = 12, 2
# Global energy scale (eV). One value shared by main.py (MD forces) and
# gpr_model.py (analytic reference) via this module.
SCALE = 1.078812634123564


def gaussian_potential_1D(x):
    """
    Double-Gaussian model potential: energy and force in 1D.

    V(x) = -SCALE * [ C1 exp(-(x-mu1)^2 / (2 sigma1^2))
                      + C2 exp(-(x-mu2)^2 / (2 sigma2^2)) ]

    Parameters
    ----------
    x : float or ndarray
        CV coordinate(s) (already periodically wrapped by the caller).

    Returns
    -------
    energy  : float or ndarray   V(x)
    force_x : float or ndarray   Fx = -dV/dx
    """
    C1 = 1.0 / (_SIGMA_1 * np.sqrt(2.0 * np.pi))
    C2 = 1.0 / (_SIGMA_2 * np.sqrt(2.0 * np.pi))

    # Gaussian hills (positive, before inversion)
    f1 = C1 * np.exp(-(x - _MU_1)**2 / (2.0 * _SIGMA_1**2))
    f2 = C2 * np.exp(-(x - _MU_2)**2 / (2.0 * _SIGMA_2**2))
    energy = f1 + f2

    # Gradient of each hill:  d f_i / dx = f_i * (-(x - mu_i) / sigma_i^2)
    dx1 = -(x - _MU_1) / _SIGMA_1**2 * f1
    dx2 = -(x - _MU_2) / _SIGMA_2**2 * f2

    # Force of the raw (positive) potential: -d(energy)/dx
    force_x = -(dx1 + dx2)

    # Invert (hills -> wells) and scale. Energy and force are multiplied by
    # the same factor (-SCALE), which preserves Fx = -dV/dx for V = -SCALE*energy.
    return -1.0 * energy * SCALE, -1.0 * force_x * SCALE


def gaussian_potential_2D(x, y):
    """
    Double-Gaussian model potential: energy and forces in 2D.

    Product Gaussians centred on the diagonal at (mu_i, mu_i) with equal
    x/y widths, inverted and scaled exactly as in gaussian_potential_1D:

    V(x, y) = -SCALE * [ C1 exp(-(x-mu1)^2/2s1^2 - (y-mu1)^2/2s1^2)
                         + C2 exp(-(x-mu2)^2/2s2^2 - (y-mu2)^2/2s2^2) ]

    Parameters
    ----------
    x, y : float or ndarray
        CV coordinates (already periodically wrapped by the caller).

    Returns
    -------
    energy  : float or ndarray   V(x, y)
    force_x : float or ndarray   Fx = -dV/dx
    force_y : float or ndarray   Fy = -dV/dy
    """
    C1 = 1.0 / (_SIGMA_1 * np.sqrt(2.0 * np.pi))
    C2 = 1.0 / (_SIGMA_2 * np.sqrt(2.0 * np.pi))

    # Product Gaussian hills (positive, before inversion)
    f1 = C1 * np.exp(-(x - _MU_1)**2 / (2.0 * _SIGMA_1**2)
                     -(y - _MU_1)**2 / (2.0 * _SIGMA_1**2))
    f2 = C2 * np.exp(-(x - _MU_2)**2 / (2.0 * _SIGMA_2**2)
                     -(y - _MU_2)**2 / (2.0 * _SIGMA_2**2))
    energy = f1 + f2

    # Per-hill partial derivatives:
    #   d f_i / dx = f_i * (-(x - mu_i) / sigma_i^2)
    #   d f_i / dy = f_i * (-(y - mu_i) / sigma_i^2)
    dx1 = -(x - _MU_1) / _SIGMA_1**2 * f1
    dx2 = -(x - _MU_2) / _SIGMA_2**2 * f2
    dy1 = -(y - _MU_1) / _SIGMA_1**2 * f1
    dy2 = -(y - _MU_2) / _SIGMA_2**2 * f2

    # Forces of the raw (positive) potential: -d(energy)/d(x, y)
    force_x = -(dx1 + dx2)
    force_y = -(dy1 + dy2)

    # Invert (hills -> wells) and scale; same (-SCALE) factor on energy and
    # forces preserves F = -grad V for V = -SCALE*energy.
    return -1.0 * energy * SCALE, -1.0 * force_x * SCALE, -1.0 * force_y * SCALE