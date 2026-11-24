"""
Gaussian Process Regression (GPR) Force Modeling & Free Energy Reconstruction.

This module provides:
1. Custom GPyTorch derivative kernels (`PeriodicDerivativeKernel2D`) for learning
   free-energy gradient fields directly from observed forces.
2. 2D Free-energy surface (FES) reconstruction via GP posterior sampling and
   Poisson/FFT integration.
3. Uncertainty-driven exploration bias generation (push forces) targeting 
   undersampled, high-variance regions of collective variable (CV) space.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

import gpytorch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from scipy.fft import fft2, ifft2
import scipy.linalg as la
import torch

if TYPE_CHECKING:
    import logger as modular_log


# ---------------------------------------------------------------------------
# GP Derivative Kernel & Model Definitions
# ---------------------------------------------------------------------------

# =============================================================================
# [DEFINITION] 2D PERIODIC DERIVATIVE COVARIANCE KERNEL
# =============================================================================

class PeriodicDerivativeKernel2D(gpytorch.kernels.Kernel):
    r"""
    Periodic ARD covariance kernel for 2D potential derivative observations.

    Models Cov( dA/dx_a(x1), dA/dx_b(x2) ) where latent potential A(x) follows:
        k(x1, x2) = exp( -2 * \sum_d sin^2(\pi (x1_d - x2_d) / p_d) / \ell_d^2 )

    Input format: [x, y, component_index] where:
        - component_index = 0 -> dA/dx  (force component along CV1)
        - component_index = 1 -> dA/dy  (force component along CV2)
    """

    has_lengthscale = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter(
            "raw_period_length", torch.nn.Parameter(torch.zeros(1, 2))
        )
        self.register_constraint("raw_period_length", gpytorch.constraints.Positive())

    @property
    def period_length(self) -> torch.Tensor:
        return self.raw_period_length_constraint.transform(self.raw_period_length)

    @period_length.setter
    def period_length(self, value: torch.Tensor | list | np.ndarray) -> None:
        if not torch.is_tensor(value):
            value = torch.tensor(value, dtype=torch.float32)
        self.initialize(
            raw_period_length=self.raw_period_length_constraint.inverse_transform(
                value.view(1, -1).to(self.raw_period_length)
            )
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params):
        x1_coords = x1[..., :2]
        x2_coords = x2[..., :2]
        c1 = x1[..., 2].long()
        c2 = x2[..., 2].long()

        ell = self.lengthscale.squeeze(0)
        p = self.period_length.squeeze(0)

        # Scale factor: c_d = 2*pi / (p_d * ell_d^2)
        c_coeff = (2.0 * torch.pi / (p * ell**2)).to(x1_coords.dtype)

        if diag:
            same = c1 == c2
            ca = c_coeff[c1]
            pa = p[c1]
            return torch.where(same, ca * (2.0 * torch.pi / pa), torch.zeros_like(ca))

        # Full cross-covariance between derivative components
        dx = x1_coords.unsqueeze(-2) - x2_coords.unsqueeze(-3)
        u = torch.pi * dx / p
        sin2u = torch.sin(2.0 * u)
        cos2u = torch.cos(2.0 * u)

        ell2 = (ell**2).to(x1_coords.dtype)
        sqdist = 2.0 * (torch.sin(u) ** 2 / ell2).sum(dim=-1)
        k_base = torch.exp(-sqdist)

        n1, n2 = k_base.shape
        c1m = c1.view(-1, 1).expand(n1, n2)
        c2m = c2.view(1, -1).expand(n1, n2)

        ca = c_coeff[c1m]
        cb = c_coeff[c2m]
        pa = p[c1m]

        sin2u_a = sin2u.gather(-1, c1m.unsqueeze(-1)).squeeze(-1)
        sin2u_b = sin2u.gather(-1, c2m.unsqueeze(-1)).squeeze(-1)
        cos2u_a = cos2u.gather(-1, c1m.unsqueeze(-1)).squeeze(-1)

        same = c1m == c2m
        k_same = k_base * (ca * (2.0 * torch.pi / pa) * cos2u_a - (ca**2) * (sin2u_a**2))
        k_cross = -k_base * ca * cb * sin2u_a * sin2u_b

        return torch.where(same, k_same, k_cross)

class PotentialGradGP(gpytorch.models.ExactGP):
    """GP regression model for predicting potential gradients."""

    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ---------------------------------------------------------------------------
# Training Data Preprocessing & Subsampling
# ---------------------------------------------------------------------------

def make_training_data(
    x_grid: np.ndarray, y_grid: np.ndarray | None, f_grid: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Prepare 1D or 2D grid/point coordinate data into GP training tensors."""
    x_arr = np.asarray(x_grid)
    f_arr = np.asarray(f_grid)

    if y_grid is None:
        x_np = x_arr.reshape(-1, 1)
        y_np = f_arr.reshape(-1)
    else:
        y_arr = np.asarray(y_grid)
        x_np = np.stack([x_arr.ravel(), y_arr.ravel()], axis=-1)
        y_np = f_arr.ravel()

    return torch.tensor(x_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.float32)


def stack_force_training_data(
    x_fx: torch.Tensor,
    y_fx: torch.Tensor,
    x_fy: torch.Tensor,
    y_fy: torch.Tensor,
    y_is_force: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Combine Fx and Fy datasets into a single derivative dataset with component labels.

    Transforms applied forces to potential gradients (dA/dx = -Fx, dA/dy = -Fy).
    """
    device, dtype = x_fx.device, x_fx.dtype
    n_fx, n_fy = x_fx.shape[0], x_fy.shape[0]

    comp_x = torch.zeros(n_fx, 1, device=device, dtype=dtype)
    comp_y = torch.ones(n_fy, 1, device=device, dtype=dtype)

    train_x = torch.cat(
        [torch.cat([x_fx, comp_x], dim=1), torch.cat([x_fy, comp_y], dim=1)], dim=0
    )

    if y_is_force:
        train_y = torch.cat([-y_fx, -y_fy], dim=0)
    else:
        train_y = torch.cat([y_fx, y_fy], dim=0)

    return train_x, train_y


def make_aug_test(x_test_xy: torch.Tensor, comp_value: float) -> torch.Tensor:
    """Append component flag (0 for d/dx, 1 for d/dy) to evaluation coordinates."""
    comp = torch.full(
        (x_test_xy.shape[0], 1),
        comp_value,
        dtype=x_test_xy.dtype,
        device=x_test_xy.device,
    )
    return torch.cat([x_test_xy, comp], dim=1)


def smart_subsample(
    rows: np.ndarray,
    cols: np.ndarray,
    hist: np.ndarray,
    max_samples: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Spatially stratified subsampling weighted by bin visit counts."""
    num_points = len(rows)
    if num_points <= max_samples:
        return rows, cols

    counts = hist[rows, cols]
    weights = np.sqrt(counts)
    weights /= weights.sum()

    # Stratify spatial regions across the active grid
    n_strata_x = min(10, len(np.unique(rows)))
    n_strata_y = min(10, len(np.unique(cols)))

    row_bins = np.digitize(rows, np.linspace(rows.min(), rows.max(), n_strata_x))
    col_bins = np.digitize(cols, np.linspace(cols.min(), cols.max(), n_strata_y))
    strata = row_bins * n_strata_y + col_bins

    samples_per_stratum = max_samples // max(1, len(np.unique(strata)))
    selected_idx = []

    for stratum in np.unique(strata):
        stratum_mask = strata == stratum
        stratum_idx = np.where(stratum_mask)[0]
        if len(stratum_idx) == 0:
            continue

        n_sample = min(samples_per_stratum, len(stratum_idx))
        s_weights = weights[stratum_idx] / weights[stratum_idx].sum()

        chosen = np.random.choice(stratum_idx, size=n_sample, replace=False, p=s_weights)
        selected_idx.extend(chosen)

    # Fill remaining budget from highest weighted unsampled points
    if len(selected_idx) < max_samples:
        rem_budget = max_samples - len(selected_idx)
        unselected = np.setdiff1d(np.arange(num_points), selected_idx)
        if len(unselected) > 0:
            rem_weights = weights[unselected] / weights[unselected].sum()
            chosen_rem = np.random.choice(
                unselected,
                size=min(rem_budget, len(unselected)),
                replace=False,
                p=rem_weights,
            )
            selected_idx.extend(chosen_rem)

    selected_arr = np.array(selected_idx)
    return rows[selected_arr], cols[selected_arr]


# ---------------------------------------------------------------------------
# Free Energy Integration & Reconstruction
# ---------------------------------------------------------------------------

def predict_energy_from_mean_forces(
    dx: float | np.ndarray,
    dy: float | np.ndarray,
    fx: np.ndarray,
    fy: np.ndarray,
) -> np.ndarray:
    """Integrate 2D mean forces into free energy via Poisson solver (FFT)."""
    ny, nx = fx.shape
    if np.ndim(dx) == 2:
        dx = dx[0, 1] - dx[0, 0]
    if np.ndim(dy) == 2:
        dy = dy[1, 0] - dy[0, 0]

    # Compute divergence of force field: div(F) = dFx/dx + dFy/dy
    dfx_dx = np.gradient(fx, dx, axis=1)
    dfy_dy = np.gradient(fy, dy, axis=0)
    rhs = -(dfx_dx + dfy_dy)

    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    kx_grid, ky_grid = np.meshgrid(kx, ky)

    denom = -(kx_grid**2 + ky_grid**2)
    denom[0, 0] = 1.0  # Avoid zero-frequency division

    rhs_hat = fft2(rhs)
    a_hat = rhs_hat / denom
    a_hat[0, 0] = 0.0

    a_fes = -1.0 * np.real(ifft2(a_hat))
    a_fes -= a_fes.min()
    return a_fes


def reconstruct_A_from_gp_samples_2D_no_norm_fast(
    xv: np.ndarray,
    yv: np.ndarray,
    samples_phys: np.ndarray,
    ell: np.ndarray,
    chi2: float,
    noise: np.ndarray,
    period: np.ndarray | None = None,
    noise_is_std: bool = True,
    y_is_force: bool = True,
    jitter_rel: float = 1e-10,
    center: str = "min",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, None]:
    """Reconstruct potential A(x,y) from GP posterior gradient samples analytically."""
    xv = np.asarray(xv, dtype=float)
    yv = np.asarray(yv, dtype=float)
    nx, ny = xv.shape
    n_pts = nx * ny

    samples_np = np.asarray(samples_phys, dtype=float)
    num_samples, n_comp, nx_s, ny_s = samples_np.shape
    assert n_comp == 2 and nx_s == nx and ny_s == ny

    coords = np.column_stack([xv.ravel(), yv.ravel()])
    ell = np.asarray(ell, dtype=float).ravel()

    if period is None:
        period = np.array([np.ptp(xv), np.ptp(yv)], dtype=float)
    period = np.asarray(period, dtype=float).ravel()

    c_coeff = 2.0 * np.pi / (period * ell**2)
    if chi2 is None or chi2 <= 0.0:
        chi2 = max(1e-6, float(np.var(samples_np)))

    # Compute base kernel
    dx = coords[:, None, :] - coords[None, :, :]
    u = np.pi * dx / period[None, None, :]
    sin_u = np.sin(u)
    sin2u = np.sin(2.0 * u)
    cos2u = np.cos(2.0 * u)
    sqdist = 2.0 * np.sum(sin_u**2 / ell[None, None, :]**2, axis=2)
    k_base = chi2 * np.exp(-sqdist)

    # Derivative covariance matrix blocks
    k_blocks = np.empty((n_pts, n_pts, 2, 2), dtype=float)
    for a in range(2):
        for b in range(2):
            if a == b:
                k_blocks[:, :, a, b] = k_base * (
                    c_coeff[a] * (2.0 * np.pi / period[a]) * cos2u[..., a]
                    - (c_coeff[a] ** 2) * (sin2u[..., a] ** 2)
                )
            else:
                k_blocks[:, :, a, b] = (
                    -k_base * c_coeff[a] * c_coeff[b] * sin2u[..., a] * sin2u[..., b]
                )

    nd = 2 * n_pts
    k_cov = k_blocks.transpose(0, 2, 1, 3).reshape(nd, nd)

    # Format noise vector
    noise_var = noise**2 if noise_is_std else noise
    if np.isscalar(noise_var) or noise_var.ndim == 0:
        sigma_diag = np.full(nd, float(noise_var), dtype=float)
    else:
        sigma_diag = np.repeat(noise_var.ravel(), 2)

    mean_diag = float(np.mean(np.diag(k_cov))) if k_cov.size else 1.0
    jitter = max(1e-14, jitter_rel * max(mean_diag, 1.0))
    k_cov[np.diag_indices_from(k_cov)] += sigma_diag + jitter

    # Function-derivative cross covariances (Kstar)
    dx_m = coords[:, None, :] - coords[None, :, :]
    u_m = np.pi * dx_m / period[None, None, :]
    sin2u_m = np.sin(2.0 * u_m)
    sqdist_star = 2.0 * np.sum(np.sin(u_m) ** 2 / ell[None, None, :] ** 2, axis=2)
    k_base_star = chi2 * np.exp(-sqdist_star)
    kstar_blocks = -k_base_star[:, :, None] * (sin2u_m * c_coeff[None, None, :])

    kstar = np.empty((nd, n_pts), dtype=float)
    kstar[0::2, :] = kstar_blocks[:, :, 0]
    kstar[1::2, :] = kstar_blocks[:, :, 1]

    # Assemble RHS targets
    fx_all = samples_np[:, 0].reshape(num_samples, n_pts)
    fy_all = samples_np[:, 1].reshape(num_samples, n_pts)
    y_target = np.empty((nd, num_samples), dtype=float)
    y_target[0::2, :] = fx_all.T
    y_target[1::2, :] = fy_all.T

    # Cholesky solve
    try:
        chol_l = la.cholesky(k_cov, lower=True, check_finite=False)
    except la.LinAlgError:
        bump = max(1e-12, 1e-6 * mean_diag)
        chol_l = la.cholesky(k_cov + bump * np.eye(nd), lower=True, check_finite=False)

    tmp = la.solve_triangular(chol_l, y_target, lower=True, check_finite=False)
    alpha_all = la.solve_triangular(chol_l.T, tmp, lower=False, check_finite=False)
    b_mean_all = kstar.T @ alpha_all

    if y_is_force:
        a_samples = -b_mean_all.T.reshape(num_samples, nx, ny)
    else:
        a_samples = b_mean_all.T.reshape(num_samples, nx, ny)

    a_mean = -1.0 * a_samples.mean(axis=0)
    a_std = a_samples.std(axis=0)

    if center == "min":
        offset = a_mean.min()
        a_samples -= offset
        a_mean -= offset

    return a_samples, a_mean, a_std, None


# ---------------------------------------------------------------------------
# Uncertainty-Driven Exploration Bias
# ---------------------------------------------------------------------------

def compute_exploration_push(
    fx_std_grid: np.ndarray,
    fy_std_grid: np.ndarray,
    x_cv_grid: np.ndarray,
    y_cv_grid: np.ndarray,
    job: str | int,
    frac_nonzero: float,
    margin_x: int = 4,
    margin_y: int = 4,
    kb: float = 0.001987,
    temp: float = 300.0,
    ramp_start: int = 5,
    ramp_iters: int = 50,
    frac_start: float = 0.85,
    frac_end: float = 0.95,
) -> tuple:
    """Compute harmonic exploration bias directed toward maximum GP uncertainty."""
    uncertainty_grid = np.sqrt(fx_std_grid**2 + fy_std_grid**2)
    nx, ny = uncertainty_grid.shape

    # Mask boundaries to prevent edge artifacts
    mask = np.zeros_like(uncertainty_grid, dtype=bool)
    mask[margin_x : nx - margin_x, margin_y : ny - margin_y] = True
    masked_uncertainty = np.where(mask, uncertainty_grid, -np.inf)

    flat_idx = np.argmax(masked_uncertainty)
    i_max, j_max = np.unravel_index(flat_idx, masked_uncertainty.shape)
    phi_max = x_cv_grid[i_max, j_max]
    psi_max = y_cv_grid[i_max, j_max]

    k_push = kb * temp * 0.05

    def wrap_delta_deg(val, center):
        return (val - center + 180.0) % 360.0 - 180.0

    dphi = wrap_delta_deg(x_cv_grid, phi_max)
    dpsi = wrap_delta_deg(y_cv_grid, psi_max)

    fx_push_raw = -k_push * dphi
    fy_push_raw = -k_push * dpsi

    # Time-based linear ramp
    iter_idx = int(job)
    if iter_idx <= ramp_start:
        time_ramp = 0.0
    else:
        time_ramp = min(1.0, (iter_idx - ramp_start) / ramp_iters)

    # Coverage-based attenuation
    if frac_nonzero <= frac_start:
        coverage_scale = 1.0
    elif frac_nonzero >= frac_end:
        coverage_scale = 0.0
    else:
        coverage_scale = (frac_end - frac_nonzero) / (frac_end - frac_start)

    exploration_scale = time_ramp * coverage_scale
    fx_push = fx_push_raw * exploration_scale
    fy_push = fy_push_raw * exploration_scale

    u_push = 0.5 * k_push * (np.square(dphi) + np.square(dpsi))
    exploration_bias_energy = exploration_scale * u_push

    return (
        fx_push,
        fy_push,
        exploration_bias_energy,
        u_push,
        phi_max,
        psi_max,
        i_max,
        j_max,
        masked_uncertainty,
        uncertainty_grid,
        exploration_scale,
    )


# ---------------------------------------------------------------------------
# Master Optimization & Execution Routine
# ---------------------------------------------------------------------------

def gp_model2D_gpder(
    log: modular_log.Logger,
    psi_last: float,
    phi_last: float,
    *,
    subfolder_name: str = "alad_wats",
    debug_gp: bool = True,
    save_plots: bool = True,
    num_samples: int = 10,
    train_iters_cold: int = 500,
    train_iters_warm: int = 200,
    use_abf_bias: bool = False,
    use_exploration_bias: bool = True,
) -> None:
    """Train GP on mean forces, reconstruct energy, and output new bias potentials."""
    job = f"{log.iter + 1:04d}"
    grid_shape = log.x_cv_grid.shape
    sim_dir = Path(log.dirtag) / subfolder_name
    data_dir = Path(log.dirtag) / "data"
    sim_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Data Preparation & Subsampling ---
    limit = 100
    dampening = np.minimum(1.0, log.hist / limit)
    xmf_damped = dampening * log.xmf_array_mf
    ymf_damped = dampening * log.ymf_array_mf

    rows, cols = np.nonzero(log.hist)
    rows_sub, cols_sub = smart_subsample(rows, cols, log.hist, max_samples=1200)

    x_sampled_mf = log.xmf_array_mf[rows_sub, cols_sub]
    y_sampled_mf = log.ymf_array_mf[rows_sub, cols_sub]
    xcv_data = log.x_cv_grid[rows_sub, cols_sub]
    ycv_data = log.y_cv_grid[rows_sub, cols_sub]

    frac_nonzero = rows.size / log.hist.size

    x_fx, y_fx = make_training_data(xcv_data, ycv_data, x_sampled_mf)
    x_fy, y_fy = make_training_data(xcv_data, ycv_data, y_sampled_mf)
    train_x, train_y = stack_force_training_data(x_fx, y_fx, x_fy, y_fy, y_is_force=True)

    # =========================================================================
    # STEP 2: MODEL SETUP (COVARIANCE KERNEL & OBSERVATION NOISE MODEL)
    # =========================================================================

    # Domain periodicity (e.g., 360 deg for dihedrals)
    extent_x = np.ptp(log.x_cv_grid)
    extent_y = np.ptp(log.y_cv_grid)

    # -------------------------------------------------------------------------
    # >>> KERNEL SPECIFICATION <<<
    # -------------------------------------------------------------------------
    base_kernel = PeriodicDerivativeKernel2D(ard_num_dims=2)
    with torch.no_grad():
        base_kernel.period_length = torch.tensor([extent_x, extent_y])
    base_kernel.raw_period_length.requires_grad_(False)  # Period is fixed by grid domain

    # Outputscale (signal variance sigma_f^2) is learnable with bounds [0.001, 50.0]
    kernel = gpytorch.kernels.ScaleKernel(
        base_kernel, 
        outputscale_constraint=gpytorch.constraints.Interval(0.001, 50.0)
    )
    kernel = gpytorch.kernels.ScaleKernel(
        base_kernel, outputscale_constraint=gpytorch.constraints.Interval(0.001, 50.0)
    )
    # -------------------------------------------------------------------------
    # >>> NOISE / LIKELIHOOD SPECIFICATION <<<
    # -------------------------------------------------------------------------
    # Gaussian noise on force/gradient observations with positive constraint
    # and Gamma prior on noise variance (concentration=2.0, rate=10.0)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.Positive()
    )
    likelihood.register_prior(
        "noise_prior", 
        gpytorch.priors.GammaPrior(concentration=2.0, rate=10.0), 
        "noise"
    )
    likelihood.noise = 0.01  # Initial noise initialization

    # -------------------------------------------------------------------------
    # >>> MODEL INITIALIZATION <<<
    # -------------------------------------------------------------------------
    model = PotentialGradGP(train_x, train_y, likelihood, kernel)
    ckpt_path = sim_dir / "gp_checkpoint.pth"

    num_iters = train_iters_cold

    # --- Step 3: Optimization Loop ---
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iters, eta_min=1e-4
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(num_iters):
        optimizer.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # --- Step 4: Full Grid Posterior Inference ---
    model.eval()
    likelihood.eval()

    x_test = torch.tensor(
        np.stack([log.x_cv_grid.ravel(), log.y_cv_grid.ravel()], axis=-1),
        dtype=torch.float32,
    )
    batch_size = 1000
    parts = {k: [] for k in ["dx_m", "dx_s", "dy_m", "dy_s", "dx_samp", "dy_samp"]}

    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples():
        for i0 in range(0, len(x_test), batch_size):
            batch = x_test[i0 : i0 + batch_size]
            p_dx = model(make_aug_test(batch, 0.0))
            p_dy = model(make_aug_test(batch, 1.0))

            parts["dx_m"].append(p_dx.mean.cpu())
            parts["dx_s"].append(p_dx.stddev.cpu())
            parts["dy_m"].append(p_dy.mean.cpu())
            parts["dy_s"].append(p_dy.stddev.cpu())
            parts["dx_samp"].append(p_dx.rsample(torch.Size([num_samples])).cpu())
            parts["dy_samp"].append(p_dy.rsample(torch.Size([num_samples])).cpu())

    def _format_grid(key: str, is_samples: bool = False) -> np.ndarray:
        t = torch.cat(parts[key], dim=1 if is_samples else 0).numpy()
        return t.reshape((num_samples, *grid_shape)) if is_samples else t.reshape(grid_shape)

    dadx_samples = _format_grid("dx_samp", is_samples=True)
    dady_samples = _format_grid("dy_samp", is_samples=True)
    fx_mean_grid = -_format_grid("dx_m")
    fy_mean_grid = -_format_grid("dy_m")
    fx_std_grid = _format_grid("dx_s")
    fy_std_grid = _format_grid("dy_s")

    del x_test, parts
    gc.collect()

    # --- Step 5: Exploration Push Force Generation ---
    if use_exploration_bias:
        (
            fx_push,
            fy_push,
            expl_bias_energy,
            _,
            phi_max,
            psi_max,
            _,
            _,
            _,
            _,
            exploration_scale,
        ) = compute_exploration_push(
            fx_std_grid, fy_std_grid, log.x_cv_grid, log.y_cv_grid, job, frac_nonzero
        )
    else:
        fx_push = np.zeros_like(fx_mean_grid)
        fy_push = np.zeros_like(fy_mean_grid)
        expl_bias_energy = np.zeros_like(fx_mean_grid)

    # --- Step 6: Direct Potential Reconstruction ---
    with torch.no_grad():
        ell = model.covar_module.base_kernel.lengthscale.cpu().numpy().ravel()
        chi2 = float(model.covar_module.outputscale.cpu())
        period = model.covar_module.base_kernel.period_length.cpu().numpy().ravel()

    noise_var = 0.5 * (fx_std_grid**2 + fy_std_grid**2)
    samples_phys = np.stack([dadx_samples, dady_samples], axis=1)

    _, a_mean, a_std, _ = reconstruct_A_from_gp_samples_2D_no_norm_fast(
        xv=log.x_cv_grid,
        yv=log.y_cv_grid,
        samples_phys=samples_phys,
        ell=ell,
        chi2=chi2,
        period=period,
        noise=noise_var,
        noise_is_std=False,
        y_is_force=False,
        jitter_rel=1e-10,
        center="min",
    )

    # --- Step 7: Checkpoint & State Serialization ---
    torch.save(
        {"model": model.state_dict(), "likelihood": likelihood.state_dict()}, ckpt_path
    )

    fx_out = xmf_damped if use_abf_bias else (fx_mean_grid + fx_push)
    fy_out = ymf_damped if use_abf_bias else (fy_mean_grid + fy_push)

    x_flat = log.x_cv_grid.ravel()
    y_flat = log.y_cv_grid.ravel()
    grid_header = f"shape: {grid_shape[1]} {grid_shape[0]}"

    np.savetxt(
        sim_dir / "model_data.txt",
        np.column_stack((x_flat, y_flat, fx_out.ravel(), fy_out.ravel())),
        fmt="%.6f",
        header="x_cv y_cv fx_pred fy_pred",
        comments="",
    )

    np.savetxt(
        data_dir / f"data_summary{job}.txt",
        np.column_stack(
            (x_flat, y_flat, a_mean.ravel(), a_std.ravel(), log.hist.ravel())
        ),
        fmt="%.6f",
        header=f"{grid_header} x_cv y_cv A_mean A_std hist",
        comments="",
    )

    with open(data_dir / "hp.txt", "a") as f:
        f.write(f"{ell[0]:.4f} {ell[1]:.4f} {chi2:.4f} {np.mean(a_std):.4f} {frac_nonzero:.4f}\n")

    # --- Step 8: Visualization Generation ---
    if save_plots:
        energy_cmap = LinearSegmentedColormap.from_list(
            "energy_cmap", ["#4E96C4", "#7BCAB9", "#F2F3B8", "#F3B27C", "#EC7A84"]
        )

        # Force field comparisons
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        fields = [
            (axes[0, 0], log.xmf_array_mf, "ABF Fx (raw)"),
            (axes[0, 1], fx_mean_grid, "GP Fx"),
            (axes[1, 0], log.ymf_array_mf, "ABF Fy (raw)"),
            (axes[1, 1], fy_mean_grid, "GP Fy"),
        ]
        for ax, field, title in fields:
            cf = ax.contourf(log.x_cv_grid, log.y_cv_grid, field, levels=30, cmap="coolwarm")
            fig.colorbar(cf, ax=ax)
            ax.set_title(title)
        for ax in (axes[0, 1], axes[1, 1]):
            ax.scatter(xcv_data, ycv_data, c="k", s=4, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{log.dirtag}/gp_results{job}.svg")
        plt.close(fig)

        # Sampling histogram
        rows_new, cols_new = np.nonzero(log.hist_new)
        fig, ax = plt.subplots()
        cf = ax.contourf(log.x_cv_grid, log.y_cv_grid, log.hist, levels=30, cmap="Blues")
        ax.scatter(xcv_data, ycv_data, c="k", s=4, alpha=0.3, label="Prior Samples")
        ax.scatter(
            log.x_cv_grid[rows_new, cols_new],
            log.y_cv_grid[rows_new, cols_new],
            c="r",
            s=4,
            alpha=0.5,
            label="Recent Samples",
        )
        ax.set_title("Sampling Histogram")
        ax.set_aspect("equal")
        cax = inset_axes(ax, width="3%", height="100%", loc="right", borderpad=-1.5)
        fig.colorbar(cf, cax=cax)
        plt.savefig(f"{log.dirtag}/hist{job}.svg")
        plt.close(fig)

        # Free energy surfaces
        fig, ax = plt.subplots(figsize=(6, 5))
        cf = ax.contourf(log.x_cv_grid, log.y_cv_grid, a_mean, levels=25, cmap=energy_cmap)
        ax.contour(log.x_cv_grid, log.y_cv_grid, a_mean, levels=25, colors="k", linewidths=0.6, alpha=0.4)
        fig.colorbar(cf, ax=ax)
        ax.set_title(f"GPR Free Energy (iter {job})")
        ax.set_xlabel("CV1")
        ax.set_ylabel("CV2")
        plt.tight_layout()
        plt.savefig(f"{log.dirtag}/energy_num{job}.svg", dpi=150)
        plt.close(fig)

    del model, likelihood
    gc.collect()