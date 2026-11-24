"""
Free-Energy Surface (FES) Reconstruction via Derivative-Kernel Gaussian Processes.

This module provides:
1. Analytic derivative kernels for 1D and 2D collective variable (CV) spaces:
   - 1D RBF derivative covariance.
   - 2D mixed RBF x Periodic (von Mises-style) derivative covariance.
   - 2D full RBF x RBF derivative covariance.
2. Exact two-stage free energy path integration via kernel algebra:
   - Propagates GPR force predictive variance (Stage 1).
   - Evaluates reconstruction GP covariance from finite derivative data (Stage 2).
3. Uncertainty-driven harmonic exploration biasing (active learning push forces).
"""

from __future__ import annotations

import os
from pathlib import Path
import pickle
from typing import NamedTuple, Optional, Tuple

import gpytorch
from gpytorch.constraints import Interval
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
from matplotlib.colors import LinearSegmentedColormap, LogNorm, PowerNorm
import numpy as np
import scipy.linalg as la
import torch

import gpr  # Shared plotting utilities


# =============================================================================
# SECTION 1 — UTILITIES & ARRAYS
# =============================================================================

def _wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    """Map angles (radians) to [-pi, pi) elementwise."""
    out = np.arctan2(np.sin(angle), np.cos(angle))
    return np.where(out == np.pi, -np.pi, out)


def _align_grid_to_reference(
    name: str, arr: np.ndarray, reference_shape: tuple
) -> Tuple[np.ndarray, bool]:
    """Align a 2D grid to a reference shape, transposing once if needed."""
    arr = np.asarray(arr)
    if arr.shape == reference_shape:
        return arr, False
    if arr.T.shape == reference_shape:
        return arr.T, True
    raise ValueError(
        f"{name}: shape {arr.shape} is incompatible with reference {reference_shape}"
    )


def get_loggers(logger_dir: str | Path = "./loggers") -> list:
    """Load all serialized logger pickle files from logger_dir."""
    loggers = []
    path = Path(logger_dir)
    if path.exists():
        for f in path.glob("logger_*.pkl"):
            with open(f, "rb") as fh:
                loggers.append(pickle.load(fh))
    return loggers


def save_abf_state(big_log, iteration: int, save_dir: str | Path = "./data") -> None:
    """Save raw ABF accumulator arrays for checkpointing and exact restart."""
    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_kwargs = dict(
        histogram=np.asarray(big_log.histogram),
        xFsum=np.asarray(big_log.xFsum),
        iteration=np.array(iteration),
    )
    if hasattr(big_log, "yFsum"):
        save_kwargs["yFsum"] = np.asarray(big_log.yFsum)

    np.savez(out_dir / "abf_state.npz", **save_kwargs)
    print(f"[save_abf_state] Checkpoint abf_state.npz saved at iteration {iteration}")


# =============================================================================
# SECTION 2 — [KERNEL DEFINITIONS] 2D DERIVATIVE COVARIANCE MODULES
# =============================================================================

class RBFxPeriodicDerivativeKernel2D(gpytorch.kernels.Kernel):
    r"""
    Derivative covariance kernel for A(R, phi) with product structure:
        k(R, phi; R', phi') = k_rbf(R, R') * k_per(phi, phi')

    where:
        k_rbf = exp( -0.5 * ((R - R') / \ell_R)^2 )
        k_per = exp( -2.0 * sin^2((phi - phi') / 2) / \ell_phi^2 )   (period = 2*pi)

    Models Cov( dA/dx_a(x1), dA/dx_b(x2) ).
    Input format: [R, phi, comp] with comp=0 -> d/dR, comp=1 -> d/dphi.
    """

    has_lengthscale = True

    def forward(self, x1, x2, diag: bool = False, **params):
        X1 = x1[..., :2]
        X2 = x2[..., :2]
        c1 = x1[..., 2].long()
        c2 = x2[..., 2].long()

        ell = self.lengthscale.squeeze(0)
        ell_R = ell[0].to(X1.dtype)
        ell_phi = ell[1].to(X1.dtype)
        qx = 1.0 / (ell_R ** 2)
        beta = 1.0 / (ell_phi ** 2)

        if diag:
            return torch.where(
                c1 == 0,
                qx.expand_as(c1).to(X1.dtype),
                beta.expand_as(c1).to(X1.dtype),
            )

        dR = X1[..., 0].unsqueeze(-1) - X2[..., 0].unsqueeze(-2)
        dphi = X1[..., 1].unsqueeze(-1) - X2[..., 1].unsqueeze(-2)

        kx = torch.exp(-0.5 * (dR ** 2) * qx)
        s = torch.sin(0.5 * dphi)
        kphi = torch.exp(-2.0 * (s ** 2) * beta)

        sin_d = torch.sin(dphi)
        cos_d = torch.cos(dphi)

        # First and second derivatives of RBF kernel
        dkx_u = -(dR * qx) * kx
        dkx_v = (dR * qx) * kx
        d2kx = (qx - (dR ** 2) * (qx ** 2)) * kx

        # First and second derivatives of Periodic kernel
        dkphi_u = -(beta * sin_d) * kphi
        dkphi_v = (beta * sin_d) * kphi
        d2kphi = (beta * cos_d - (beta ** 2) * (sin_d ** 2)) * kphi

        # Product mixed derivative blocks
        K_xx = d2kx * kphi
        K_pp = kx * d2kphi
        K_xp = dkx_u * dkphi_v
        K_px = dkphi_u * dkx_v

        n1, n2 = K_xx.shape
        c1m = c1.view(-1, 1).expand(n1, n2)
        c2m = c2.view(1, -1).expand(n1, n2)

        out = torch.empty_like(K_xx)
        out[(c1m == 0) & (c2m == 0)] = K_xx[(c1m == 0) & (c2m == 0)]
        out[(c1m == 1) & (c2m == 1)] = K_pp[(c1m == 1) & (c2m == 1)]
        out[(c1m == 0) & (c2m == 1)] = K_xp[(c1m == 0) & (c2m == 1)]
        out[(c1m == 1) & (c2m == 0)] = K_px[(c1m == 1) & (c2m == 0)]
        return out


class RBFDerivativeKernel2D(gpytorch.kernels.Kernel):
    r"""
    Derivative covariance kernel for A(x1, x2) with 2D non-periodic RBF product structure:
        k(x1, x2; x1', x2') = k_rbf_1(x1, x1') * k_rbf_2(x2, x2')

    Models Cov( dA/dx_a(x), dA/dx_b(x') ).
    Input format: [x1, x2, comp] where comp=0 -> d/dx1, comp=1 -> d/dx2.
    """

    has_lengthscale = True

    def forward(self, x1, x2, diag: bool = False, **params):
        X1 = x1[..., :2]
        X2 = x2[..., :2]
        c1 = x1[..., 2].long()
        c2 = x2[..., 2].long()

        ell = self.lengthscale.squeeze(0)
        ell_1 = ell[0].to(X1.dtype)
        ell_2 = ell[1].to(X1.dtype)
        q1 = 1.0 / (ell_1 ** 2)
        q2 = 1.0 / (ell_2 ** 2)

        if diag:
            return torch.where(
                c1 == 0,
                q1.expand_as(c1).to(X1.dtype),
                q2.expand_as(c1).to(X1.dtype),
            )

        d1 = X1[..., 0].unsqueeze(-1) - X2[..., 0].unsqueeze(-2)
        d2 = X1[..., 1].unsqueeze(-1) - X2[..., 1].unsqueeze(-2)

        k1 = torch.exp(-0.5 * (d1 ** 2) * q1)
        k2 = torch.exp(-0.5 * (d2 ** 2) * q2)

        dk1_u = -(d1 * q1) * k1
        dk1_v = (d1 * q1) * k1
        dk2_u = -(d2 * q2) * k2
        dk2_v = (d2 * q2) * k2

        d2k1 = (q1 - (d1 ** 2) * (q1 ** 2)) * k1
        d2k2 = (q2 - (d2 ** 2) * (q2 ** 2)) * k2

        K_11 = d2k1 * k2
        K_22 = k1 * d2k2
        K_12 = dk1_u * dk2_v
        K_21 = dk2_u * dk1_v

        n1, n2 = K_11.shape
        c1m = c1.view(-1, 1).expand(n1, n2)
        c2m = c2.view(1, -1).expand(n1, n2)

        out = torch.empty_like(K_11)
        out[(c1m == 0) & (c2m == 0)] = K_11[(c1m == 0) & (c2m == 0)]
        out[(c1m == 1) & (c2m == 1)] = K_22[(c1m == 1) & (c2m == 1)]
        out[(c1m == 0) & (c2m == 1)] = K_12[(c1m == 0) & (c2m == 1)]
        out[(c1m == 1) & (c2m == 0)] = K_21[(c1m == 1) & (c2m == 0)]
        return out


# =============================================================================
# SECTION 3 — [GP MODELS] 1D & 2D FORCE PREDICTION
# =============================================================================

class DerivativeGPModel1D(gpytorch.models.ExactGP):
    """1D GP model encoding the analytic second-derivative RBF kernel directly."""

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.base_kernel = gpytorch.kernels.RBFKernel(
            lengthscale_constraint=Interval(0.02, 2.0)
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x):
        x1d = x.squeeze(-1)
        n = x1d.shape[0]
        ell = self.base_kernel.lengthscale.squeeze()
        sigma2 = self.covar_module.outputscale

        diff = x1d.unsqueeze(1) - x1d.unsqueeze(0)
        k_rbf = sigma2 * torch.exp(-0.5 * (diff / ell) ** 2)
        k_dd = k_rbf * (1.0 / (ell ** 2) - (diff ** 2) / (ell ** 4))
        k_dd = 0.5 * (k_dd + k_dd.t()) + 1e-6 * torch.eye(n, dtype=x.dtype, device=x.device)

        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), k_dd)


class PotentialGradGP(gpytorch.models.ExactGP):
    """2D GP model for predicting potential gradients (forces)."""

    def __init__(self, train_x, train_y, likelihood, kernel, mean_module=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean_module or gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


# =============================================================================
# SECTION 4 — TRAINING TENSOR GENERATION
# =============================================================================

def _make_training_data_2d(xcv: np.ndarray, ycv: np.ndarray, forces: np.ndarray):
    """Convert grid coordinates and forces into 2D torch tensors."""
    x_np = np.stack([xcv.ravel(), ycv.ravel()], axis=-1).astype(np.float32)
    y_np = forces.ravel().astype(np.float32)
    return torch.tensor(x_np), torch.tensor(y_np)


def _stack_force_training_data(x_fx: torch.Tensor, y_fx: torch.Tensor, x_fy: torch.Tensor, y_fy: torch.Tensor):
    """Combine Fx and Fy training sets into [x1, x2, comp] representation.

    Transforms applied forces F to potential gradients: dA/dq = -F.
    """
    device, dtype = x_fx.device, x_fx.dtype
    comp_x = torch.zeros(x_fx.shape[0], 1, device=device, dtype=dtype)
    comp_y = torch.ones(x_fy.shape[0], 1, device=device, dtype=dtype)

    train_x = torch.cat(
        [torch.cat([x_fx, comp_x], dim=1), torch.cat([x_fy, comp_y], dim=1)], dim=0
    )
    train_y = torch.cat([-y_fx, -y_fy], dim=0)
    return train_x, train_y


def _make_aug_test(x_test_xy: torch.Tensor, comp_value: float) -> torch.Tensor:
    """Append component flag (0: dA/dx, 1: dA/dy) to evaluation coordinates."""
    comp = torch.full(
        (x_test_xy.shape[0], 1), comp_value, dtype=x_test_xy.dtype, device=x_test_xy.device
    )
    return torch.cat([x_test_xy, comp], dim=1)


# =============================================================================
# SECTION 5 — ENERGY RECONSTRUCTION & TWO-STAGE UNCERTAINTY PROPAGATION
# =============================================================================

class EnergyReconstructionResult(NamedTuple):
    """Container for reconstructed free energy and decomposed uncertainty sources."""
    A_samples: np.ndarray
    A_mean: np.ndarray
    force_uncertainty: np.ndarray           # Stage 1: GPR force prediction std
    reconstruction_uncertainty: Optional[np.ndarray]  # Stage 2: GP path integration std
    total_uncertainty: Optional[np.ndarray]           # Quadrature sum: sqrt(Stage1^2 + Stage2^2)


def reconstruct_energy_from_forces(
    xv: np.ndarray,
    yv: np.ndarray | None = None,
    samples_phys: np.ndarray | None = None,
    ell: float | np.ndarray | None = None,
    chi2: float | None = None,
    noise: float | np.ndarray | None = None,
    noise_is_std: bool = False,
    jitter_rel: float = 1e-10,
    center: str = "min",
    compute_variance: bool = True,
    cv1_name: str = "dist",
    cv2_name: str = "dihedral",
) -> EnergyReconstructionResult:
    """Reconstruct potential A(x) from GPR force samples across 1D or 2D CV space."""
    xv = np.asarray(xv, dtype=float)
    is_2d = yv is not None

    if is_2d:
        yv = np.asarray(yv, dtype=float)
        nx, ny = xv.shape
        n_pts = nx * ny
        grid_shape = (nx, ny)
    else:
        n_pts = xv.shape[0]
        grid_shape = (n_pts,)

    samples_np = np.asarray(samples_phys, dtype=float)
    if is_2d:
        s_count, c_count, nx_s, ny_s = samples_np.shape
        assert c_count == 2 and nx_s == nx and ny_s == ny
    else:
        if samples_np.ndim == 3:
            samples_np = samples_np[:, 0, :]
        s_count, n_s = samples_np.shape
        assert n_s == n_pts

    if chi2 is None or chi2 <= 0.0:
        raise ValueError("chi2 must be a positive float (energy kernel outputscale in eV^2).")

    ell = np.asarray(ell, dtype=float).ravel()
    ell_1 = float(ell[0])
    ell_2 = float(ell[1]) if is_2d else None

    def _kernel_terms(diff, ell_val, cv_type):
        if cv_type == "dihedral":
            beta = 1.0 / (ell_val ** 2)
            s = np.sin(0.5 * diff)
            k = np.exp(-2.0 * (s ** 2) * beta)
            sin_d = np.sin(diff)
            cos_d = np.cos(diff)
            dk_u = -(beta * sin_d) * k
            dk_v = (beta * sin_d) * k
            d2k = (beta * cos_d - (beta ** 2) * (sin_d ** 2)) * k
        else:
            q = 1.0 / (ell_val ** 2)
            k = np.exp(-0.5 * (diff ** 2) * q)
            dk_u = -(diff * q) * k
            dk_v = (diff * q) * k
            d2k = (q - (diff ** 2) * (q ** 2)) * k
        return k, dk_u, dk_v, d2k

    if is_2d:
        coords = np.column_stack([xv.ravel(), yv.ravel()])
        d1 = coords[:, None, 0] - coords[None, :, 0]
        d2 = coords[:, None, 1] - coords[None, :, 1]

        k1, dk1_u, dk1_v, d2k1 = _kernel_terms(d1, ell_1, cv1_name)
        k2, dk2_u, dk2_v, d2k2 = _kernel_terms(d2, ell_2, cv2_name)

        k_blocks = np.empty((n_pts, n_pts, 2, 2), dtype=float)
        k_blocks[:, :, 0, 0] = chi2 * d2k1 * k2
        k_blocks[:, :, 1, 1] = chi2 * k1 * d2k2
        k_blocks[:, :, 0, 1] = chi2 * dk1_u * dk2_v
        k_blocks[:, :, 1, 0] = chi2 * dk2_u * dk1_v

        nd = 2 * n_pts
        k_mat = k_blocks.transpose(0, 2, 1, 3).reshape(nd, nd)

        kstar = np.empty((nd, n_pts), dtype=float)
        kstar[0::2, :] = chi2 * dk1_u * k2
        kstar[1::2, :] = chi2 * k1 * dk2_u
    else:
        d1 = xv[:, None] - xv[None, :]
        _, dk1_u, _, d2k1 = _kernel_terms(d1, ell_1, cv1_name)
        nd = n_pts
        k_mat = chi2 * d2k1
        kstar = chi2 * dk1_u

    # Apply diagonal noise regularization
    noise_var = noise ** 2 if noise_is_std else noise
    if noise_var is None:
        sigma_diag = np.zeros(nd, dtype=float)
    elif np.isscalar(noise_var) or np.asarray(noise_var).ndim == 0:
        sigma_diag = np.full(nd, float(noise_var))
    elif is_2d:
        sigma_diag = np.empty(nd, dtype=float)
        s_arr = np.asarray(noise_var).reshape(n_pts, 2)
        sigma_diag[0::2] = s_arr[:, 0]
        sigma_diag[1::2] = s_arr[:, 1]
    else:
        sigma_diag = np.asarray(noise_var).ravel()

    np.fill_diagonal(k_mat, np.diag(k_mat) + sigma_diag)
    mean_diag = float(np.mean(np.diag(k_mat)))
    jitter = max(1e-14, jitter_rel * max(mean_diag, 1.0))
    np.fill_diagonal(k_mat, np.diag(k_mat) + jitter)

    # Assemble RHS force sample matrix
    if is_2d:
        fx_all = samples_np[:, 0].reshape(s_count, n_pts)
        fy_all = samples_np[:, 1].reshape(s_count, n_pts)
        y_mat = np.empty((nd, s_count), dtype=float)
        y_mat[0::2, :] = fx_all.T
        y_mat[1::2, :] = fy_all.T
    else:
        y_mat = samples_np.reshape(s_count, n_pts).T

    try:
        chol_l = la.cholesky(k_mat, lower=True, check_finite=False)
    except la.LinAlgError:
        bump = max(1e-12, 1e-6 * mean_diag)
        np.fill_diagonal(k_mat, np.diag(k_mat) + bump)
        chol_l = la.cholesky(k_mat, lower=True, check_finite=False)

    tmp = la.solve_triangular(chol_l, y_mat, lower=True, check_finite=False)
    alpha = la.solve_triangular(chol_l.T, tmp, lower=False, check_finite=False)
    b_all = kstar.T @ alpha

    # Recover potential (F = -dA/dq -> A = -B)
    a_samples = (-b_all).T.reshape((s_count,) + grid_shape)
    a_mean = a_samples.mean(axis=0)
    force_uncertainty = a_samples.std(axis=0)

    recon_uncertainty = None
    total_uncertainty = None

    if compute_variance:
        v = la.solve_triangular(chol_l, kstar, lower=True, check_finite=False)
        recon_var_1d = np.maximum(chi2 - np.einsum("ij,ij->j", v, v), 0.0)
        recon_var = recon_var_1d.reshape(grid_shape)
        recon_uncertainty = np.sqrt(recon_var)
        total_uncertainty = np.sqrt(force_uncertainty ** 2 + recon_var)

    if center == "min":
        offset = float(a_mean.min())
        a_mean -= offset
        a_samples -= offset

    return EnergyReconstructionResult(
        A_samples=a_samples,
        A_mean=a_mean,
        force_uncertainty=force_uncertainty,
        reconstruction_uncertainty=recon_uncertainty,
        total_uncertainty=total_uncertainty,
    )


# =============================================================================
# SECTION 6 — HARMONIC EXPLORATION BIAS FIELD
# =============================================================================

def _compute_exploration_push_2d(
    a_var: np.ndarray,
    x_cv_grid: np.ndarray,
    y_cv_grid: np.ndarray,
    iteration: int,
    frac_nonzero: float,
    push_coeff: float = 1.0,
    cv1_name: str = "dist",
    cv2_name: str = "dihedral",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Generate 2D harmonic exploration bias targeting the maximum uncertainty cell."""
    x_cv_grid = np.asarray(x_cv_grid)
    y_cv_grid = np.asarray(y_cv_grid)
    uncertainty = np.asarray(a_var)

    if uncertainty.shape != x_cv_grid.shape:
        if uncertainty.T.shape == x_cv_grid.shape:
            uncertainty = uncertainty.T
        else:
            raise ValueError(f"Uncertainty grid shape {uncertainty.shape} incompatible with CV grid {x_cv_grid.shape}")

    nx_u, ny_u = uncertainty.shape
    margin_x = max(1, int(nx_u * 0.10))
    margin_y = max(1, int(ny_u * 0.10))
    mask = np.zeros_like(uncertainty, dtype=bool)
    mask[margin_x: nx_u - margin_x, margin_y: ny_u - margin_y] = True
    masked = np.where(mask, uncertainty, -np.inf)

    flat_idx = int(np.argmax(masked))
    x_target = float(x_cv_grid.ravel()[flat_idx])
    y_target = float(y_cv_grid.ravel()[flat_idx])

    # Dynamic scaling: ramp in time, decay with grid coverage
    ramp_start, ramp_iters = 2, 50
    time_ramp = 0.0 if iteration <= ramp_start else min((iteration - ramp_start) / ramp_iters, 1.0)

    frac_start, frac_end = 0.85, 0.95
    if frac_nonzero <= frac_start:
        coverage_scale = 1.0
    elif frac_nonzero >= frac_end:
        coverage_scale = 0.0
    else:
        coverage_scale = (frac_end - frac_nonzero) / (frac_end - frac_start)

    exploration_scale = time_ramp * coverage_scale * push_coeff

    kb, temp = 8.62e-5, 300.0
    k_push = push_coeff * kb * temp

    d_x = x_cv_grid - x_target
    d_y = y_cv_grid - y_target

    if cv1_name == "dihedral":
        d_x = _wrap_to_pi(d_x)
    if cv2_name == "dihedral":
        d_y = _wrap_to_pi(d_y)

    fx_push = -k_push * d_x * exploration_scale
    fy_push = -k_push * d_y * exploration_scale
    u_push = (0.5 * k_push * (d_x ** 2) + 0.5 * k_push * (d_y ** 2)) * exploration_scale

    return fx_push, fy_push, u_push, x_target, y_target, exploration_scale


# =============================================================================
# SECTION 7 — TRAINING OPTIMIZATION LOOP
# =============================================================================

def _train_model(model, likelihood, train_x, train_y, num_iters: int = 500) -> float:
    """Optimize GP marginal likelihood using Adam with cosine annealing."""
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(
        [
            {"params": model.covar_module.parameters(), "lr": 0.01},
            {"params": model.mean_module.parameters(), "lr": 0.01},
            {"params": likelihood.parameters(), "lr": 0.001},
        ],
        lr=0.01,
    )
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters)

    for i in range(num_iters):
        optimizer.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

    return loss.item()


# =============================================================================
# SECTION 8 — 2D MASTER ROUTINE
# =============================================================================

def gpr_pytorch_2d(
    iteration: int,
    big_log,
    cv1_name: str = "dist",
    cv2_name: str = "dihedral",
    abf_only: bool = False,
) -> None:
    """Train 2D derivative GP, reconstruct potential, and write bias force table."""
    limit = 100
    num_iters = 500
    num_samples = 10
    plot_dir = Path("./figs/gpr_plots")
    data_dir = Path("./data")
    plot_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    cv = big_log.cv
    counts = np.asarray(big_log.histogram, dtype=float)
    mask_nz = counts > 5

    x_fsum = np.asarray(big_log.xFsum, dtype=float)
    y_fsum = np.asarray(big_log.yFsum, dtype=float)

    # Compute damped raw ABF mean forces
    x_abf = np.where(counts > 0, x_fsum / counts, np.nan)
    y_abf = np.where(counts > 0, y_fsum / counts, np.nan)
    dampening = np.minimum(1.0, counts / limit)
    x_damped = dampening * x_abf
    y_damped = dampening * y_abf

    xcv_data = cv[0].T[mask_nz]
    ycv_data = cv[1].T[mask_nz]
    x_target_mf = x_damped[mask_nz]
    y_target_mf = y_damped[mask_nz]

    frac_nonzero = np.count_nonzero(counts) / max(counts.size, 1)

    # 1. Build training tensors
    x_fx, y_fx = _make_training_data_2d(xcv_data, ycv_data, x_target_mf)
    x_fy, y_fy = _make_training_data_2d(xcv_data, ycv_data, y_target_mf)
    train_x, train_y = _stack_force_training_data(x_fx, y_fx, x_fy, y_fy)

    # 2. Configure Derivative Kernel
    # =========================================================================
    # >>> [KERNEL INSTANTIATION & BOUNDS] <<<
    # =========================================================================
    def ls_upper(name):
        return 0.638 if name == "dihedral" else 0.4

    has_periodic = (cv1_name == "dihedral") or (cv2_name == "dihedral")
    if has_periodic:
        base_kernel = RBFxPeriodicDerivativeKernel2D(
            ard_num_dims=2,
            lengthscale_constraint=Interval(
                torch.tensor([0.1, 0.1]),
                torch.tensor([ls_upper(cv1_name), ls_upper(cv2_name)]),
            ),
        )
    else:
        base_kernel = RBFDerivativeKernel2D(
            ard_num_dims=2,
            lengthscale_constraint=Interval(
                torch.tensor([0.1, 0.1]),
                torch.tensor([ls_upper(cv1_name), ls_upper(cv2_name)]),
            ),
        )

    kernel = gpytorch.kernels.ScaleKernel(
        base_kernel, outputscale_constraint=Interval(0.001, 10.0)
    )

    # =========================================================================
    # >>> [LIKELIHOOD / NOISE MODEL] <<<
    # =========================================================================
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = PotentialGradGP(train_x, train_y, likelihood, kernel)

    # 3. Model Optimization
    _train_model(model, likelihood, train_x, train_y, num_iters=num_iters)

    with torch.no_grad():
        ell_opt = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
        chi2_opt = float(model.covar_module.outputscale.detach().cpu())

    # 4. Grid Posterior Inference
    model.eval()
    likelihood.eval()
    grid_shape = cv[0].shape

    x_test_np = np.stack([cv[0].ravel(), cv[1].ravel()], axis=-1).astype(np.float32)
    x_test_xy = torch.tensor(x_test_np)

    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples():
        post_d1 = model(_make_aug_test(x_test_xy, 0.0))
        post_d2 = model(_make_aug_test(x_test_xy, 1.0))

        fx_mean_grid = (-post_d1.mean).cpu().numpy().reshape(grid_shape)
        fy_mean_grid = (-post_d2.mean).cpu().numpy().reshape(grid_shape)
        fx_std_grid = post_d1.stddev.cpu().numpy().reshape(grid_shape)
        fy_std_grid = post_d2.stddev.cpu().numpy().reshape(grid_shape)

        d1_samples = post_d1.rsample(torch.Size([num_samples])).cpu().numpy().reshape(num_samples, *grid_shape)
        d2_samples = post_d2.rsample(torch.Size([num_samples])).cpu().numpy().reshape(num_samples, *grid_shape)

    # 5. Potential Reconstruction
    noise_for_recon = np.stack([fx_std_grid, fy_std_grid], axis=-1)
    samples_phys = np.stack([-d1_samples, -d2_samples], axis=1)

    result = reconstruct_energy_from_forces(
        xv=cv[0],
        yv=cv[1],
        samples_phys=samples_phys,
        ell=ell_opt,
        chi2=chi2_opt,
        noise=noise_for_recon,
        noise_is_std=True,
        center="min",
        compute_variance=True,
        cv1_name=cv1_name,
        cv2_name=cv2_name,
    )

    # 6. Active Exploration Push
    fx_push, fy_push, u_push, x_tgt, y_tgt, _ = _compute_exploration_push_2d(
        result.total_uncertainty ** 2,
        cv[0],
        cv[1],
        iteration,
        frac_nonzero,
        cv1_name=cv1_name,
        cv2_name=cv2_name,
    )

    # 7. Grid Alignment & Serialization
    ref_shape = cv[0].shape
    fx_mean_al, _ = _align_grid_to_reference("fx_mean", fx_mean_grid, ref_shape)
    fy_mean_al, _ = _align_grid_to_reference("fy_mean", fy_mean_grid, ref_shape)
    fx_push_al, _ = _align_grid_to_reference("fx_push", fx_push, ref_shape)
    fy_push_al, _ = _align_grid_to_reference("fy_push", fy_push, ref_shape)

    fx_total = fx_mean_al + fx_push_al
    fy_total = fy_mean_al + fy_push_al

    # Export model_data.txt for simulation engine bias coupling
    x_flat = cv[0].ravel()
    y_flat = cv[1].ravel()
    fx_out = x_damped.ravel() if abf_only else fx_total.ravel()
    fy_out = y_damped.ravel() if abf_only else fy_total.ravel()

    np.savetxt(
        "model_data.txt",
        np.column_stack((x_flat, y_flat, fx_out, fy_out)),
        fmt="%.6f",
        comments="",
    )

    # Save summary logs and model checkpoint
    torch.save({"model": model.state_dict(), "likelihood": likelihood.state_dict()}, data_dir / "gp_checkpoint_2d.pth")
    save_abf_state(big_log, iteration, save_dir=data_dir)