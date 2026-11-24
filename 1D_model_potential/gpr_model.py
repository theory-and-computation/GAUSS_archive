"""gpr_model.py — GP force model, free-energy reconstruction, and diagnostics.

═══════════════════════════════════════════════════════════════════════════
 WHERE THE MODEL COMPONENTS LIVE
═══════════════════════════════════════════════════════════════════════════
 KERNEL (prior over the force)
   [KERNEL 1/3] DerivativeGPModel1D.__init__
                Base kernel: RBF + Scale wrapper. Learned hyperparameters
                are the lengthscale (ell) and outputscale (sigma^2).
   [KERNEL 2/3] DerivativeGPModel1D.forward
                Closed-form force kernel
                    K_f(d) = (1/ell^2 - d^2/ell^4) * sigma^2 * exp(-d^2/2ell^2)
                = Cov[dA/dx, dA/dx'] for an RBF latent potential A(x).
                The force is observed as -dA/dx.
   [KERNEL 3/3] reconstruct_A_from_gp_mean_1D
                Analytic force/energy kernel matrices (K_dd, Kstar, K_ff)
                used to integrate the force posterior back into an energy
                (free energy) posterior.

 NOISE (observation model for the force data)
   [NOISE 1/3] gp_model1D_simple
               gpytorch.likelihoods.GaussianLikelihood(); when
               autocorrelation_correction is OFF the noise is LEARNED by
               the optimizer together with the kernel hyperparameters.
   [NOISE 3/3] gp_model1D_simple (end of function)
               The effective noise per iteration is logged to noise.txt.

 Note: the small diagonal "jitter" terms inside the GP/reconstruction code
 are numerical stabilizers only — they are NOT the observation noise model.
═══════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import json
import os
import time

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la
import torch
from gpytorch.constraints import Interval
from gaussian_potential import gaussian_potential_1D

# Compute device — GP training and inference all run on it.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directory for summaries, samples, logs, and plots.
# Change DATA_DIR_NAME to store outputs elsewhere.
DATA_DIR_NAME = "0_1D_model_potential_data"
DATA_DIR = os.path.join(os.path.dirname(__file__), DATA_DIR_NAME)
GAUSS_LOG_FILE = os.path.join(DATA_DIR, "gauss_log.txt")


# ---------------------------------------------------------------------------
# Timing log (JSON lines)
# ---------------------------------------------------------------------------
def reset_timing_log():
    """Start a fresh JSON-lines timing log for the current simulation."""
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(GAUSS_LOG_FILE, "w") as f:
        f.write(json.dumps({"phase": "run", "event": "start"}) + "\n")


def log_timing(phase, iteration, elapsed_seconds, **details):
    """Append one timing record and flush it immediately."""
    record = {
        "phase": phase,
        "iteration": int(iteration),
        "elapsed_seconds": float(elapsed_seconds),
        **details,
    }
    with open(GAUSS_LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# GP model
# ---------------------------------------------------------------------------
class DerivativeGPModel1D(gpytorch.models.ExactGP):
    """1D GP whose prior is defined directly on the force (-dA/dx).

    The latent potential A(x) has an RBF prior (ell, sigma^2); the force is
    (minus) its first derivative, so the force prior uses the analytic
    force kernel K_f computed in ``forward``. Observations are the per-bin
    ABF mean forces; the observation noise model is set up in
    ``gp_model1D_simple`` (see the [NOISE x/3] markers there).
    """

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        # ══════════════════════════════════════════════════════════════════
        # [KERNEL 1/3] Base kernel + learned hyperparameters
        #   RBF base kernel with a constrained lengthscale, wrapped in a
        #   Scale kernel (learned outputscale sigma^2). These two parameters
        #   fully define the prior covariance; they are trained via the
        #   marginal log-likelihood in gp_model1D_simple.
        # ══════════════════════════════════════════════════════════════════
        self.base_kernel = gpytorch.kernels.RBFKernel(
            lengthscale_constraint=Interval(0.02, 2.0)
        )
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x):
        # ══════════════════════════════════════════════════════════════════
        # [KERNEL 2/3] Analytic force kernel (closed form, no autograd)
        #   For an RBF latent potential K(d) = sigma^2 * exp(-d^2 / 2 ell^2):
        #       Cov[dA/dx, dA/dx'] = -d^2K/dd^2 = (1/ell^2 - d^2/ell^4) * K(d)
        #   so the force prior covariance is
        #       K_f(d) = sigma^2 * (1/ell^2 - d^2/ell^4) * exp(-d^2 / 2 ell^2)
        #   The 1e-6 diagonal jitter below is numerical only (not the
        #   observation noise model).
        # ══════════════════════════════════════════════════════════════════
        x_1d = x.squeeze(-1)
        N = x_1d.shape[0]
        ell = self.base_kernel.lengthscale.squeeze()
        sigma2 = self.covar_module.outputscale

        diff = x_1d.unsqueeze(1) - x_1d.unsqueeze(0)
        K_rbf = sigma2 * torch.exp(-0.5 * (diff / ell) ** 2)
        K_f = K_rbf * (1.0 / ell**2 - diff**2 / ell**4)
        K_f = 0.5 * (K_f + K_f.t())  # enforce exact symmetry
        K_f = K_f + 1e-6 * torch.eye(N, dtype=x.dtype, device=x.device)

        mean_x = self.mean_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, K_f)


# ---------------------------------------------------------------------------
# Analytic reference potential
# ---------------------------------------------------------------------------
def sum_gaus1D(x):
    """Analytic reference (energy, force) of the double-Gaussian model potential.

    Thin wrapper around ``gaussian_potential_1D`` so the potential parameters
    and energy scale live in exactly one place (gaussian_potential.py).
    """
    return gaussian_potential_1D(x)


# ---------------------------------------------------------------------------
# Training-data helper
# ---------------------------------------------------------------------------
def make_training_data(x_grid, f_grid):
    """Convert numpy arrays to training tensors (1D only)."""
    X = torch.tensor(np.asarray(x_grid).reshape(-1, 1),
                     dtype=torch.float32, device=device)
    y = torch.tensor(np.asarray(f_grid).ravel(),
                     dtype=torch.float32, device=device)
    return X, y


# ---------------------------------------------------------------------------
# 1D energy reconstruction (posterior-based, heteroscedastic, with threshold)
# ---------------------------------------------------------------------------
def reconstruct_A_from_gp_mean_1D(
    xv, fx_mean, ell, chi2, noise_per_point,
    num_samples=25, jitter_rel=1e-10, variance_threshold=None
):
    """
    Reconstruct free energy A(x) from the force GP's posterior mean,
    with per-point heteroscedastic noise from the force GP's variance.

    Only force predictions where noise_per_point < variance_threshold
    are used as observations.  Remaining points are pure prediction
    targets where the reconstruction GP falls back to its prior.

    Parameters
    ----------
    xv                 : (N,)  coordinate grid (all points — prediction targets)
    fx_mean            : (N,)  posterior mean forces from force GP (= -dA/dx)
    ell                : float GP lengthscale
    chi2               : float GP outputscale (signal variance)
    noise_per_point    : (N,)  force GP posterior variance at each grid point
    num_samples        : int   number of posterior energy samples to draw
    jitter_rel         : float relative jitter for numerical stability
    variance_threshold : float or None
        Drop points with noise_per_point >= threshold.
        None -> auto: 90% of the derivative-kernel prior variance (chi2/ell^2).

    Returns
    -------
    A_mean          : (N,)   posterior mean free energy (shifted so min == 0)
    A_posterior_std : (N,)   posterior std of A at each grid point
    A_samples       : (S, N) posterior samples of A (centred on A_mean)
    A_trap          : (N,)   trapezoidal sanity check from fx_mean
    obs_mask        : (N,)   boolean mask of which grid points were used
    """
    xv = np.asarray(xv, dtype=float).ravel()
    fx_mean = np.asarray(fx_mean, dtype=float).ravel()
    noise_per_point = np.asarray(noise_per_point, dtype=float).ravel()
    N = xv.shape[0]

    ell = float(np.asarray(ell).ravel()[0])
    if chi2 is None or chi2 <= 1e-12:
        chi2 = max(1e-3, np.var(fx_mean))

    # ---- threshold: separate observations from prediction-only points ----
    if variance_threshold is None:
        variance_threshold = 0.9 * chi2 / ell**2

    obs_mask = noise_per_point < variance_threshold
    N_obs = np.count_nonzero(obs_mask)

    if N_obs < 2:
        # Not enough informative points — return flat prior
        A_mean = np.zeros(N)
        A_posterior_std = np.full(N, np.sqrt(chi2))
        rng = np.random.default_rng()
        A_samples = rng.standard_normal((num_samples, N)) * np.sqrt(chi2)
        A_trap = np.zeros(N)
        return A_mean, A_posterior_std, A_samples, A_trap, obs_mask

    x_obs = xv[obs_mask]                        # (N_obs,)
    f_obs = fx_mean[obs_mask]                   # (N_obs,)
    noise_obs = noise_per_point[obs_mask]       # (N_obs,)
    x_test = xv                                 # (N,) — predict everywhere

    # ══════════════════════════════════════════════════════════════════════
    # [KERNEL 3/3] Analytic kernel matrices for the energy reconstruction
    #   With force f = -dA/dx and the RBF latent potential of [KERNEL 2/3]:
    #     K_dd    force-force covariance at observation points (N_obs x N_obs)
    #             (1/ell^2 - d^2/ell^4) * chi2 * exp(-d^2/2ell^2)   (same as K_f)
    #     Kstar   force-energy cross covariance   (N_obs x N)
    #             -(d/ell^2) * chi2 * exp(-d^2/2ell^2)
    #     K_ff    energy-energy prior covariance  (N x N)
    #             chi2 * exp(-d^2/2ell^2)
    #   NOTE: the "noise" added to the K_dd diagonal below is the force GP
    #   posterior variance per grid point (heteroscedastic) — it is NOT the
    #   observation noise model (that lives in gp_model1D_simple, [NOISE 1/3]).
    # ══════════════════════════════════════════════════════════════════════
    # K_dd: derivative-derivative at observation points  (N_obs x N_obs)
    dx_oo = x_obs[:, None] - x_obs[None, :]
    K_rbf_oo = chi2 * np.exp(-0.5 * (dx_oo / ell) ** 2)
    K_dd = (1.0 / ell**2 - dx_oo**2 / ell**4) * K_rbf_oo
    del dx_oo, K_rbf_oo

    # Kstar: cross-covariance  derivative(obs) -> function(test)  (N_obs x N)
    dx_ot = x_obs[:, None] - x_test[None, :]
    K_rbf_ot = chi2 * np.exp(-0.5 * (dx_ot / ell) ** 2)
    Kstar = -(dx_ot / ell**2) * K_rbf_ot
    del dx_ot, K_rbf_ot

    # K_ff: prior function covariance at test points  (N x N)
    dx_tt = x_test[:, None] - x_test[None, :]
    K_ff = chi2 * np.exp(-0.5 * (dx_tt / ell) ** 2)
    del dx_tt

    # heteroscedastic noise + numerical jitter on the K_dd diagonal
    jitter = jitter_rel * np.trace(K_dd) / max(1, N_obs)
    np.fill_diagonal(K_dd, K_dd.diagonal() + noise_obs + jitter)

    # ---- Cholesky of K_dd  (N_obs x N_obs) ------------------------------
    try:
        L = la.cholesky(K_dd, lower=True)
    except la.LinAlgError:
        bump = max(1e-8, 1e-6 * np.mean(K_dd.diagonal()))
        np.fill_diagonal(K_dd, K_dd.diagonal() + bump)
        L = la.cholesky(K_dd, lower=True)
    del K_dd

    # ---- z = L^-1 Kstar   (N_obs x N) — do NOT overwrite Kstar ----------
    z = la.solve_triangular(L, Kstar, lower=True)
    del Kstar

    # ---- posterior covariance  Cov_A = K_ff - z^T z   (N x N) -----------
    Cov_A = K_ff - (z.T @ z)
    del K_ff
    Cov_A = 0.5 * (Cov_A + Cov_A.T)
    jitter_cov = max(1e-12, jitter_rel * max(np.trace(Cov_A) / N, 1e-12))
    np.fill_diagonal(Cov_A, Cov_A.diagonal() + jitter_cov)

    # posterior std (diagonal)
    A_posterior_std = np.sqrt(np.maximum(np.diag(Cov_A), 0.0))

    # ---- v = L^-T z   (N_obs x N) — safe to overwrite z now -------------
    v = la.solve_triangular(L.T, z, lower=False, overwrite_b=True)
    del L

    # ---- posterior mean of A  (N,)  (A = -integral of force) ------------
    A_mean = -(v.T @ f_obs)
    del v
    A_mean -= A_mean.min()

    # ---- draw posterior samples  (S x N) ---------------------------------
    try:
        LA = la.cholesky(Cov_A, lower=True)
    except la.LinAlgError:
        w, V = np.linalg.eigh(Cov_A)
        w = np.maximum(w, 1e-12)
        Cov_A = (V * w) @ V.T
        Cov_A = 0.5 * (Cov_A + Cov_A.T)
        try:
            LA = la.cholesky(Cov_A, lower=True)
        except la.LinAlgError:
            bump = max(1e-8, 1e-6 * np.mean(np.diag(Cov_A)))
            np.fill_diagonal(Cov_A, Cov_A.diagonal() + bump)
            LA = la.cholesky(Cov_A, lower=True)
    del Cov_A

    rng = np.random.default_rng()
    Z = rng.standard_normal((num_samples, N))
    A_samples = (Z @ LA.T) + A_mean[None, :]
    del LA

    # ---- trapezoidal sanity check (from full fx_mean) --------------------
    A_trap = np.zeros(N)
    A_trap[1:] = np.cumsum(
        -0.5 * (fx_mean[:-1] + fx_mean[1:]) * np.diff(xv)
    )
    A_trap -= A_trap.min()

    return A_mean, A_posterior_std, A_samples, A_trap, obs_mask


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_mean_forces_and_energy_1D(
    xv, fx_mean, fx_std, A_mean, A_posterior_std,
    histogram, new_counts, xcv_data, xnonzero_mf,
    fx_samples=None, A_samples=None, obs_mask=None,
    iter_num=0, show_reference=True, save_dir=DATA_DIR
):
    """Side-by-side plot: GP force distribution | reconstructed energy distribution."""
    os.makedirs(save_dir, exist_ok=True)
    bar_width = np.mean(np.diff(xv)) * 0.8

    if show_reference:
        ref_energy, ref_force = sum_gaus1D(xv)
        ref_energy -= ref_energy.min()

    fig, (ax_force, ax_energy) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

    # ===================== Left panel: forces ============================
    # Force GP posterior samples — show the distribution
    if fx_samples is not None:
        ax_force.plot(xv, fx_samples[0], color='C0', alpha=0.10,
                      linewidth=0.8, label='Force samples')
        for s in fx_samples[1:]:
            ax_force.plot(xv, s, color='C0', alpha=0.10, linewidth=0.8)

    # 1-sigma band and mean
    ax_force.fill_between(xv, fx_mean - fx_std, fx_mean + fx_std,
                          color='C0', alpha=0.25, label='GP 1-sigma band')
    ax_force.plot(xv, fx_mean, color='C0', lw=2, label='GP mean force')

    if show_reference:
        ax_force.plot(xv, ref_force, '--k', lw=2, label='Reference force')

    # Training data: damped per-bin mean forces
    ax_force.scatter(xcv_data, xnonzero_mf, color='darkorange', s=30,
                     zorder=5, edgecolors='black', linewidth=0.5,
                     label='Mean force data', alpha=0.8)
    ax_force.axhline(0, color='gray', ls=':', lw=1, alpha=0.5)
    ax_force.set_xlabel('Collective Variable (CV)')
    ax_force.set_ylabel('Force')
    ax_force.set_title(f'Force Distribution — Iteration {iter_num}')
    ax_force.legend(loc='upper left', fontsize=9)
    ax_force.grid(alpha=0.3)

    # Histogram overlay on force panel
    ax_fh = ax_force.twinx()
    ax_fh.bar(xv, histogram, width=bar_width, color='gray', alpha=0.25,
              edgecolor='black', linewidth=0.3, label='Total counts')
    ax_fh.bar(xv, new_counts, width=bar_width, color='red', alpha=0.45,
              edgecolor='darkred', linewidth=0.3, label='Recent counts')
    ax_fh.set_ylabel('Histogram Counts')
    ax_fh.legend(loc='upper right', fontsize=9)

    # ===================== Right panel: energy ===========================
    # Posterior energy samples — show the distribution
    if A_samples is not None and len(A_samples) > 0:
        ax_energy.plot(xv, A_samples[0], color='C1', alpha=0.15,
                       linewidth=0.8, label='Energy samples')
        for s in A_samples[1:]:
            ax_energy.plot(xv, s, color='C1', alpha=0.15, linewidth=0.8)

    # 1-sigma band and mean
    ax_energy.fill_between(xv, A_mean - A_posterior_std,
                           A_mean + A_posterior_std,
                           color='C1', alpha=0.25, label='Posterior 1-sigma band')
    ax_energy.plot(xv, A_mean, color='C1', lw=2, label='Posterior mean')

    if show_reference:
        ax_energy.plot(xv, ref_energy, '--k', lw=2, label='Reference energy')

    # Trapezoidal sanity check from raw training data
    A_trap_data = np.zeros(len(xcv_data))
    for i in range(1, len(xcv_data)):
        dx_i = xcv_data[i] - xcv_data[i - 1]
        A_trap_data[i] = (A_trap_data[i - 1]
                          - 0.5 * (xnonzero_mf[i - 1] + xnonzero_mf[i]) * dx_i)
    A_trap_data -= A_trap_data.min()
    ax_energy.scatter(xcv_data, A_trap_data, color='darkorange', s=30,
                      zorder=5, edgecolors='black', linewidth=0.5,
                      label='Trapezoid (data)', alpha=0.8)

    # Shade the region where points passed the variance threshold
    if obs_mask is not None:
        obs_x = xv[obs_mask]
        if len(obs_x) > 0:
            ax_energy.axvspan(obs_x.min(), obs_x.max(),
                              color='green', alpha=0.04, label='Observed region')

    ax_energy.set_xlabel('Collective Variable (CV)')
    ax_energy.set_ylabel('Free Energy')
    ax_energy.set_title(f'Energy Distribution — Iteration {iter_num}')
    ax_energy.legend(loc='upper left', fontsize=9)
    ax_energy.grid(alpha=0.3)

    # Histogram overlay on energy panel
    ax_eh = ax_energy.twinx()
    ax_eh.bar(xv, histogram, width=bar_width, color='gray', alpha=0.25,
              edgecolor='black', linewidth=0.3, label='Total counts')
    ax_eh.bar(xv, new_counts, width=bar_width, color='red', alpha=0.45,
              edgecolor='darkred', linewidth=0.3, label='Recent counts')
    ax_eh.set_ylabel('Histogram Counts')
    ax_eh.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    path = os.path.join(save_dir, f'forces_and_energy_{iter_num}.svg')
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {path}")



# ---------------------------------------------------------------------------
# Main entry point: retrain the force GP + reconstruct the free energy
# ---------------------------------------------------------------------------
def gp_model1D_simple(log):
    """Retrain the force GP on damped ABF mean forces and reconstruct A(x).

    Called every ``interval`` simulation steps (see Langevin_int1D in
    main.py). Returns (model, likelihood), which are stored on ``log`` and
    used to evaluate the bias force during the next interval.
    """
    # ---- Training data: damped mean force per visited bin ----------------
    # Low-visit bins have noisy mean forces; damp them by count/limit so a
    # handful of samples in a barely-sampled bin cannot dominate the fit.
    limit = 100
    dampening = np.minimum(1, log.count_array / limit)
    xmean_forces_damped = dampening * log.xmf_array_abf

    rows = np.nonzero(log.count_array)
    xnonzero_mf = xmean_forces_damped[rows]
    xcv_data = log.x_cv_grid[rows]

    X_fx, y_fx = make_training_data(xcv_data, xnonzero_mf)

    # ══════════════════════════════════════════════════════════════════════
    # [NOISE 1/3] Observation noise model
    #   Isotropic Gaussian likelihood on the (damped) force observations.
    #   By default its noise parameter is LEARNED (added to the optimizer
    #   parameters below) together with the kernel hyperparameters.
    # ══════════════════════════════════════════════════════════════════════
    lik_fx = gpytorch.likelihoods.GaussianLikelihood().to(device)


    model_fx = DerivativeGPModel1D(X_fx, y_fx, lik_fx).to(device)

    # ---- Training: 300 Adam steps on the marginal log-likelihood --------
    model_fx.train(); lik_fx.train()
    optimizer_parameters = list(model_fx.parameters())
    
    optimizer_parameters += list(lik_fx.parameters())
    optimizer = torch.optim.Adam(optimizer_parameters, lr=0.05)
    mll_fx = gpytorch.mlls.ExactMarginalLogLikelihood(lik_fx, model_fx)

    gp_training_start = time.perf_counter()
    with gpytorch.settings.cholesky_jitter(1e-4):
        for _ in range(300):
            optimizer.zero_grad()
            loss = -mll_fx(model_fx(X_fx), y_fx)
            loss.backward()
            optimizer.step()

    log_timing(
        "gpr_training",
        int(log.step / log.interval),
        time.perf_counter() - gp_training_start,
        simulation_step=log.step,
        optimizer_iterations=300,
        training_data_points=int(xnonzero_mf.size),
    )

    with torch.no_grad():
        print("fx lengthscale:", model_fx.covar_module.base_kernel.lengthscale)
        print("fx noise:      ", lik_fx.noise.item())

    # ---- Prediction on the full CV grid ----------------------------------
    model_fx.eval(); lik_fx.eval()
    X_test = torch.tensor(log.x_cv_grid.ravel().astype(np.float32),
                          dtype=torch.float32, device=device)

    num_samples = 50
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior_fx = model_fx(X_test)
        fx_mean = posterior_fx.mean.cpu().numpy()
        fx_std = posterior_fx.stddev.cpu().numpy()
        # Posterior force variance — reused below as the per-point
        # heteroscedastic noise of the energy reconstruction (see [KERNEL 3/3]).
        fx_var = posterior_fx.variance.cpu().numpy()
        fx_samples = posterior_fx.rsample(
            torch.Size([num_samples])
        ).cpu().numpy()

    # ---- Reconstruct energy ----------------------------------------------
    with torch.no_grad():
        ell = model_fx.covar_module.base_kernel.lengthscale.cpu().numpy().ravel()[0]
        chi2 = model_fx.covar_module.outputscale.item()

    A_mean, A_posterior_std, A_samples, A_trap, obs_mask = \
        reconstruct_A_from_gp_mean_1D(
            xv=log.x_cv_grid,
            fx_mean=fx_mean,
            ell=ell,
            chi2=chi2,
            noise_per_point=fx_var,
            num_samples=num_samples,
        )

    iter_num = int(log.step / log.interval)

    # ---- Save per-iteration summary data ----------------------------------
    data_dir = DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    summary_file = os.path.join(data_dir, f"data_summary{iter_num}.txt")

    xv = np.asarray(log.x_cv_grid).ravel()
    raw_mean = np.asarray(log.xmf_array_abf).ravel()
    damped_mean = np.asarray(xmean_forces_damped).ravel()
    gp_mean = np.asarray(fx_mean).ravel()
    gp_std = np.asarray(fx_std).ravel()
    energy_mean = np.asarray(A_mean).ravel()
    energy_std = np.asarray(A_posterior_std).ravel()
    histogram = np.asarray(log.count_array).ravel()
    new_counts = np.asarray(getattr(log, 'new_counts', np.zeros_like(histogram))).ravel()

    out_arr = np.column_stack((xv, raw_mean, damped_mean, gp_mean, gp_std,
                               energy_mean, energy_std, histogram, new_counts))
    header = ("x raw_mean_force damped_mean_force gp_force_mean gp_force_std "
              "energy_mean energy_std histogram new_counts")
    np.savetxt(summary_file, out_arr, header=header, comments='', fmt='%.6e')
    print(f"Wrote per-iteration summary to {summary_file}")

    # ---- Plot --------------------------------------------------------------
    plot_mean_forces_and_energy_1D(
        xv=log.x_cv_grid, fx_mean=fx_mean, fx_std=fx_std,
        A_mean=A_mean, A_posterior_std=A_posterior_std,
        histogram=log.count_array, new_counts=log.new_counts,
        xcv_data=xcv_data, xnonzero_mf=xnonzero_mf,
        fx_samples=fx_samples, A_samples=A_samples,
        obs_mask=obs_mask, iter_num=iter_num,
        show_reference=True,
        save_dir=data_dir)

    # ---- Save up to 10 force / energy posterior samples per iteration -----
    try:
        n_fx_samples = fx_samples.shape[0] if fx_samples is not None else 0
        n_A_samples = A_samples.shape[0] if A_samples is not None else 0
        n_save = min(10, n_fx_samples, n_A_samples)
        if n_save > 0:
            rng = np.random.default_rng()
            idxs = rng.choice(n_fx_samples, n_save, replace=False)
            force_file = os.path.join(data_dir, f"force_samples_{iter_num}.txt")
            energy_file = os.path.join(data_dir, f"energy_samples_{iter_num}.txt")
            np.savetxt(force_file, fx_samples[idxs], fmt='%.6e')
            np.savetxt(energy_file, A_samples[idxs], fmt='%.6e')
            print(f"Wrote {n_save} force samples -> {force_file}")
            print(f"Wrote {n_save} energy samples -> {energy_file}")
    except Exception as e:
        print(f"Warning: could not save posterior samples: {e}")

    # ---- Log hyperparameters and uncertainties (hp.txt) -------------------
    frac_over_limit = (
        int(np.count_nonzero(log.count_array > limit))
        / float(np.asarray(log.count_array).size)
    )
    hp_file = os.path.join(data_dir, "hp.txt")

    posterior_std_arr = np.asarray(A_posterior_std).ravel()
    posterior_energy_variance = posterior_std_arr ** 2
    sample_energy_variance_10 = np.var(A_samples[:10], axis=0, ddof=1)
    sample_energy_variance_25 = np.var(A_samples[:25], axis=0, ddof=1)
    sample_energy_variance_50 = np.var(A_samples, axis=0, ddof=1)
    ref_energy, _ = sum_gaus1D(xv)
    ref_energy = np.asarray(ref_energy).ravel() - np.asarray(ref_energy).ravel().min()
    gp_energy_arr = np.asarray(energy_mean).ravel()
    try:
        rmse_ref = float(np.sqrt(np.mean((gp_energy_arr - ref_energy) ** 2)))
    except Exception:
        rmse_ref = float('nan')
    record = {
        "iter": int(iter_num),
        "lengthscale": float(ell),
        "outputscale": float(chi2),
        "frac_over_limit": frac_over_limit,
        "n_obs_used": int(np.count_nonzero(obs_mask)),
        "n_grid_total": int(len(obs_mask)),
        "posterior_std_mean": float(np.mean(posterior_std_arr)),
        "posterior_std_std": float(np.std(posterior_std_arr)),
        "posterior_energy_variance_mean": float(np.mean(posterior_energy_variance)),
        "sample_energy_variance_mean_10": float(np.mean(sample_energy_variance_10)),
        "sample_energy_variance_mean_25": float(np.mean(sample_energy_variance_25)),
        "sample_energy_variance_mean_50": float(np.mean(sample_energy_variance_50)),
        "rmse_ref_energy": float(rmse_ref),
    }
    with open(hp_file, "a") as f:
        f.write(json.dumps(record) + "\n")

    # ══════════════════════════════════════════════════════════════════════
    # [NOISE 3/3] Log the effective observation noise for this iteration.
    #   Either the learned likelihood noise (default) or the fixed
    #   autocorrelation-corrected estimate; written to noise.txt so the
    #   noise trajectory can be inspected per iteration.
    # ══════════════════════════════════════════════════════════════════════
    noise_file = os.path.join(data_dir, "noise.txt")
    noise_record = {
        "iter": int(iter_num),
        "noise": float(lik_fx.noise.item()),
    }
    
    with open(noise_file, "a") as f:
        f.write(json.dumps(noise_record) + "\n")
    print(f"Logged noise {noise_record['noise']:.6e} to {noise_file}")

    return model_fx, lik_fx


if __name__ == "__main__":
    print("Made it to gpr_model.py")