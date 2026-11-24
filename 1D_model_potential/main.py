"""main.py — 1D/2D Langevin MD with ABF + GP bias on a double-Gaussian potential.

Flow
----
main()
  -> setup_output_dir()   wipe/create output directories
  -> init_log()           build Logger with all step-0 state
  -> run_sim()            Langevin loop with periodic GP retraining

Every Langevin step (Langevin_int1D / Langevin_int2D):
  1. Advance the particle (Langevin position + velocity update, periodic wrap)
  2. Accumulate ABF force samples into CV bins
  3. Every `interval` steps, retrain the force GP (gpr_model.gp_model1D_simple)
  4. Apply the GP bias force to the total force
  5. Store the new state in the circular buffer

NOTE: The GP *kernel* and *noise model* are NOT defined here — they live in
gpr_model.py (see the [KERNEL x/3] / [NOISE x/3] markers in that file).
This file only *uses* the trained model to evaluate the bias force.
"""
from __future__ import annotations

import gc
import os
import shutil
import time
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch

import logger
from gaussian_potential import gaussian_potential_1D, gaussian_potential_2D
# gp_model2D_simple is only needed for 2D runs and does not exist yet in
# gpr_model.py — keep it commented until it is implemented:
from gpr_model import (DATA_DIR, device, gp_model1D_simple, log_timing,
                       reset_timing_log)  # , gp_model2D_simple


# ══════════════════════════════════════════════════════════════════════════════
# Bias switch
# ══════════════════════════════════════════════════════════════════════════════
# ← flip to False to run unbiased Langevin dynamics (no GP bias force applied,
#   and no GP retraining in 1D mode).
BIAS: bool = True


# ══════════════════════════════════════════════════════════════════════════════
# Simulation configuration
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class SimConfig:
    """All tunable simulation parameters in one place. Edit here only."""

    # ── Thermostat / integrator ────────────────────────────────────────────
    gamma: float = 0.05            # Langevin friction coefficient (1/fs)
    k_B: float   = 8.6173324e-5    # Boltzmann constant (eV / K)
    T: float     = 500.0           # temperature (K)
    mass: float  = 3.0             # particle mass (amu)
    h: float     = 1.0             # time step (fs)

    # ── ABF / GP schedule ──────────────────────────────────────────────────
    start_bias: int = 1000         # first step at which the GP bias is applied
    interval: int   = 200          # retrain the force GP every N steps
    iter_tot: int   = 1000         # number of GP-retraining intervals

    # ── CV domain (periodic) ───────────────────────────────────────────────
    x1: float = 0.0;  x2: float = 20.0
    y1: float = 0.0;  y2: float = 20.0
    x_step_size: float = 0.3       # bin width in x  -> x_bins = (x2-x1)/dx
    y_step_size: float = 0.8       # bin width in y  -> y_bins = (y2-y1)/dy

    # ── Run mode / analysis options ────────────────────────────────────────
    dim2D: bool    = False         # True: 2D CV (x, y); False: 1D CV (x)
    min_pos: float = 7.0           # initial position (first potential well)
    center_binning: bool = True    # bins centered on grid points (periodic)

    @property
    def kT(self) -> float:
        """Thermal energy k_B*T (eV)."""
        return self.k_B * self.T

    @property
    def total_steps(self) -> int:
        """Total integration steps = warm-up + intervals * steps/interval."""
        return self.start_bias + self.interval * self.iter_tot

    @property
    def x_bins(self) -> int:
        return int((self.x2 - self.x1) / self.x_step_size)

    @property
    def y_bins(self) -> int:
        return int((self.y2 - self.y1) / self.y_step_size)


# ══════════════════════════════════════════════════════════════════════════════
# General helpers
# ══════════════════════════════════════════════════════════════════════════════

def maxwell_boltzmann_velocity(kT: float, mass: float) -> tuple[float, float]:
    """Draw (vx, vy) from the Maxwell-Boltzmann distribution at temperature T."""
    a = np.sqrt(kT / mass)
    return float(np.random.normal(0, a)), float(np.random.normal(0, a))


def _bin_index(value: float, grid: np.ndarray, center_binning: bool) -> int:
    """Return the periodic bin index of `value` under the chosen convention.

    center_binning=True : bins are centred on the grid points — the value is
                          assigned to the nearest grid point (wraps around).
    center_binning=False: bins are the intervals [grid[i], grid[i+1]) —
                          floor convention, clipped to the domain.
    """
    n = grid.size
    if n <= 1:
        return 0
    step = grid[1] - grid[0]
    offset = (value - grid[0]) / step
    if center_binning:
        return int(np.floor(offset + 0.5)) % n
    return int(np.clip(np.floor(offset), 0, n - 1))


def _periodic_upper_bound(grid: np.ndarray) -> float:
    """Exclusive upper bound x1 + n*dx for a periodic centre grid (wrap range)."""
    grid = np.asarray(grid)
    if grid.size <= 1:
        return float(grid[0]) if grid.size else 0.0
    return float(grid[0] + (grid[1] - grid[0]) * grid.size)


def abf_getindex(x: float, y, log) -> int | tuple[int, int]:
    """Map wrapped coordinates to ABF bin index(es) (1D: y is None, else 2D)."""
    center_binning = getattr(log, "center_binning", True)
    if y is None:
        x_vals = np.asarray(log.x_cv_grid)
        return _bin_index(x, x_vals, center_binning)

    x_vals = np.asarray(log.x_cv_grid[0, :])
    y_vals = np.asarray(log.y_cv_grid[:, 0])
    return (
        _bin_index(x, x_vals, center_binning),
        _bin_index(y, y_vals, center_binning),
    )


def setup_output_dir() -> None:
    """Remove stale output files and (re)create the data directory.

    WARNING: the os.system call also deletes *.pdf/*.jpeg/*.png from the
    *current working directory*, not just the data folder.
    """
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    os.system("rm -r data *.pdf *.jpeg *.png")
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs("data", exist_ok=True)


def plot_potential(x_cv: np.ndarray, y_cv: np.ndarray,
                   path: str = "./data/potential.png") -> None:
    """Save a filled-contour plot of the 2D Gaussian model potential."""
    V, _, _ = gaussian_potential_2D(x_cv, y_cv)
    fig, ax = plt.subplots(figsize=(6, 5))
    cs = ax.contourf(x_cv, y_cv, V, levels=50, cmap="viridis")
    fig.colorbar(cs, ax=ax, label="V(x, y)")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title("Double Gaussian potential")
    plt.tight_layout()
    plt.savefig(path)
    plt.close(fig)


def init_log(cfg: SimConfig, x_cv: np.ndarray, y_cv: np.ndarray) -> logger.Logger:
    """Construct Logger and set all step-0 quantities.

    Sets the initial position/Maxwell-Boltzmann velocity, evaluates the
    model potential at the initial position, seeds the first-bin force
    accumulator, and records the initial velocity to ./data/vel.txt.
    """
    log = logger.Logger(
        cfg.total_steps, cfg.x_bins, cfg.y_bins,
        x_cv, y_cv, cfg.dim2D, cfg.start_bias, cfg.interval
    )
    log.kbT        = cfg.kT
    log.max_steps  = cfg.total_steps
    log.start_bias = cfg.start_bias
    log.interval   = cfg.interval
    log.center_binning = cfg.center_binning
    log.atom1_xpositions[0] = cfg.min_pos

    if cfg.dim2D:
        log.atom1_ypositions[0]  = cfg.min_pos
        vx0, vy0 = maxwell_boltzmann_velocity(cfg.kT, cfg.mass)
        log.atom1_xvelocities[0] = vx0
        log.atom1_yvelocities[0] = vy0
        with open("./data/vel.txt", "w") as f:
            f.write("vx   vy\n")
            f.write(f"{vx0} {vy0}\n")
        log.energies[0], log.xforces[0], log.yforces[0] = gaussian_potential_2D(
            cfg.min_pos, cfg.min_pos)
        log.total_xforce_array[0] = log.xforces[0]
    else:
        vx0, _ = maxwell_boltzmann_velocity(cfg.kT, cfg.mass)
        log.atom1_xvelocities[0] = vx0
        with open("./data/vel.txt", "w") as f:
            f.write("vx\n")
            f.write(f"{vx0}\n")
        log.energies[0], log.xforces[0] = gaussian_potential_1D(cfg.min_pos)
        log.total_xforce_array[0] = log.xforces[0]

    return log


# ══════════════════════════════════════════════════════════════════════════════
# Langevin integrators
# ══════════════════════════════════════════════════════════════════════════════

def Langevin_int1D(log, gamma, kT, mass, h, current_step,
                   xbias_applied, start_bias, interval):
    """Advance the 1D system by one Langevin step.

    Parameters
    ----------
    log : logger.Logger
        Mutable simulation state (circular buffers, ABF accumulators, GP model).
    gamma, kT, mass, h :
        Friction, thermal energy, mass, time step.
    current_step : int
        Global step index.
    xbias_applied : float
        GP bias force used during the current interval (held fixed between
        GP retrains; evaluated from the GP at each step below).
    start_bias, interval : int
        Bias schedule (see SimConfig).

    Returns
    -------
    (log, conv_bool, f_xbias)
        conv_bool is always False for now (retraining-convergence flag, unused).
    """
    x_grid = np.asarray(log.x_cv_grid)
    x1 = x_grid[0]
    x2 = _periodic_upper_bound(x_grid)

    sigma = np.sqrt(2 * kT * gamma / mass)   # Langevin noise strength
    R1    = np.random.normal()
    R2    = np.random.normal()

    # ── Previous state from circular buffer ──────────────────────────────
    idx_old          = log.get_buffer_index(current_step)
    x_old            = log.atom1_xpositions[idx_old]
    vx_old           = log.atom1_xvelocities[idx_old]
    total_xforce_old = log.total_xforce_array[idx_old]

    # ── Position update ───────────────────────────────────────────────────
    # x_{n+1} = x_n + h v_n + A_n, with the stochastic acceleration
    # A_n = 0.5 h^2 (F_old - gamma v_old) + sigma h^1.5 (0.5 R1 + R2/(2 sqrt 3))
    # and periodic wrapping into [x1, x2).
    Ax_n     = (0.5 * h**2 * (total_xforce_old - gamma * vx_old)
                + sigma * h**1.5 * (0.5 * R1 + R2 / (2 * np.sqrt(3))))
    x_wrapped = x1 + (x_old + h * vx_old + Ax_n - x1) % (x2 - x1)

    E_new, fx      = gaussian_potential_1D(x_wrapped)   # model potential force
    xabf_index_new = abf_getindex(x_wrapped, None, log)

    # ── ABF finite-difference force estimate ─────────────────────────────
    # First derivative of the CV momentum (W_p = m v) via a 3-point backward
    # difference; this is the ABF force sample accumulated per bin.
    if current_step >= 2:
        i0, i1, i2 = (log.get_buffer_index(current_step - k) for k in (0, 1, 2))
        xdWp_dt = (1.5 * log.xwp_array[i0]
                   - 2.0 * log.xwp_array[i1]
                   + 0.5 * log.xwp_array[i2]) / h
    else:
        xdWp_dt = 0.0

    # ── GP retraining ─────────────────────────────────────────────────────
    # Every `interval` steps, retrain the force GP on the accumulated ABF
    # bin-mean forces (kernel + noise model: see gpr_model.py).
    conv_bool = False
    if current_step % interval == 0 and current_step > 0 and BIAS:
        log.model, log.lh = gp_model1D_simple(log)
        log.new_counts[:] = 0.0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    # ── ABF accumulation ──────────────────────────────────────────────────
    # Before start_bias the bias is not applied, so the raw ABF sample is
    # stored; afterwards the GP bias is subtracted so the ABF mean tracks
    # the *unbiased* force.
    if current_step < start_bias:
        force_sample = xdWp_dt
    else:
        force_sample = xdWp_dt + xbias_applied
    log.xfsum_abf[xabf_index_new] += force_sample
    log.count_array[xabf_index_new]   += 1
    log.new_counts[xabf_index_new]    += 1
    log.xmf_array_abf[xabf_index_new] = (log.xfsum_abf[xabf_index_new]
                                         / log.count_array[xabf_index_new])
    log.xmf_array_us[xabf_index_new]  = -fx   # reference (potential-only) force

    # ── GP bias force ─────────────────────────────────────────────────────
    # Evaluate the trained force GP at the new position (posterior mean via
    # the likelihood) and use it as the bias in the total force.
    f_xbias = 0.0
    if current_step > start_bias and BIAS:
        x_t = torch.tensor([[x_wrapped]], dtype=torch.float32, device=device)
        with torch.no_grad():
            f_xbias = log.lh(log.model(x_t)).mean[0].item()

    total_xforce_new = fx - f_xbias

    # ── Velocity update ───────────────────────────────────────────────────
    # Trapezoidal force term + friction + stochastic kick + correction.
    vx_new = (vx_old
              + 0.5 * h * (total_xforce_new + total_xforce_old)
              - h * gamma * vx_old
              + sigma * R1 * np.sqrt(h)
              - gamma * Ax_n)

    if current_step % 200 == 0:
        print(f"[1D step {current_step:6d}]  fx={fx:.4f}  bias={f_xbias:.4f}"
              f"  Ftot={total_xforce_new:.4f}  vx={vx_new:.4f}")

    # NOTE: this line writes one entry per step, and the buffer flush below
    # re-writes the same 1000 entries — lines are duplicated in this file
    # (kept as-is so existing post-processing keeps working).
    with open("./data/xforces_summary.txt", "a") as f:
        f.write(f"{x_wrapped} {fx} {f_xbias}\n")

    # ── Buffer force data, flush every 1000 steps ─────────────────────────
    log.force_buffer.append((x_wrapped, fx, f_xbias))

    if current_step % 1000 == 0 and current_step > 0:
        with open("./data/xforces_summary.txt", "a") as f:
            for entry in log.force_buffer:
                f.write(f"{entry[0]} {entry[1]} {entry[2]}\n")
        log.force_buffer.clear()

    # ── Store in circular buffer ──────────────────────────────────────────
    idx_new = log.get_buffer_index(current_step + 1)
    log.atom1_xpositions[idx_new]   = x_wrapped
    log.atom1_xvelocities[idx_new]  = vx_new
    log.energies[idx_new]           = E_new
    log.xforces[idx_new]            = fx
    log.xwp_array[idx_new]          = mass * vx_new
    log.total_xforce_array[idx_new] = total_xforce_new

    return log, conv_bool, f_xbias

# ══════════════════════════════════════════════════════════════════════════════
# Simulation runner
# ══════════════════════════════════════════════════════════════════════════════

def run_sim(log, cfg: SimConfig) -> None:
    """Run the full Langevin MD loop (step 0 init, then main loop to total_steps).

    Logs a "simulation_interval" timing record every `interval` steps to the
    JSON-lines timing file (gpr_model.log_timing).
    """

    # ── Step 0 setup ──────────────────────────────────────────────────────
    # Seed the first-bin force accumulator, then take the first step.
    simulation_start = time.perf_counter()
    log.xwp_array[0] = cfg.mass * log.atom1_xvelocities[0]
    ri = abf_getindex(log.atom1_xpositions[0], None, log)
    log.xfsum_abf[ri]   = log.xforces[0]
    log.count_array[ri] += 1
    log, _, xbias = Langevin_int1D(
        log, cfg.gamma, cfg.kT, cfg.mass, cfg.h,
        0, 0.0, cfg.start_bias, cfg.interval)
    log_timing("simulation_initialization", 0,
               time.perf_counter() - simulation_start,
               dimension=2 if cfg.dim2D else 1)

    # ── Main loop ─────────────────────────────────────────────────────────
    # `xbias`/`ybias` from the previous step are passed in as the bias for
    # the ABF correction during this step (piecewise-constant between
    # GP retrains, matching the bias actually applied last).
    interval_start = time.perf_counter()
    interval_start_step = log.step
    while log.step < cfg.total_steps:
        log, _, xbias = Langevin_int1D(
            log, cfg.gamma, cfg.kT, cfg.mass, cfg.h,
            log.step, xbias, cfg.start_bias, cfg.interval)
        log.step += 1

        interval_finished = (
            log.step % cfg.interval == 0 or log.step >= cfg.total_steps
        )
        if interval_finished:
            log_timing(
                "simulation_interval",
                (log.step + cfg.interval - 1) // cfg.interval,
                time.perf_counter() - interval_start,
                interval_start_step=interval_start_step,
                interval_end_step=log.step,
                steps=log.step - interval_start_step,
                configured_interval=cfg.interval,
                dimension=2 if cfg.dim2D else 1,
            )
            interval_start = time.perf_counter()
            interval_start_step = log.step


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    start_time = time.time()
    cfg = SimConfig()

    # CV grids: 1D centre grids meshed into a 2D array (works for both modes).
    x_1D = np.linspace(cfg.x1, cfg.x2, cfg.x_bins, endpoint=False)
    y_1D = np.linspace(cfg.y1, cfg.y2, cfg.y_bins, endpoint=False)
    x_cv, y_cv = np.meshgrid(x_1D, y_1D)

    setup_output_dir()
    reset_timing_log()
    plot_potential(x_cv, y_cv)   # saved to ./data/ so cleanup doesn't delete it

    log = init_log(cfg, x_cv, y_cv)
    run_sim(log, cfg)

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Total execution time: {elapsed:.2f} s ({elapsed/60:.2f} min)")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()