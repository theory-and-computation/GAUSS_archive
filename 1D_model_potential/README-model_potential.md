# GAUSS work flow on model potentials (1D / 2D Langevin dynamics)

This repository provides a workflow for GAUSS, an enhanced sampling method that uses molecular dynamics simulations with Bayesian inference based on Gaussian Process Regression (GPR) to compute the free energy surface using an iterative biasing scheme.
Goal of using double Gaussian well potentials with Langevin dynamics: To recover the model free energy A(x) and compare it against an exact analytical free energy function.

## Files

| File                   | Role |
|------------------------|------|
| `main.py`              | Driver: `SimConfig` (all tunables), Langevin integrators (1D/2D), main loop, output setup |
| `logger.py`            | `Logger`: circular trajectory buffers (last 3 steps), ABF bin accumulators, per-bin force blocks, storage of the trained GP |
| `gpr_model.py`         | **The GP itself**: force-kernel model, training loop, free-energy reconstruction, diagnostics/plots |
| `gaussian_potential.py`| Model potential — single source of truth for the double-Gaussian wells (used by `main.py` and by the analytic reference) |

## Flow
main()
└─ setup_output_dir, reset_timing_log, plot_potential
└─ init_log() Logger + step-0 state (x0, Maxwell–Boltzmann v0)
└─ run_sim() while step < total_steps
└─ Langevin_int1D (or int2D) position/velocity update, periodic wrap
├─ ABF sample: dW_p/dt (3-pt backward difference) → bin accumulators
│ (Logger.xfsum_abf → xmf_array_abf = bin-mean force)
├─ every interval steps → gp_model1D_simple(log)
│ ├─ damp bin means by min(1, count/100)
│ ├─ train DerivativeGPModel1D (300 Adam steps, MLL)
│ ├─ predict on full grid; reconstruct A(x) via
│ │ reconstruct_A_from_gp_mean_1D (K_dd / Kstar / K_ff)
│ └─ write data_summary{i}.txt, hp.txt, noise.txt, plot
└─ bias force = likelihood(model(x)).mean → F_total = F_potential − F_bias


## Kernel (prior over the force) — `gpr_model.py`

| Marker | Location | What |
|--------|----------|------|
| `[KERNEL 1/3]` | `DerivativeGPModel1D.__init__` | RBF base kernel (lengthscale constrained to [0.02, 2.0]) wrapped in a Scale kernel — the two learned hyperparameters (ell, σ²) |
| `[KERNEL 2/3]` | `DerivativeGPModel1D.forward` | Closed-form **force kernel** K_f(d) = σ²(1/ell² − d²/ell⁴)exp(−d²/2ell²) = Cov[dA/dx, dA/dx′] |
| `[KERNEL 3/3]` | `reconstruct_A_from_gp_mean_1D` | Analytic K_dd / Kstar / K_ff matrices used to integrate the force posterior back into the free-energy posterior |

## Noise model (observation model for the forces) — `gpr_model.py`

| Marker | Location | What |
|--------|----------|------|
| `[NOISE 1/3]` | `gp_model1D_simple` | `GaussianLikelihood()`; by default the noise is **learned** (added to the optimizer parameters) |
| `[NOISE 2/3]` | `_estimate_global_block_noise` + frozen-likelihood block | If `autocorrelation_correction=True`: one **fixed** global noise from the variance of per-bin block means (blocks recorded by `Logger.record_force_sample` → `force_block_means`); likelihood noise frozen |
| `[NOISE 3/3]` | end of `gp_model1D_simple` | Effective noise per iteration logged to `noise.txt` |

Note: the small diagonal jitters inside the GP/reconstruction code are
numerical stabilizers, **not** the observation noise model.

## Data flow into the GP (via `Logger`)

- `count_array` — bin visits → training-data damping + visited-bin selection
- `xmf_array_abf` — bin-mean ABF force → **training targets**
- `force_block_means` — block means → autocorrelation noise estimate
- `new_counts` — visits since last retrain → plots only

## Run & switch

- `python main.py` — 1D by default (`SimConfig.dim2D = False`); set `BIAS = False`
  at the top of `main.py` for unbiased dynamics.
- ⚠️ 2D mode is **not runnable yet**: `gp_model2D_simple` does not exist in
  `gpr_model.py` (import is commented out in `main.py`).
- Outputs: `data/` (trajectory forces, velocities) and
  `gpr_model.DATA_DIR_NAME/` (summaries, `hp.txt`, `noise.txt`, plots).
