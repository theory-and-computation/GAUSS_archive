# GAUSS implementation in NAMD with solvated alanine dipeptide

This repository provides an iterative enhanced sampling workflow combining molecular dynamics simulations (e.g., NAMD/VASP) with Gaussian Process Regression (GPR). The framework models the free-energy surface (FES) and its gradient field directly from observed mean forces (Adaptive Biasing Force / ABF formulation) and incorporates an uncertainty-driven exploration bias.

---

## 1. Codebase Structure

The workflow consists of four modular Python scripts:

| File | Purpose |
| :--- | :--- |
| `main.py` | Orchestration entry point; handles the outer iterative simulation loop, trajectory accumulation, directory creation, and job scheduling. |
| `gpr_model.py` | Implements the analytic 2D derivative covariance kernel, GP training, Poisson/FFT integration, and exploration push force calculation. |
| `logger.py` | Maintains simulation state across iterations, bin visit histograms, running average force accumulators, and collective variable (CV) geometry definitions. |
| `namd_func.py` | Interface wrapper for the NAMD MD engine; computes CV Jacobians via JAX reverse-mode automatic differentiation and atomic conjugate momenta. |

---

## 2. Where to Find the Model Components

All statistical and machine learning modeling components reside in **`gpr_model.py`**:

### A. Covariance Kernel Formulation
- **Location**: Class `PeriodicDerivativeKernel2D` (top of `modular_gprmodel_clean.py`).
- **Details**: Evaluates the analytic second derivatives $\text{Cov}\left(\frac{\partial A}{\partial x_a}, \frac{\partial A}{\partial x_b}\right)$ of a 2D periodic potential kernel:
  $$k(\mathbf{x}_1, \mathbf{x}_2) = \exp\left(-2 \sum_{d=1}^2 \frac{\sin^2(\pi(x_{1,d} - x_{2,d}) / p_d)}{\ell_d^2}\right)$$
- **Periodicity Setup**: In Step 2 of `gp_model2D_gpder`, the domain period $p_d$ is fixed to the known coordinate domain extent ($\Delta\phi, \Delta\psi = 360^\circ$ for dihedrals), while ARD lengthscales $(\ell_\phi, \ell_\psi)$ and signal variance $\sigma_f^2$ (`ScaleKernel`) are optimized via maximum marginal likelihood.

### B. Observation Noise Model
- **Location**: Step 2 inside function `gp_model2D_gpder()`.
- **Details**: Observation noise on the force components is modeled using `gpytorch.likelihoods.GaussianLikelihood`. It enforces positivity via `Positive()` constraints and regularizes variance estimation via a Gamma prior:
  $$\sigma_\epsilon^2 \sim \text{Gamma}(\alpha = 2.0, \beta = 10.0)$$

### C. Free Energy Reconstruction (FES)
- **Analytic Matrix Formulation**: `reconstruct_A_from_gp_samples_2D_no_norm_fast()` computes the zero-mean latent free energy from posterior force samples using the analytic function-derivative cross-covariance $\mathbf{K}_*$.
- **Poisson / FFT Integration**: `predict_energy_from_mean_forces()` solves $\nabla^2 A = -\nabla \cdot \mathbf{F}$ in Fourier space via 2D Fast Fourier Transforms (FFT).

### D. Uncertainty-Driven Exploration Bias
- **Location**: Function `compute_exploration_push()` in `modular_gprmodel_clean.py`.
- **Details**: Evaluates local model uncertainty $\sigma_F(\mathbf{x}) = \sqrt{\sigma_{Fx}^2 + \sigma_{Fy}^2}$ and directs a harmonic restraint toward the maximal-uncertainty region of CV space, scaled by simulation iteration and grid coverage.

---

## 3. Configuring Custom Paths & Environments

To ensure privacy and portability, hardcoded paths have been removed. You can configure your local execution environment using either **environment variables** or **command-line arguments**.

### 1. Template Input Files (PDB, PSF, NAMD Configs)
The simulation runner copies the configuration templates (coordinate files, execution scripts `run_exe.sh` / `eq_exe.sh`, and Tcl extraction scripts) into the working run folder.

- **Option A (CLI Argument)**:
  ```bash
  python main.py --tag run_01 --template-dir /path/to/your/templates