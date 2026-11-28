# GAUSS implementaion for VASP simulation of butane hydrogneolysis on Ru

This repository provides a workflow for GAUSS, an enhanced sampling method that uses molecular dynamics simulations (e.g., NAMD and VASP) with Bayesian inference based on Gaussian Process Regression (GPR) to compute the free energy surface using an iterative biasing scheme. The framework models the free-energy surface (FES) and its gradient field directly from observed mean forces and incorporates an uncertainty-driven exploration bias directly into VASP ionic steps via the interactive Python plugin interface.

---

## 1. Codebase Architecture

The VASP implementation consists of five Python components:

| File | Purpose |
| :--- | :--- |
| `main.py` | Primary workflow orchestrator: manages walker runs, parallel MPI launches (`ibrun`/`mpirun`), thermal spike monitoring, file archiving, and iteration cycling. |
| `./inputs/vasp_plugin.py` | Direct runtime interface hook (`force_and_stress`) called by VASP at every ionic step to evaluate CVs, lookup GPR bias forces, apply soft wall restraints, and log applied forces. |
| `gpr_model.py` | Statistical engine implementing 1D/2D derivative covariance kernels, posterior GP inference, two-stage energy path reconstruction, and harmonic exploration push calculation. |
| `abf.py` | ABF trajectory mechanics: extracts coordinates and velocities via `py4vasp`, computes Jacobians ($J = \partial \xi / \partial \mathbf{r}$) via PyTorch autograd, and projects momenta ($W_p$). |
| `logger.py` | State tracking: encapsulates `Atom` and `CollectiveVariable` descriptors, maintains 1D/2D bin histograms and running force accumulators, and handles checkpoint serialization. |

---

## 2. Model & Statistical Details

All machine learning models and covariance formulations are located in **`gpr_model.py`**:

### A. Derivative Covariance Kernels (Section 2)
The framework models the potential gradient observations $\mathbf{y} = \nabla A(\mathbf{x})$ directly using analytic derivative kernels:

1. **Mixed Non-Periodic $\times$ Periodic Space (e.g., Distance $\times$ Dihedral)**:
   - **Class**: `RBFxPeriodicDerivativeKernel2D`
   - **Base Latent Potential Kernel**:
     $$k(\mathbf{x}_1, \mathbf{x}_2) = \exp\left(-\frac{1}{2}\frac{(R_1 - R_2)^2}{\ell_R^2}\right) \exp\left(-2\frac{\sin^2\left(\frac{\phi_1 - \phi_2}{2}\right)}{\ell_\phi^2}\right)$$
   - Evaluates the analytic second mixed derivatives $\text{Cov}\left(\frac{\partial A}{\partial x_a}, \frac{\partial A}{\partial x_b}\right)$ with periodic wrapping strictly maintained on $[-\pi, \pi)$.

2. **Full Non-Periodic Space (e.g., Distance $\times$ Radius of Gyration)**:
   - **Class**: `RBFDerivativeKernel2D`
   - Fully analytic 2D squared-exponential product kernel derivative covariance.

3. **1D Derivative Space**:
   - **Class**: `DerivativeGPModel1D` (Section 3)

### B. Observation Noise & Likelihood Models (Section 8)
- Force observations are modeled with independent Gaussian noise: `gpytorch.likelihoods.GaussianLikelihood()`.
- Signal variance $\sigma_f^2$ is bounded via `gpytorch.kernels.ScaleKernel(..., outputscale_constraint=Interval(0.001, 10.0))`.
- Hyperparameters (lengthscales $\boldsymbol{\ell}$ and outputscale $\sigma_f^2$) are optimized via Adam with Cosine Annealing (`_train_model`).

### C. Two-Stage Free Energy Reconstruction (Section 5)
Free energy profiles are reconstructed via `reconstruct_energy_from_forces()` with full uncertainty decomposition:
1. **Stage 1 (Force Model Uncertainty)**: Spread of reconstructed profiles across posterior GP force samples ($\sigma_{\text{force}}$).
2. **Stage 2 (Reconstruction Uncertainty)**: Posterior covariance from the finite derivative inversion ($\sigma_{\text{recon}}$).
3. **Total Uncertainty**: Combined in quadrature:
   $$\sigma_{\text{total}} = \sqrt{\sigma_{\text{force}}^2 + \sigma_{\text{recon}}^2}$$

### D. Active Exploration Bias (Section 6)
- **Function**: `_compute_exploration_push_2d()`
- Evaluates total model uncertainty across the CV grid, selects the maximal-variance coordinate $(\xi_1^*, \xi_2^*)$, and constructs a harmonic restraint field scaled dynamically by simulation step and phase-space coverage.

---

## 3. Environment & Path Configuration

All system-dependent paths have been abstracted into environment variables. Configure them in your submission script or shell before launching:

### Environment Variables
```bash
# Path to your VASP executable (must be compiled with Python plugin support)
export VASP_BINARY="/path/to/vasp.6.x.x/bin/vasp_std"

# Conda or Python library path (if VASP needs shared dynamic libs at runtime)
export VASP_LIB_PATH="/path/to/miniconda3/envs/gpr_env/lib"

# MPI launcher: 'ibrun' (TACC), 'srun' (Slurm), or 'mpirun' (OpenMPI/Intel)
export MPI_RUN_CMD="ibrun"

# Hardware allocation: number of MPI cores per walker job
export CORES_PER_JOB=48
```

### Template Directory Layout
Place reference inputs in an `./inputs/` folder at the root of your working directory:
```
./
├── inputs/
│   ├── INCAR_1            # Unbiased MD parameters (used for iteration 0 & re-equilibration)
│   ├── INCAR_2            # Biased MD parameters (loads vasp_plugin.py)
│   ├── POSCAR_1           # Starting structure(s) (e.g., POSCAR_1, POSCAR_2 for multiple walkers)
│   ├── POTCAR             # Standard VASP pseudopotential file
│   ├── KPOINTS            # Gamma-point or grid sampling
│   ├── ICONST             # Constraint / CV definitions
│   └── vasp_plugin.py     # Interactive biasing script template
```

---

## 4. Collective Variable Setup in `vasp_plugin.py`

Atom indices (0-based) for your collective variables are defined at the top of `vasp_plugin.py`:

```python
# =============================================================================
# CONFIGURABLE ATOM INDICES
# =============================================================================
dist_ion1 = 1      # Atom 1 index for distance CV
dist_ion2 = 3      # Atom 2 index for distance CV

dih_ion1  = 3      # Atom 1 index for dihedral CV
dih_ion2  = 1      # Atom 2 index for dihedral CV
dih_ion3  = 2      # Atom 3 index for dihedral CV
dih_ion4  = 0      # Atom 4 index for dihedral CV (set to 0 for 1D distance runs)
```

During execution, `main_vasp.py` automatically keeps these synchronized with the `Logger` atom definitions via `update_vasp_plugin_atoms()`.

---

## 5. Software Requirements & Dependencies

Ensure your environment has Python 3.10+ and the following packages:

```bash
# Core math and visualization
pip install numpy scipy matplotlib imageio

# PyTorch and GP framework
pip install torch gpytorch

# VASP trajectory and output parsing
pip install py4vasp

# Optional: JAX (if automatic differentiation for CVs is needed)
pip install jax jaxlib
```

---

## 6. Running the Simulation

Execute the workflow from your submission script or terminal:

```bash
# Usage: python main_vasp.py <iterations> <walkers> <dimensions> [--cv1 CV1] [--cv2 CV2]
python main_vasp.py 20 4 2 --cv1 dist --cv2 dihedral --folder runs
```

### Command-Line Arguments:
- `iterations` *(int)*: Number of iterative cycles to run.
- `walkers` *(int)*: Number of parallel VASP replica directories (`run_01`, `run_02`, ...).
- `dimensions` *(int)*: Dimensionality of CV space (`1` or `2`).
- `--cv1` *(str)*: Primary CV identifier (`dist`, `dihedral`, `rg`). Default: `dist`.
- `--cv2` *(str)*: Secondary CV identifier (`dist`, `dihedral`, `rg`). Default: `dihedral`.
- `--folder` *(str)*: Base directory containing walker subfolders. Default: `runs`.
- `--cold-start`: Wipes prior `runs/`, `data/`, and `figs/` directories for a clean run (default is warm start / resume).

### Output Files & Diagnostics
- `job_details.txt`: Runtime log tracking VASP exit statuses, wall clock durations, and GPR training statistics.
- `model_data.txt`: Latest 2D/1D force grid evaluated by GPR and read by `vasp_plugin.py`.
- `data/abf_state.npz`: Checkpoint containing raw histograms and force sums for restarting.
- `figs/gpr_plots/`: Contour maps of the learned free energy, applied exploration biases, and visit histograms.
- `figs/gpr_evolution.gif`: Animated summary of free energy surface convergence across iterations.
