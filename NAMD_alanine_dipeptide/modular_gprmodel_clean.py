import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import scipy.linalg as la
import torch
import gc


def compute_exploration_push(fx_std_grid, fy_std_grid, x_cv_grid, y_cv_grid,
                             job, frac_nonzero,
                             margin_x=4, margin_y=4,
                             kb=0.001987, T=300, push_coeff=0.01,
                             ramp_start=5, ramp_iters=50,
                             frac_start=0.85, frac_end=0.95):
    """Compute exploration push forces and bias energy from uncertainty grids.

    Returns: fx_push, fy_push, exploration_bias_energy, U_push, phi_max, psi_max,
             i_max, j_max, masked_uncertainty, uncertainty_grid, exploration_scale
    """
    uncertainty_grid = np.sqrt(fx_std_grid**2 + fy_std_grid**2)
    nx, ny = uncertainty_grid.shape

    # mask borders to avoid edge artifacts
    mask = np.zeros_like(uncertainty_grid, dtype=bool)
    mask[margin_x:nx-margin_x, margin_y:ny-margin_y] = True
    masked_uncertainty = np.where(mask, uncertainty_grid, -np.inf)

    flat_idx = np.argmax(masked_uncertainty)
    i_max, j_max = np.unravel_index(flat_idx, masked_uncertainty.shape)

    phi_max = x_cv_grid[i_max, j_max]
    psi_max = y_cv_grid[i_max, j_max]

    # harmonic spring centered at max uncertainty
    k_push = kb * T * 0.05

    def wrap_delta_deg(a, a0):
        return (a - a0 + 180.0) % 360.0 - 180.0

    dphi = wrap_delta_deg(x_cv_grid, phi_max)
    dpsi = wrap_delta_deg(y_cv_grid, psi_max)

    fx_push_raw = -k_push * dphi
    fy_push_raw = -k_push * dpsi

    iter_idx = int(job)
    if iter_idx <= ramp_start:
        time_ramp = 0.0
    else:
        time_ramp = (iter_idx - ramp_start) / ramp_iters
        time_ramp = min(time_ramp, 1.0)

    if frac_nonzero <= frac_start:
        coverage_scale = 1.0
    elif frac_nonzero >= frac_end:
        coverage_scale = 0.0
    else:
        coverage_scale = (frac_end - frac_nonzero) / (frac_end - frac_start)

    exploration_scale = time_ramp * coverage_scale

    fx_push = fx_push_raw * exploration_scale
    fy_push = fy_push_raw * exploration_scale

    U_push = 0.5 * k_push * (np.square(dphi) + np.square(dpsi))
    exploration_bias_energy_raw = U_push
    exploration_bias_energy = exploration_scale * exploration_bias_energy_raw

    return (fx_push, fy_push, exploration_bias_energy, U_push,
        phi_max, psi_max, i_max, j_max, masked_uncertainty, uncertainty_grid, exploration_scale)


class ForceGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel 

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def make_training_data(x_grid, y_grid, f_grid, subsample=True):
    """
    Build training data for GP regression.

    Cases:
      1D: x_grid, f_grid: shape (n,) or (nx, 1)
          -> X: [n, 1], y: [n]

      2D: x_grid, y_grid, f_grid: shape (nx, ny)
          -> X: [nx*ny, 2], y: [nx*ny]
    """
    print('making training data')

    x_arr = np.asarray(x_grid)
    f_arr = np.asarray(f_grid)

    if y_grid is None:
        # 1D case
        X_np = x_arr.reshape(-1, 1)          # [N, 1]
        y_np = f_arr.reshape(-1)             # [N]
    else:
        # 2D case
        if subsample:
            y_arr = np.asarray(y_grid)
            X_np = np.stack([x_arr.ravel()[::2], y_arr.ravel()[::2]], axis=-1)  # [N, 2]
            y_np = f_arr.ravel()[::2]
        else:
            y_arr = np.asarray(y_grid)
            X_np = np.stack([x_arr.ravel(), y_arr.ravel()], axis=-1)  # [N, 2]
            y_np = f_arr.ravel()                                      # [N]


    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    return X, y


def reconstruct_A_from_gp_samples_2D_no_norm_fast(
    xv, yv,
    samples_phys,   # torch or np array, shape (S, 2, nx, ny): [Fx, Fy]
    ell,            # length-2 array, periodic ARD lengthscales (same units as xv,yv)
    chi2,           # scalar, kernel outputscale (σ_f^2)
    noise,          # noise on forces; same semantics as before
    period=None,    # length-2 array, period per dimension (same units as xv,yv)
    noise_is_std=True,
    y_is_force=True,
    jitter_rel=1e-10,
    center="min",
):
    """
    Reconstruct A(x,y) from multiple GP force samples in one shot (no normalization).
    Uses the periodic ARD kernel (matching PeriodicDerivativeKernel2D).

    Inputs
    ------
    xv, yv       : (nx, ny) grids
    samples_phys : (S, 2, nx, ny), GP samples of forces or gradients
                   samples_phys[s, 0] = Fx^(s)(x,y), samples_phys[s, 1] = Fy^(s)(x,y)
    ell          : length-2 array (ℓ_x, ℓ_y), same units as xv,yv
    chi2         : scalar kernel outputscale (σ_f^2)
    period       : length-2 array (p_x, p_y), period per dimension.
                   Defaults to domain extent (xv.ptp(), yv.ptp()) if None.
    noise        : noise on derivative observations; supports same shapes as before
    noise_is_std : True if `noise` is std, False if variance
    y_is_force   : True if samples_phys are forces (= -∇A); False if +∇A
    jitter_rel   : relative jitter for numerical stability
    center       : "min" -> shift A so global min(A_mean) = 0

    Returns
    -------
    A_samples : (S, nx, ny) reconstructed A(x,y) samples
    A_mean    : (nx, ny) sample mean of A
    A_std     : (nx, ny) sample std of A across S samples
    A_trap    : None (no trapezoidal integration)
    """
    # ----- shapes & convert to numpy -----
    xv = np.asarray(xv, dtype=float)
    yv = np.asarray(yv, dtype=float)
    nx, ny = xv.shape
    N = nx * ny

    samples_np = np.asarray(samples_phys, dtype=float)  # (S, 2, nx, ny)
    S, C, nx_s, ny_s = samples_np.shape
    assert C == 2, "samples_phys should have 2 components (Fx, Fy)"
    assert nx_s == nx and ny_s == ny, "samples_phys grid must match xv,yv"

    # ----- coordinates -----
    X = np.column_stack([xv.ravel(), yv.ravel()])  # (N, 2)

    # ----- hyperparameters -----
    ell = np.asarray(ell, dtype=float).ravel()
    if ell.size != 2:
        raise ValueError("ell must be length 2 (for 2D).")

    if period is None:
        period = np.array([np.ptp(xv), np.ptp(yv)], dtype=float)
    period = np.asarray(period, dtype=float).ravel()
    if period.size != 2:
        raise ValueError("period must be length 2 (for 2D).")

    # c[d] = 2π / (p_d * ℓ_d²)
    c_coeff = 2.0 * np.pi / (period * ell**2)         # (2,)

    if chi2 is None or chi2 <= 0.0:
        # fallback if needed
        chi2 = max(1e-6, np.var(samples_np))

    # ----- base periodic kernel k(x1, x2) -----
    dx = X[:, None, :] - X[None, :, :]                 # (N,N,2)
    u  = np.pi * dx / period[None, None, :]             # (N,N,2)  u_d = π Δx_d / p_d
    sin_u = np.sin(u)                                   # (N,N,2)
    sin2u = np.sin(2.0 * u)                             # (N,N,2)
    cos2u = np.cos(2.0 * u)                             # (N,N,2)
    sqdist = 2.0 * np.sum(sin_u**2 / ell[None, None, :]**2, axis=2)  # (N,N)
    Kbase = chi2 * np.exp(-sqdist)                      # (N,N)

    # ----- K_blocks[i,j,a,b] = Cov(dA/dx_a(xi), dA/dx_b(xj)) -----
    # same component (a==b): k * [c_a*(2π/p_a)*cos(2u_a) - c_a²*sin²(2u_a)]
    # cross component (a!=b): -k * c_a * c_b * sin(2u_a)*sin(2u_b)
    K_blocks = np.empty((N, N, 2, 2), dtype=float)
    for _a in range(2):
        for _b in range(2):
            if _a == _b:
                K_blocks[:, :, _a, _b] = Kbase * (
                    c_coeff[_a] * (2.0 * np.pi / period[_a]) * cos2u[..., _a]
                    - c_coeff[_a]**2 * sin2u[..., _a]**2
                )
            else:
                K_blocks[:, :, _a, _b] = (
                    -Kbase * c_coeff[_a] * c_coeff[_b]
                    * sin2u[..., _a] * sin2u[..., _b]
                )

    # Interleave components → (2N, 2N)
    ND = 2 * N
    K = K_blocks.transpose(0, 2, 1, 3).reshape(ND, ND)

    # ----- noise on forces, as variance per-point per-component -----
    def _broadcast_noise(noise_val, nx_, ny_, noise_is_std_):
        if noise_val is None:
            return np.zeros((nx_*ny_, 2), dtype=float)
        n = np.asarray(noise_val)
        if noise_is_std_:
            n = n.astype(float)**2  # convert std → var

        if n.ndim == 0:
            n = np.full((nx_*ny_, 2), float(n), dtype=float)
        elif n.ndim == 1:
            if n.size == 2:
                n = np.tile(n.reshape(1, 2), (nx_*ny_, 1))
            elif n.size == nx_*ny_:
                n = np.stack([n, n], axis=1)
            else:
                raise ValueError("1D `noise` must be length 2 or N=nx*ny.")
        elif n.ndim == 2:
            if n.shape == (nx_, ny_):
                n = np.repeat(n.reshape(nx_*ny_, 1), 2, axis=1)
            elif n.shape == (nx_*ny_, 2):
                pass
            else:
                raise ValueError("2D `noise` must be (nx,ny) or (N,2).")
        elif n.ndim == 3 and n.shape == (nx_, ny_, 2):
            n = n.reshape(nx_*ny_, 2)
        else:
            raise ValueError("Unsupported `noise` shape.")
        return n.astype(float)

    Sigma_pt = _broadcast_noise(noise, nx, ny, noise_is_std)   # (N,2)
    Sigma_diag = np.empty(ND, dtype=float)
    Sigma_diag[0::2] = Sigma_pt[:, 0]
    Sigma_diag[1::2] = Sigma_pt[:, 1]

    # ----- add noise + jitter to K diagonal -----
    mean_diag = float(np.mean(np.diag(K))) if K.size else 1.0
    jitter = max(1e-14, jitter_rel * max(mean_diag, 1.0))

    K_plus = K
    diag_idx = np.diag_indices_from(K_plus)
    K_plus[diag_idx] += Sigma_diag + jitter

    # ----- Kstar: derivative–function cross covariances -----
    # ∂k_per/∂x_{i,a} = -k * c_coeff[a] * sin(2*u_a)
    Xm = X
    dxm  = X[:, None, :] - Xm[None, :, :]                       # (N,M,2)
    um   = np.pi * dxm / period[None, None, :]                   # (N,M,2)
    sin2um  = np.sin(2.0 * um)                                   # (N,M,2)
    sqdist_star = 2.0 * np.sum(np.sin(um)**2 / ell[None, None, :]**2, axis=2)  # (N,M)
    Kbase_star = chi2 * np.exp(-sqdist_star)                     # (N,M)
    # Kstar_blocks[i,m,a] = ∂k(x_i, x_m)/∂x_{i,a}
    Kstar_blocks = -Kbase_star[:, :, None] * (sin2um * c_coeff[None, None, :])  # (N,M,2)

    Kstar = np.empty((ND, Xm.shape[0]), dtype=float)             # (2N, N)
    Kstar[0::2, :] = Kstar_blocks[:, :, 0]
    Kstar[1::2, :] = Kstar_blocks[:, :, 1]

    # ----- assemble all RHS (forces) for S samples -----
    # samples_np: (S, 2, nx, ny) → (S, 2, N)
    fx_all = samples_np[:, 0].reshape(S, N)  # (S,N)
    fy_all = samples_np[:, 1].reshape(S, N)  # (S,N)

    Y = np.empty((ND, S), dtype=float)      # (2N, S)
    Y[0::2, :] = fx_all.T                   # (N,S)
    Y[1::2, :] = fy_all.T

    # ----- solve (K_plus) alpha = Y for all S at once -----
    try:
        L = la.cholesky(K_plus, lower=True, check_finite=False)
    except la.LinAlgError:
        bump = max(1e-12, 1e-6 * mean_diag)
        L = la.cholesky(K_plus + bump * np.eye(ND), lower=True, check_finite=False)

    # two triangular solves with multiple RHS
    tmp = la.solve_triangular(L, Y, lower=True, check_finite=False)      # (2N,S)
    alpha_all = la.solve_triangular(L.T, tmp, lower=False, check_finite=False)  # (2N,S)

    # ----- posterior means of B(x) for all samples -----
    # B_mean_all: (M,S) = (N,S)
    B_mean_all = Kstar.T @ alpha_all

    # If y are forces = -∇A, then B = -A + const -> A = -B
    if y_is_force:
        A_samples = -B_mean_all.T.reshape(S, nx, ny)
    else:
        A_samples = B_mean_all.T.reshape(S, nx, ny)

    # ----- statistics across samples -----
    A_mean = -1*A_samples.mean(axis=0)      # (nx,ny)
    A_std = A_samples.std(axis=0)       # (nx,ny)

    # shift reference
    if center == "min":
        offset = A_mean.min()
        A_samples -= offset
        A_mean -= offset

    A_trap = None
    return A_samples, A_mean, A_std, A_trap


class RBFDerivativeKernel2D(gpytorch.kernels.Kernel):
    """
    Kernel that returns Cov( dA/dx_a(x1), dA/dx_b(x2) ) for a latent RBF-ARD kernel on A.
    Input format: [x, y, comp] where comp=0 -> d/dx, comp=1 -> d/dy.
    """
    has_lengthscale = True

    def forward(self, x1, x2, diag: bool = False, **params):
        X1 = x1[..., :2]
        X2 = x2[..., :2]
        c1 = x1[..., 2].long()
        c2 = x2[..., 2].long()

        ell = self.lengthscale.squeeze(0)            # (2,)
        q = (1.0 / (ell ** 2)).to(X1.dtype)          # (2,)

        if diag:
            return q[c1]  # since k(x,x)=1 (ScaleKernel adds outputscale)

        dx = X1.unsqueeze(-2) - X2.unsqueeze(-3)     # (n1,n2,2)
        sqdist = (dx.pow(2) * q).sum(dim=-1)         # (n1,n2)
        Kbase = torch.exp(-0.5 * sqdist)             # (n1,n2)

        n1, n2 = Kbase.shape
        c1m = c1.view(-1, 1).expand(n1, n2)
        c2m = c2.view(1, -1).expand(n1, n2)

        q_a = q[c1m]
        q_b = q[c2m]

        dx_a = dx.gather(-1, c1m.unsqueeze(-1)).squeeze(-1)
        dx_b = dx.gather(-1, c2m.unsqueeze(-1)).squeeze(-1)

        term1 = Kbase * (c1m == c2m).to(Kbase.dtype) * q_a
        term2 = Kbase * (dx_a * q_a) * (dx_b * q_b)
        return term1 - term2


class PeriodicDerivativeKernel2D(gpytorch.kernels.Kernel):
    """
    Kernel for Cov( dA/dx_a(x1), dA/dx_b(x2) ) where the latent A has a
    2-D ARD periodic base kernel:

        k(x1, x2) = exp( -2 * sum_d sin^2(pi*(x1_d - x2_d)/p_d) / l_d^2 )

    Input format: [x, y, comp] where comp=0 -> d/dx, comp=1 -> d/dy.

    The analytic second derivative is:
      same component (a == b):
        k * [ (4*pi^2)/(p_a^2 * l_a^2) * cos(2*u_a)
              - (4*pi^2)/(p_a^2 * l_a^4) * sin^2(2*u_a) ]
      cross component (a != b):
        -k * [2*pi/(p_a*l_a^2)] * [2*pi/(p_b*l_b^2)] * sin(2*u_a)*sin(2*u_b)
    """
    has_lengthscale = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # period_length: shape (1, 2) — can be fixed after construction
        self.register_parameter(
            "raw_period_length",
            torch.nn.Parameter(torch.zeros(1, 2)))
        self.register_constraint(
            "raw_period_length",
            gpytorch.constraints.Positive())

    @property
    def period_length(self):
        return self.raw_period_length_constraint.transform(self.raw_period_length)

    @period_length.setter
    def period_length(self, value):
        if not torch.is_tensor(value):
            value = torch.tensor(value)
        self.initialize(
            raw_period_length=self.raw_period_length_constraint.inverse_transform(
                value.view(1, -1).to(self.raw_period_length)))

    def forward(self, x1, x2, diag: bool = False, **params):
        # Split spatial coords and component indicator
        X1 = x1[..., :2]          # (..., n1, 2)
        X2 = x2[..., :2]          # (..., n2, 2)
        c1 = x1[..., 2].long()    # (..., n1)
        c2 = x2[..., 2].long()    # (..., n2)

        ell = self.lengthscale.squeeze(0)      # (2,)
        p   = self.period_length.squeeze(0)    # (2,)

        # c_d = 2*pi / (p_d * l_d^2)
        c_coeff = (2.0 * torch.pi / (p * ell ** 2)).to(X1.dtype)   # (2,)

        if diag:
            # x1 == x2 at each point: u_d = 0, sin=0, cos=1
            # same-component: c_a^2 / c_a * p_a / (2*pi) ... simplify:
            #   k_diag = 1 (Kbase=1) * 4*pi^2/(p_a^2*l_a^2) = c_a * 2*pi/p_a
            same = (c1 == c2)
            ca = c_coeff[c1]
            pa = p[c1]
            return torch.where(same,
                               ca * (2.0 * torch.pi / pa),
                               torch.zeros_like(ca))

        # --- full n1 x n2 matrix ---
        # u_d[i,j,d] = pi * (X1[i,d] - X2[j,d]) / p[d]
        dx  = X1.unsqueeze(-2) - X2.unsqueeze(-3)        # (n1,n2,2)
        u   = torch.pi * dx / p                          # (n1,n2,2)  u_d
        sin2u = torch.sin(2.0 * u)                       # (n1,n2,2)
        cos2u = torch.cos(2.0 * u)                       # (n1,n2,2)

        # base periodic kernel (no outputscale — ScaleKernel adds it)
        ell2 = (ell ** 2).to(X1.dtype)                   # (2,)
        sqdist = 2.0 * (torch.sin(u) ** 2 / ell2).sum(dim=-1)  # (n1,n2)
        Kbase = torch.exp(-sqdist)                        # (n1,n2)

        n1, n2 = Kbase.shape
        c1m = c1.view(-1, 1).expand(n1, n2)              # (n1,n2)
        c2m = c2.view(1, -1).expand(n1, n2)              # (n1,n2)

        ca  = c_coeff[c1m]                               # (n1,n2)
        cb  = c_coeff[c2m]                               # (n1,n2)
        pa  = p[c1m]
        sin2u_a = sin2u.gather(-1, c1m.unsqueeze(-1)).squeeze(-1)   # (n1,n2)
        sin2u_b = sin2u.gather(-1, c2m.unsqueeze(-1)).squeeze(-1)
        cos2u_a = cos2u.gather(-1, c1m.unsqueeze(-1)).squeeze(-1)

        same = (c1m == c2m)

        # same-component term
        # k * [4*pi^2/(p_a^2*l_a^2)*cos(2u_a) - (ca^2)*sin^2(2u_a)]
        # note: 4*pi^2/(p_a^2*l_a^2) = ca * (2*pi/p_a)
        K_same = Kbase * (ca * (2.0 * torch.pi / pa) * cos2u_a
                          - ca ** 2 * sin2u_a ** 2)
        # cross-component term
        K_cross = -Kbase * ca * cb * sin2u_a * sin2u_b

        return torch.where(same, K_same, K_cross)


class PotentialGradGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = kernel

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x),
            self.covar_module(x))


def stack_force_training_data(X_fx, y_fx, X_fy, y_fy, y_is_force=True):
    """
    Build one dataset with augmented input [x,y,comp] and targets as derivatives of A.
    If y_is_force=True (ABF forces), we convert to derivatives of A:
      dA/dx = -Fx, dA/dy = -Fy
    """
    device, dtype = X_fx.device, X_fx.dtype
    Nfx, Nfy = X_fx.shape[0], X_fy.shape[0]

    comp_x = torch.zeros(Nfx, 1, device=device, dtype=dtype)
    comp_y = torch.ones (Nfy, 1, device=device, dtype=dtype)

    train_x = torch.cat([
        torch.cat([X_fx, comp_x], dim=1),
        torch.cat([X_fy, comp_y], dim=1),
    ], dim=0)

    if y_is_force:
        train_y = torch.cat([-y_fx, -y_fy], dim=0)  # derivatives of A
    else:
        train_y = torch.cat([y_fx, y_fy], dim=0)

    return train_x, train_y


def make_aug_test(X_test_xy, comp_value: float):
    """
    X_test_xy: (N,2) tensor
    returns: (N,3) tensor [x,y,comp]
    """
    comp = torch.full((X_test_xy.shape[0], 1), comp_value, dtype=X_test_xy.dtype, device=X_test_xy.device)
    return torch.cat([X_test_xy, comp], dim=1)


def smart_subsample(rows, cols, hist, max_samples=5000, min_count_threshold=5):
    """
    Smart subsampling for GP training on 2D binned force data.
    
    Args:
        rows, cols: indices from np.nonzero(hist)
        hist: 2D histogram array
        max_samples: target number of training points
        min_count_threshold: minimum count to consider reliable
    """
    import numpy as np
    
    num_points = len(rows)
    
    if num_points <= max_samples:
        return rows, cols  # No subsampling needed
    
    # ============ Strategy 1: Prioritize High-Count Bins ============
    counts = hist[rows, cols]
    
    # Create sampling weights: favor high-count bins but ensure coverage
    weights = np.sqrt(counts)  # sqrt to balance between uniform and count-proportional
    weights = weights / weights.sum()
    
    # ============ Strategy 2: Ensure Spatial Coverage ============
    # Add stratified component: divide grid into regions
    n_strata_x = min(10, len(np.unique(rows)))
    n_strata_y = min(10, len(np.unique(cols)))
    
    row_bins = np.digitize(rows, np.linspace(rows.min(), rows.max(), n_strata_x))
    col_bins = np.digitize(cols, np.linspace(cols.min(), cols.max(), n_strata_y))
    strata = row_bins * n_strata_y + col_bins
    
    # Sample proportionally from each stratum
    samples_per_stratum = max_samples // len(np.unique(strata))
    
    selected_idx = []
    for stratum in np.unique(strata):
        stratum_mask = (strata == stratum)
        stratum_idx = np.where(stratum_mask)[0]
        
        if len(stratum_idx) == 0:
            continue
            
        # Sample from this stratum using count weights
        n_sample = min(samples_per_stratum, len(stratum_idx))
        stratum_weights = weights[stratum_idx]
        stratum_weights = stratum_weights / stratum_weights.sum()
        
        selected = np.random.choice(
            stratum_idx, 
            size=n_sample, 
            replace=False, 
            p=stratum_weights
        )
        selected_idx.extend(selected)
    
    # If we're under budget, add more high-count samples
    if len(selected_idx) < max_samples:
        remaining = max_samples - len(selected_idx)
        unselected = np.setdiff1d(np.arange(num_points), selected_idx)
        
        if len(unselected) > 0:
            remaining_weights = weights[unselected]
            remaining_weights = remaining_weights / remaining_weights.sum()
            
            additional = np.random.choice(
                unselected,
                size=min(remaining, len(unselected)),
                replace=False,
                p=remaining_weights
            )
            selected_idx.extend(additional)
    
    selected_idx = np.array(selected_idx)
    return rows[selected_idx], cols[selected_idx]


from scipy.fft import fft2, ifft2

def predict_energy_from_mean_forces(dx, dy, fx, fy, periodic=True):
    """
    Integrate 2D mean forces to free energy via Poisson equation.
    
    Parameters:
        fx, fy : 2D arrays of mean forces (shape: Ny x Nx)
        dx, dy : grid spacings along x and y
        periodic: use periodic boundary conditions (FFT-based)
    
    Returns:
        A : 2D free energy surface (same shape as fx, fy)
    """
    ny, nx = fx.shape
    
    # Handle case where meshgrids are passed instead of spacings
    if np.ndim(dx) == 2:
        dx = dx[0, 1] - dx[0, 0]  # extract scalar from meshgrid
    if np.ndim(dy) == 2:
        dy = dy[1, 0] - dy[0, 0]  # extract scalar from meshgrid
    
    dfx_dx = np.gradient(fx, dx, axis=1)
    dfy_dy = np.gradient(fy, dy, axis=0)
    
    # Step 1: Compute divergence of the force field
    # div_f = dfx/dx + dfy/dy
    dfx_dx = np.gradient(fx, dx, axis=1)  # derivative along x (columns)
    dfy_dy = np.gradient(fy, dy, axis=0)  # derivative along y (rows)
    rhs = -(dfx_dx + dfy_dy)  # right-hand side: -div(f)
    
    # Step 2: Solve Laplacian(A) = rhs using FFT (periodic BCs)
    # Eigenvalues of the discrete Laplacian in Fourier space
    kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky)
    
    # Laplacian in Fourier space: -(kx^2 + ky^2)
    denom = -(KX**2 + KY**2)
    denom[0, 0] = 1.0  # avoid division by zero (sets the constant/reference)
    
    # Solve in Fourier space
    rhs_hat = fft2(rhs)
    A_hat = rhs_hat / denom
    A_hat[0, 0] = 0.0  # free energy is defined up to a constant
    
    A = -1*np.real(ifft2(A_hat))
    
    # Shift so minimum is zero
    A -= A.min()
    
    return A


def predict_energy_from_log(log, center="min"):
    """Wrapper: compute energy from `log` object fields.
    Expects `log.x_cv_grid`, `log.y_cv_grid`, `log.xmf_array_mf`, `log.ymf_array_mf`.
    """
    return predict_energy_from_mean_forces(log.x_cv_grid, log.y_cv_grid,
                                           log.xmf_array_mf, log.ymf_array_mf)


def periodic_embed(X_deg):
    """(phi, psi) in degrees → 4D Cartesian embedding that respects periodicity"""
    X_rad = X_deg * (torch.pi / 180.0)
    return torch.stack([
        torch.cos(X_rad[:, 0]), torch.sin(X_rad[:, 0]),  # phi
        torch.cos(X_rad[:, 1]), torch.sin(X_rad[:, 1]),  # psi
    ], dim=-1)


def transform_derivatives(X_deg, dF_dphi, dF_dpsi):
    """Transform force observations to the embedded space.
    
    If f(phi,psi) = g(cos(phi), sin(phi), cos(psi), sin(psi)), then:
      dF/dphi = -sin(phi) * dg/d(cos_phi) + cos(phi) * dg/d(sin_phi)
      dF/dpsi = -sin(psi) * dg/d(cos_psi) + cos(psi) * dg/d(sin_psi)
    
    We have dF/dphi and dF/dpsi (observed forces) and need to solve
    for the 4D embedded-space derivatives dg/d(x_i).
    """
    X_rad = X_deg * (torch.pi / 180.0)
    
    # Jacobian J (2×4): maps embedded derivatives → angle derivatives
    # Each row has only 2 nonzero entries (block diagonal)
    sp, cp = torch.sin(X_rad[:, 0]), torch.cos(X_rad[:, 0])
    ss, cs = torch.sin(X_rad[:, 1]), torch.cos(X_rad[:, 1])
    
    # dF/dphi = -sin(phi) * dg/dc_phi + cos(phi) * dg/ds_phi
    # dF/dpsi = -sin(psi) * dg/dc_psi + cos(psi) * dg/ds_psi
    # Solve the 2×2 blocks (each is a rotation matrix → inverse = transpose)
    # J_phi = [-sin(phi), cos(phi)]  →  J_phi^T = J_phi^{-1}
    dg_dc_phi = -sp * dF_dphi  # * (pi/180) if forces are per-degree
    dg_ds_phi =  cp * dF_dphi
    dg_dc_psi = -ss * dF_dpsi
    dg_ds_psi =  cs * dF_dpsi
    
    return torch.stack([dg_dc_phi, dg_ds_phi, dg_dc_psi, dg_ds_psi], dim=-1)


def gp_model2D_gpder(log, psi_last, phi_last, *, debug_gp=True, save_plots=True,
                     num_samples=10, train_iters_cold=500, train_iters_warm=200,
                     use_fixed_noise=False, use_abf_bias=False,
                     use_exploration_bias=True):   # <--- ADD HERE
    """
    Train a derivative-observation GP on ABF mean forces, reconstruct the
    free-energy surface, and compute an uncertainty-driven exploration bias.

    Parameters
    ----------
    log : LogObject
        CV grids, mean forces, histogram data.
    psi_last, phi_last : float
        Current CV coordinates of the walker.
    debug_gp : bool
        Verbose diagnostic output.
    save_plots : bool
        Persist SVG visualisations.
    num_samples : int
        Posterior gradient samples for energy reconstruction.
    train_iters_cold / train_iters_warm : int
        Adam iterations for cold / warm start.
    use_fixed_noise : bool
        If True, likelihood noise is entirely count-based (no learned term).
    """
    from pathlib import Path

    print("Entering GP model")
    job = format(log.iter + 1, "04d")
    grid_shape = log.x_cv_grid.shape

    # ============ STEP 1: Data Preparation ============
    limit = 100

    dampening = np.minimum(1, log.hist / limit)
    xmf_damped = dampening * log.xmf_array_mf
    ymf_damped = dampening * log.ymf_array_mf

    rows, cols = np.nonzero(log.hist)
    rows_sub, cols_sub = smart_subsample(rows, cols, log.hist, max_samples=1200)

    # Raw forces as targets; dampening enters via heteroscedastic noise instead
    xnonzero_mf = log.xmf_array_mf[rows_sub, cols_sub]
    ynonzero_mf = log.ymf_array_mf[rows_sub, cols_sub]
    xcv_data = log.x_cv_grid[rows_sub, cols_sub]
    ycv_data = log.y_cv_grid[rows_sub, cols_sub]

    # Low-count bins → high noise so GP trusts them less
    damp_sub = dampening[rows_sub, cols_sub].clip(1e-3, 1.0)

    frac_nonzero = rows.size / log.hist.size

    if debug_gp:
        print(f"\n=== ITERATION {job} DIAGNOSTICS ===")
        print(f"Fraction nonzero: {frac_nonzero:.3f}")
        print(f"Hist counts: [{log.hist.min():.0f}, {log.hist.max():.0f}]")
        print(f"Training pts (sub/nonzero): {len(rows_sub)} / {rows.size}")
        for label, arr in [("Fx", xnonzero_mf), ("Fy", ynonzero_mf)]:
            print(f"  {label}: [{arr.min():.4f}, {arr.max():.4f}] "
                  f"mean={arr.mean():.4f} std={arr.std():.4f}")
        print(f"  Damped Fx: [{xmf_damped.min():.4f}, {xmf_damped.max():.4f}]")

    # Build training tensors
    X_fx, y_fx = make_training_data(xcv_data, ycv_data, xnonzero_mf)
    X_fy, y_fy = make_training_data(xcv_data, ycv_data, ynonzero_mf)
    train_x, train_y = stack_force_training_data(X_fx, y_fx, X_fy, y_fy, y_is_force=True)

    # ============ STEP 2: Kernel and Model Setup ============

    cold_start = (int(job) <= 1) or (int(job) % 1000 == 0)

    # Compute domain extent (used to set the known period)
    domain_extent_x = np.ptp(log.x_cv_grid)
    domain_extent_y = np.ptp(log.y_cv_grid)

    # --- Build periodic derivative kernel (analytic d²k/dx_a dx_b) ---
    os_min, os_max = 0.001, 50.0

    # base_kernel = gpytorch.kernels.PeriodicKernel(ard_num_dims=2)
    base_kernel = PeriodicDerivativeKernel2D(ard_num_dims=2)

    # Fix the period to the known domain extent (e.g. 360° for dihedral angles).
    # If your CVs aren't periodic over exactly this range, learn it instead.
    with torch.no_grad():
        base_kernel.period_length = torch.tensor([domain_extent_x, domain_extent_y])
    base_kernel.raw_period_length.requires_grad_(False)  # period is known, not learned

    # Lengthscale is LEFT LEARNABLE — controls smoothness within each period

    kernel = gpytorch.kernels.ScaleKernel(
        base_kernel,
        outputscale_constraint=gpytorch.constraints.Interval(os_min, os_max))

    # --- Model + likelihood (noise + lengthscale + outputscale all learned) ---
    likelihood = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.Positive())
    likelihood.register_prior(
        "noise_prior",
        gpytorch.priors.GammaPrior(concentration=2.0, rate=10.0),
        "noise")
    
    likelihood.noise = 0.01
    model = PotentialGradGP(train_x, train_y, likelihood, kernel)

    if not cold_start:
        ckpt = torch.load(
            f"{log.dirtag}/alad_wats/gp_checkpoint.pth", weights_only=True
        )
        model.load_state_dict(ckpt["model"], strict=False)
        model.set_train_data(inputs=train_x, targets=train_y, strict=False)
        num_iters = train_iters_warm
    else:
        num_iters = train_iters_cold

    # ============ STEP 3: Training ============
    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iters, eta_min=1e-4)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    log_every = max(1, num_iters // 10)

    for i in range(num_iters):
        optimizer.zero_grad()
        loss = -mll(model(train_x), train_y)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if debug_gp and i % log_every == 0:
            ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
            os_val = model.covar_module.outputscale.item()
            nz = likelihood.noise.mean().item()
            print(f"  Iter {i}/{num_iters}: loss={loss.item():.4f}, "
                  f"ls={ls}, os={os_val:.4f}, noise={nz:.4e}")

    if debug_gp:
        with torch.no_grad():
            ls = model.covar_module.base_kernel.lengthscale.cpu().numpy().ravel()
            os_val = model.covar_module.outputscale.item()
            nz_val = likelihood.noise.mean().item()
            print(f"\nFinal hyperparameters:")
            print(f"  lengthscales: {ls}")
            print(f"  outputscale:  {os_val:.6f}")
            print(f"  noise:        {nz_val:.4e}")

            model.eval(); likelihood.eval()
            pred_mean = model(train_x).mean.cpu().numpy()
            true_vals = train_y.cpu().numpy()
            residuals = pred_mean - true_vals
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((true_vals - true_vals.mean()) ** 2)
            r2 = 1.0 - ss_res / (ss_tot + 1e-12)
            rmse = np.sqrt(np.mean(residuals ** 2))
            print(f"\n  R²   = {r2:.4f}  (<0.9 → underfitting)")
            print(f"  RMSE = {rmse:.4f}")
            print(f"  Pred  range: [{pred_mean.min():.4f}, {pred_mean.max():.4f}]")
            print(f"  True  range: [{true_vals.min():.4f}, {true_vals.max():.4f}]")
            model.train(); likelihood.train()

    # ============ STEP 4: Batched Posterior Prediction ============
    model.eval()
    likelihood.eval()

    X_test = torch.tensor(
        np.stack([log.x_cv_grid.ravel(), log.y_cv_grid.ravel()], axis=-1),
        dtype=torch.float32)
    n_test = len(X_test)
    batch_size = 1000

    # Accumulators keyed by output name
    parts = {k: [] for k in [
        "dAdx_m", "dAdx_s", "dAdy_m", "dAdy_s", "dAdx_samp", "dAdy_samp"]}

    with torch.no_grad(), gpytorch.settings.fast_pred_var(), \
         gpytorch.settings.fast_pred_samples():
        for i0 in range(0, n_test, batch_size):
            batch = X_test[i0 : i0 + batch_size]
            post_dx = model(make_aug_test(batch, 0.0))
            post_dy = model(make_aug_test(batch, 1.0))

            parts["dAdx_m"].append(post_dx.mean.cpu())
            parts["dAdx_s"].append(post_dx.stddev.cpu())
            parts["dAdy_m"].append(post_dy.mean.cpu())
            parts["dAdy_s"].append(post_dy.stddev.cpu())
            parts["dAdx_samp"].append(
                post_dx.rsample(torch.Size([num_samples])).cpu())
            parts["dAdy_samp"].append(
                post_dy.rsample(torch.Size([num_samples])).cpu())

    # Concatenate, convert to numpy, reshape
    def _to_grid(key, is_samples=False):
        t = torch.cat(parts[key], dim=1 if is_samples else 0).numpy()
        return t.reshape((num_samples, *grid_shape)) if is_samples else t.reshape(grid_shape)

    dAdx_samples = _to_grid("dAdx_samp", is_samples=True)
    dAdy_samples = _to_grid("dAdy_samp", is_samples=True)

    fx_mean_grid = -_to_grid("dAdx_m")   # F = -∇A
    fy_mean_grid = -_to_grid("dAdy_m")
    fx_std_grid  =  _to_grid("dAdx_s")   # std unaffected by sign
    fy_std_grid  =  _to_grid("dAdy_s")

    del X_test, parts
    gc.collect()

    # ============ STEP 5: Exploration Bias ============
    if use_exploration_bias:
        (fx_push, fy_push, exploration_bias_energy, U_push,
         phi_max, psi_max, i_max, j_max,
         masked_uncertainty, uncertainty_grid,
         exploration_scale) = compute_exploration_push(
            fx_std_grid, fy_std_grid,
            log.x_cv_grid, log.y_cv_grid, job, frac_nonzero)

        print(f"Exploration scale: {exploration_scale}")
        print(f"Exploration energy range: "
              f"[{exploration_bias_energy.min():.4f}, {exploration_bias_energy.max():.4f}]")
        print(f"Max-uncertainty target: phi={phi_max}, psi={psi_max}")
        print(f"Last CVs: psi={psi_last}  phi={phi_last}")
    else:
        fx_push = np.zeros_like(fx_mean_grid)
        fy_push = np.zeros_like(fy_mean_grid)
        exploration_bias_energy = np.zeros_like(fx_mean_grid)
        exploration_scale = 0.0
        uncertainty_grid = np.sqrt(fx_std_grid**2 + fy_std_grid**2)
        print("Exploration bias disabled (push forces = 0)")

    # ============ STEP 6: Energy Reconstruction ============
    with torch.no_grad():
        ell    = model.covar_module.base_kernel.lengthscale.cpu().numpy().ravel()
        chi2   = float(model.covar_module.outputscale.cpu())
        period = model.covar_module.base_kernel.period_length.cpu().numpy().ravel()

    noise_for_recon = 0.5 * (fx_std_grid ** 2 + fy_std_grid ** 2)
    samples_phys = np.stack([dAdx_samples, dAdy_samples], axis=1)  # (S, 2, ny, nx)

    print(f"Learned hyperparameters — ell: {ell}  period: {period}  chi²: {chi2}")

    _, A_mean, A_std, _ = reconstruct_A_from_gp_samples_2D_no_norm_fast(
        xv=log.x_cv_grid, yv=log.y_cv_grid,
        samples_phys=samples_phys,
        ell=ell, chi2=chi2, period=period,
        noise=noise_for_recon, noise_is_std=False,
        y_is_force=False, jitter_rel=1e-10, center="min")

    # ============ STEP 7: Save Checkpoint & Data ============
    ckpt = {"model": model.state_dict()}
    if not use_fixed_noise:
        ckpt["likelihood"] = likelihood.state_dict()
    torch.save(ckpt, f"{log.dirtag}/alad_wats/gp_checkpoint.pth")
    print("Saved GP checkpoint")

    fx_total = fx_mean_grid + fx_push
    fy_total = fy_mean_grid + fy_push

    data_dir = Path(log.dirtag) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    x_flat = log.x_cv_grid.ravel()
    y_flat = log.y_cv_grid.ravel()
    grid_header = f"shape: {grid_shape[1]} {grid_shape[0]}"

    with open(data_dir / "hp.txt", "a") as f:
        f.write(f"{ell} {chi2} {np.mean(A_std)} {frac_nonzero}\n")

    np.savetxt(
        data_dir / f"data_summary{job}.txt",
        np.column_stack((x_flat, y_flat, A_mean.ravel(),
                         A_std.ravel(), log.hist.ravel())),
        fmt="%.6f",
        header=f"{grid_header} x_cv y_cv A_mean A_std hist", comments="")

    A_damped = predict_energy_from_mean_forces(
        log.x_cv_grid, log.y_cv_grid, xmf_damped, ymf_damped)
    np.savetxt(
        data_dir / f"data_summary_damped{job}.txt",
        np.column_stack((x_flat, y_flat, A_damped.ravel(), log.hist.ravel())),
        fmt="%.6f",
        header=f"{grid_header} x_cv y_cv A_damped hist", comments="")

    A_mf = predict_energy_from_mean_forces(
        log.x_cv_grid, log.y_cv_grid, log.xmf_array_mf, log.ymf_array_mf)
    np.savetxt(
        data_dir / f"data_summary_mf{job}.txt",
        np.column_stack((x_flat, y_flat, A_mf.ravel(), log.hist.ravel())),
        fmt="%.6f",
        header=f"{grid_header} x_cv y_cv A_mf hist", comments="")

    if use_abf_bias:
        fx_out = xmf_damped
        fy_out = ymf_damped
        bias_label = "ABF mean forces"
    else:
        fx_out = fx_total
        fy_out = fy_total
        bias_label = "GP predicted"
    print(f"Writing model_data.txt using {bias_label} forces")
    np.savetxt(
        f"{log.dirtag}/alad_wats/model_data.txt",
        np.column_stack((x_flat, y_flat, fx_out.ravel(), fy_out.ravel())),
        fmt="%.6f",
        header="x_cv y_cv fx_pred fy_pred", comments="")

    # ============ STEP 8: Plots ============
    if save_plots:
        energy_colors = ["#4E96C4", "#7BCAB9", "#F2F3B8", "#F3B27C", "#EC7A84"]
        energy_cmap = LinearSegmentedColormap.from_list("energy_cmap", energy_colors)

        # --- Force comparison (2×2): ABF raw vs GP ---
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for ax, field, title in [
            (axes[0, 0], log.xmf_array_mf, "ABF Fx (raw)"),
            (axes[0, 1], fx_mean_grid,      "GP Fx"),
            (axes[1, 0], log.ymf_array_mf,  "ABF Fy (raw)"),
            (axes[1, 1], fy_mean_grid,       "GP Fy"),
        ]:
            cf = ax.contourf(log.x_cv_grid, log.y_cv_grid, field,
                             levels=30, cmap="coolwarm")
            fig.colorbar(cf, ax=ax)
            ax.set_title(title)
        for ax in (axes[0, 1], axes[1, 1]):
            ax.scatter(xcv_data, ycv_data, c="k", s=4, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{log.dirtag}/gp_results{job}.svg")
        plt.close(fig)

        # --- Sampling histogram ---
        rows_new, cols_new = np.nonzero(log.hist_new)
        fig, ax = plt.subplots()
        cf = ax.contourf(log.x_cv_grid, log.y_cv_grid, log.hist,
                         levels=30, cmap="Blues")
        ax.contour(log.x_cv_grid, log.y_cv_grid, log.hist,
                   levels=30, colors="k", linewidths=0.8, alpha=0.4)
        ax.scatter(xcv_data, ycv_data, c="k", s=4, alpha=0.3, label="Old data")
        ax.scatter(log.x_cv_grid[rows_new, cols_new],
                   log.y_cv_grid[rows_new, cols_new],
                   c="r", s=4, alpha=0.5, label="New data")
        ax.set_title("Sampling histogram")
        ax.set_xlabel("x"); ax.set_ylabel("y")
        cax = inset_axes(ax, width="3%", height="100%", loc="right", borderpad=-1.5)
        fig.colorbar(cf, cax=cax)
        ax.set_aspect("equal")
        plt.savefig(f"{log.dirtag}/hist{job}.svg")
        plt.close(fig)

        # --- Exploration bias energy ---
        # fig, ax = plt.subplots(figsize=(6, 5))
        # cf = ax.contourf(log.x_cv_grid, log.y_cv_grid, exploration_bias_energy,
        #                  levels=25, cmap=energy_cmap)
        # ax.contour(log.x_cv_grid, log.y_cv_grid, exploration_bias_energy,
        #            levels=25, colors="k", linewidths=0.8, alpha=0.4)
        # ax.scatter(phi_max, psi_max, color="red", s=100, marker="*",
        #            edgecolors="black", linewidths=1.5, label="Max uncertainty")
        # fig.colorbar(cf, ax=ax)
        # ax.legend()
        # ax.set_title(f"Exploration bias energy (iter {job})")
        # ax.set_xlabel("phi"); ax.set_ylabel("psi")
        # plt.savefig(f"{log.dirtag}/exploration_bias_energy{job}.svg")
        # plt.close(fig)

        # # --- Push-direction quiver ---
        # fig, ax = plt.subplots(figsize=(6, 5))
        # cf = ax.contourf(log.x_cv_grid, log.y_cv_grid, uncertainty_grid,
        #                  levels=20, cmap="Reds", alpha=0.7)
        # ax.contour(log.x_cv_grid, log.y_cv_grid, uncertainty_grid,
        #            levels=20, colors="k", linewidths=0.6, alpha=0.4)
        # cbar = fig.colorbar(cf, ax=ax, pad=0.1)
        # cbar.set_label("Force uncertainty (std)")

        # dphi = phi_max - phi_last
        # dpsi = psi_max - psi_last
        # ax.quiver(phi_last, psi_last, dphi, dpsi,
        #           color="black", angles="xy", scale_units="xy", scale=1,
        #           width=0.012, headwidth=4, headlength=5, headaxislength=4.5,
        #           label=f"Push direction (Δ={np.hypot(dphi, dpsi):.1f}°)")
        # ax.scatter(phi_max, psi_max, color="red", s=100, marker="*",
        #            edgecolors="black", linewidths=1.5, label="Target", zorder=5)
        # ax.scatter(phi_last, psi_last, color="green", s=100, marker="o",
        #            edgecolors="black", linewidths=1.5, label="Current", zorder=5)
        # ax.set_title("Exploration push field")
        # ax.set_xlabel("phi"); ax.set_ylabel("psi")
        # ax.set_xlim(log.x_cv_grid.min(), log.x_cv_grid.max())
        # ax.set_ylim(log.y_cv_grid.min(), log.y_cv_grid.max())
        # ax.set_aspect("auto")
        # plt.tight_layout()
        # plt.savefig(f"{log.dirtag}/push_forces_quiver{job}.svg", dpi=150,
        #             bbox_inches="tight")
        # plt.close(fig)

        # --- GPR energy surface ---
        fig, ax = plt.subplots(figsize=(6, 5))
        cf = ax.contourf(log.x_cv_grid, log.y_cv_grid, A_mean,
                         levels=25, cmap=energy_cmap)
        ax.contour(log.x_cv_grid, log.y_cv_grid, A_mean,
                   levels=25, colors="k", linewidths=0.8, alpha=0.4)
        fig.colorbar(cf, ax=ax)
        ax.set_title(f"GPR energy surface (iter {job})")
        ax.set_xlabel("phi"); ax.set_ylabel("psi")
        plt.savefig(f"{log.dirtag}/energy_num{job}.svg", dpi=150)
        plt.close(fig)

        # --- GPR vs dampened-trapz side-by-side ---
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5))
        for ax, field, title in [
            (ax_l, A_mean,   f"GPR energy (iter {job})"),
            (ax_r, A_damped, f"Dampened trapz energy (iter {job})"),
        ]:
            cf = ax.contourf(log.x_cv_grid, log.y_cv_grid, field,
                             levels=25, cmap=energy_cmap)
            ax.contour(log.x_cv_grid, log.y_cv_grid, field,
                       levels=25, colors="k", linewidths=0.6, alpha=0.4)
            fig.colorbar(cf, ax=ax)
            ax.set_title(title)
            ax.set_xlabel("phi"); ax.set_ylabel("psi")
        plt.tight_layout()
        plt.savefig(f"{log.dirtag}/energy_comparison{job}.svg", dpi=150)
        plt.close(fig)

        # --- Total energy (free + exploration) ---
        A_total = A_mean + exploration_bias_energy
        fig, ax = plt.subplots(figsize=(6, 5))
        cf = ax.contourf(log.x_cv_grid, log.y_cv_grid, A_total,
                         levels=25, cmap=energy_cmap)
        ax.contour(log.x_cv_grid, log.y_cv_grid, A_total,
                   levels=25, colors="k", linewidths=0.8, alpha=0.4)
        # ax.scatter(phi_max, psi_max, color="red", s=100, marker="*",
        #            edgecolors="black", linewidths=1.5, label="Max uncertainty")
        # ax.scatter(phi_last, psi_last, color="green", s=80,
        #            label="Current location")
        fig.colorbar(cf, ax=ax)
        ax.legend()
        ax.set_title(f"Total energy (free + exploration) (iter {job})")
        ax.set_xlabel("phi"); ax.set_ylabel("psi")
        plt.savefig(f"{log.dirtag}/energy_total{job}.svg", dpi=150)
        plt.close(fig)

    # ============ Cleanup ============
    del model, likelihood
    gc.collect()

    if debug_gp:
        print("Workflow complete.\n")
