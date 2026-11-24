import torch
from torch.utils.data import random_split, DataLoader
import gpytorch
import math
from matplotlib import pyplot as plt
import matplotlib.ticker as tck
import numpy as np
# import jax
# import jax.numpy as jnp
import os
import pickle
import scipy.linalg as la

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, LogNorm, PowerNorm





from gpytorch.constraints import Interval
class ForceGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel 
        

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    

def make_training_data(x_grid, y_grid, f_grid):
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
        y_arr = np.asarray(y_grid)
        X_np = np.stack([x_arr.ravel(), y_arr.ravel()], axis=-1)  # [N, 2]
        y_np = f_arr.ravel()                                      # [N]

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)
    return X, y


def plot_data(train_x_torch, train_y, test_x, mean, loggers, iteration, tot_counts):
    plot_dir = "./figs/gpr_plots"
    logger = loggers[0]  # Use the first logger as a template
    # Initialize plot
    f, ax = plt.subplots() # figsize=(8, 6)
    # plotting histograms
    ax2 = ax.twinx()
    bin_edges = logger.cv # test_x.numpy()

    ax2.hist(logger.cv, bins=bin_edges, weights=tot_counts, color='dimgrey', alpha=0.75, zorder=1) # 'firebrick'
    #EDITED HERE FOR MULTIWALKER
    for i in range(len(loggers)):
        plot_logger = loggers[i]
        ax2.hist(plot_logger.cv, bins=bin_edges, weights=plot_logger.hist_new, alpha=0.75, zorder=2)
    # Use a log scale for the secondary y-axis and set a reasonable top limit
    ax2.set_yscale('log')
    ymax = max(np.max(tot_counts), 1) * 1.2
    ax2.set_ylim(top=ymax)

    # Plot training data as black stars
    ax.plot(train_x_torch.numpy(), train_y.numpy(), 'k*', zorder=10)

    print('mean forces for gpr', train_y.numpy() )
    # Plot predictive means as blue line
    ax.plot(test_x, mean, 'b', zorder=10)
    # Shade between the lower and upper confidence bounds
    # ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.legend(['Observed Data', 'Mean', 'Confidence'], loc='upper center', bbox_to_anchor=(0.5, -0.15),
          fancybox=True, shadow=True, ncol=5)
    
    step = logger.step
    
    # ax.set_ylabel('Mean Force Prediction (eV/$\t{\AA}$)')
    ax2.set_ylabel('Bin Frequency')
    # ax2.yaxis.set_major_locator(tck.MultipleLocator())
    # ax.set_xlabel('Bond Distance ($\t{\AA}$)')
    ax.set_xlim(min(logger.cv), max(logger.cv))
    ax.set_title('GPR sampling, iteration ' + str(iteration))
    plt.tight_layout()
    # plt.savefig(logger.dirtag + '/' + logger.tag + '_gpr_'+str(step)+'.png', format='png') #, transparent=True) #, bbox_inches="tight") #
    plt.savefig(f"{plot_dir}/" + '_gpr_'+str(iteration)+'.png', format='png')
    # plt.show()
    plt.close(f)

    # reset the most recent historgram 
    logger.hist_new = [0 for i in range(logger.size)]


def plot_all_mf_data(train_x_torch, train_y, test_x, mean, loggers, iteration):
    plot_dir = "./figs/gpr_plots"
    logger = loggers[0]  # Use the first logger as a template
    # Initialize plot
    # add all of the data together
    tot_counts = np.zeros_like(logger.histogram)
    tot_fsum = np.zeros_like(logger.Fsum)
    for i in range(len(loggers)):
        tot_counts += loggers[i].histogram
        tot_fsum += loggers[i].Fsum

    tot_mf = np.zeros_like(tot_fsum, dtype=float)
    mask = tot_counts > 0
    tot_mf[mask] = tot_fsum[mask] / tot_counts[mask]

    f, ax = plt.subplots() # figsize=(8, 6)
    # plotting histograms
    ax2 = ax.twinx()
    bin_edges = logger.cv # test_x.numpy()
    ax2.hist(logger.cv, bins=bin_edges, weights=tot_counts, color='dimgrey', alpha=0.75, zorder=1) # 'firebrick'
    #EDITED HERE FOR MULTIWALKER
    # for i in range(len(loggers)):
    #     colors = ['r', 'orange', 'y', 'c', 'm', 'g', 'purple']
    #     plot_logger = loggers[i]
    #     ax2.hist(plot_logger.cv, bins=bin_edges, weights=plot_logger.hist_new, alpha=0.75, zorder=2)
    # Use a log scale for the secondary y-axis and set a reasonable top limit
    ax2.set_yscale('log')
    ymax = max(np.max(tot_counts), 1) * 1.2
    ax2.set_ylim(top=ymax)

    # Plot mf as black stars
    ax.plot(logger.cv, tot_mf, 'k*', zorder=10, label='mean_forces')
    # training data
    ax.plot(train_x_torch.numpy(), train_y.numpy(), 'r', zorder=10, label='training data')

    # print('mean forces for gpr', train_y.numpy() )
    # Plot predictive means as blue line
    # ax.plot(test_x, mean, 'b', zorder=10)
    # Shade between the lower and upper confidence bounds
    # ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    
    ax.set_zorder(ax2.get_zorder() + 1)
    ax.patch.set_visible(False)
    
    step = logger.step
    
    # ax.set_ylabel('Mean Force Prediction (eV/$\t{\AA}$)')
    ax2.set_ylabel('Bin Frequency')
    # ax2.yaxis.set_major_locator(tck.MultipleLocator())
    # ax.set_xlabel('Bond Distance ($\t{\AA}$)')
    ax.set_xlim(min(logger.cv), max(logger.cv))
    ax.set_title('GPR sampling, iteration ' + str(iteration))
    plt.tight_layout()
    plt.legend()
    # plt.savefig(logger.dirtag + '/' + logger.tag + '_gpr_'+str(step)+'.png', format='png') #, transparent=True) #, bbox_inches="tight") #
    plt.savefig(f"{plot_dir}/" + '_mf_'+str(iteration)+'.png', format='png')
    # plt.show()
    plt.close(f)

    # reset the most recent historgram 
    logger.hist_new = [0 for i in range(logger.size)]


def plot_energy(cv, A_mean, A_samples, A_std, iteration):
    plot_dir = "./figs/gpr_plots"

    A_trans = (A_mean - min(A_mean))
    samples_trans = np.array([sample - np.min(A_mean) for sample in A_samples])

    f, ax = plt.subplots() # figsize=(8, 6)

    ax.fill_between(cv, A_trans - A_std, 
                A_trans + A_std, alpha=0.2, color='lightpink', label='±2σ', zorder=3)
    
    for i in range(5):
        ax.plot(cv, samples_trans[i], color='k', alpha=0.6, linewidth=0.5, zorder=2)
    ax.plot(cv, A_trans, color='tab:pink', lw=1, label='Mean Free energy', zorder=20)

    ax.set_title('Free energy, iteration ' + str(iteration))
    plt.tight_layout()
    # plt.savefig(logger.dirtag + '/' + logger.tag + '_gpr_'+str(step)+'.png', format='png') #, transparent=True) #, bbox_inches="tight") #
    plt.savefig(f"{plot_dir}/" + '_energy_'+str(iteration)+'.png', format='png')
    # plt.show()
    plt.close(f)



def plot_gp_contours(cv, fx_mean, fy_mean, fx_std, fy_std, iteration,
                     train_fx=None, train_fy=None):
    """Create contour plots for GP predicted forces and their uncertainty.

    All array arguments are 2D grids aligned with cv[0] / cv[1].

    cv:                tuple (X_grid, Y_grid), each shape (nx, ny)
    fx_mean, fy_mean:  2D arrays, GP predicted force components
    fx_std,  fy_std:   2D arrays, GP force standard deviations
    train_fx, train_fy: 2D arrays of training force values (NaN for unvisited bins)
    """
    plot_dir = "./figs/gpr_plots"
    os.makedirs(plot_dir, exist_ok=True)

    X = cv[0]
    Y = cv[1]

    fx_mean = np.asarray(fx_mean)
    fy_mean = np.asarray(fy_mean)
    fx_std  = np.asarray(fx_std)
    fy_std  = np.asarray(fy_std)

    colors = ["#4E96C4", "#7BCAB9", "#F2F3B8", "#F3B27C", "#EC7A84"]
    cmap = LinearSegmentedColormap.from_list("energy_cmap", colors)

    f, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Pre-compute scatter positions and values from non-NaN training grid cells
    if train_fx is not None:
        train_fx = np.asarray(train_fx, dtype=float)
    if train_fy is not None:
        train_fy = np.asarray(train_fy, dtype=float)

    # Use whichever grid is available to find visited bin coordinates
    _ref_grid = train_fx if train_fx is not None else train_fy
    if _ref_grid is not None:
        _mask = ~np.isnan(_ref_grid)
        scat_x  = X[_mask]
        scat_y  = Y[_mask]
        scat_fx = train_fx[_mask] if train_fx is not None else None
        scat_fy = train_fy[_mask] if train_fy is not None else None
    else:
        scat_x = scat_y = scat_fx = scat_fy = None

    _scatter_kw = dict(s=14, edgecolors='k', linewidths=0.3, zorder=5)

    # ── Subplot 1: Training data – Fx ─────────────────────────────────────────
    ax00 = axes[0, 0]
    if train_fx is not None:
        pc0 = ax00.pcolormesh(X, Y, train_fx, cmap=cmap, shading='auto')
        plt.colorbar(pc0, ax=ax00, label='Fx (training)')
        ax00.scatter(scat_x, scat_y, c=scat_fx, cmap=cmap,
                     vmin=np.nanmin(train_fx), vmax=np.nanmax(train_fx), **_scatter_kw)
    else:
        ax00.text(0.5, 0.5, 'No training data', transform=ax00.transAxes,
                  ha='center', va='center')
    ax00.set_xlabel('CV x')
    ax00.set_ylabel('CV y')
    ax00.set_title(f'Training data – Fx (iter {iteration})')

    # ── Subplot 2: Training data – Fy ─────────────────────────────────────────
    ax01 = axes[0, 1]
    if train_fy is not None:
        pc1 = ax01.pcolormesh(X, Y, train_fy, cmap=cmap, shading='auto')
        plt.colorbar(pc1, ax=ax01, label='Fy (training)')
        ax01.scatter(scat_x, scat_y, c=scat_fy, cmap=cmap,
                     vmin=np.nanmin(train_fy), vmax=np.nanmax(train_fy), **_scatter_kw)
    else:
        ax01.text(0.5, 0.5, 'No training data', transform=ax01.transAxes,
                  ha='center', va='center')
    ax01.set_xlabel('CV x')
    ax01.set_ylabel('CV y')
    ax01.set_title(f'Training data – Fy (iter {iteration})')

    # ── Subplot 3: GP prediction – Fx ─────────────────────────────────────────
    ax10 = axes[1, 0]
    cf0 = ax10.contourf(X, Y, fx_mean, levels=25, cmap=cmap)
    plt.colorbar(cf0, ax=ax10, label='Fx (GP mean)')
    if scat_x is not None and scat_fx is not None:
        ax10.scatter(scat_x, scat_y, c=scat_fx, cmap=cmap,
                     vmin=fx_mean.min(), vmax=fx_mean.max(), **_scatter_kw)
    ax10.set_xlabel('CV x')
    ax10.set_ylabel('CV y')
    ax10.set_title(f'GP prediction – Fx (iter {iteration})')

    # ── Subplot 4: GP prediction – Fy ─────────────────────────────────────────
    ax11 = axes[1, 1]
    cf1 = ax11.contourf(X, Y, fy_mean, levels=25, cmap=cmap)
    plt.colorbar(cf1, ax=ax11, label='Fy (GP mean)')
    if scat_x is not None and scat_fy is not None:
        ax11.scatter(scat_x, scat_y, c=scat_fy, cmap=cmap,
                     vmin=fy_mean.min(), vmax=fy_mean.max(), **_scatter_kw)
    ax11.set_xlabel('CV x')
    ax11.set_ylabel('CV y')
    ax11.set_title(f'GP prediction – Fy (iter {iteration})')

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/gpr_contours_{iteration}.png", dpi=150, bbox_inches='tight')
    plt.close(f)


def plot_energy_2d(cv, A_mean, A_std, A_samples, iteration):
    """Contour plot for reconstructed energy surface and its std.

    A_samples may be None or array (S, nx, ny)."""
    plot_dir = "./figs/gpr_plots"
    os.makedirs(plot_dir, exist_ok=True)

    X = cv[0]
    Y = cv[1]

    colors = ["#4E96C4", "#7BCAB9", "#F2F3B8", "#F3B27C", "#EC7A84"]
    cmap = LinearSegmentedColormap.from_list("energy_cmap", colors)

    f, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    cf = ax.contourf(X, Y, A_mean, levels=30, cmap=cmap)
    cs = ax.contour(X, Y, A_mean, levels=30, colors='k', linewidths=0.4, alpha=0.6)

    # overlay a few sample contours if available
    try:
        if A_samples is not None:
            A_samples = np.asarray(A_samples)
            n_samp = min(5, A_samples.shape[0])
            for s in range(n_samp):
                ax.contour(X, Y, A_samples[s], levels=8, colors='k', linewidths=0.5, alpha=0.25)
    except Exception:
        pass

    ax.set_title(f'Reconstructed free energy (iter {iteration})')
    plt.colorbar(cf, ax=ax)

    ax2 = axes[1]
    cf2 = ax2.contourf(X, Y, A_std, levels=25, cmap='viridis')
    ax2.set_title(f'Free energy std (iter {iteration})')
    plt.colorbar(cf2, ax=ax2)

    plt.tight_layout()
    plt.savefig(f"{plot_dir}/energy_2d_{iteration}.png", dpi=150, bbox_inches='tight')
    plt.close(f)


def plot_exploratory_bias_2d(cv, fx_push, fy_push, exploration_bias_energy, iteration):
    """Plot the exploratory bias energy as a contour and the pushing forces as a quiver plot."""
    plot_dir = "./figs/gpr_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    X = cv[0]
    Y = cv[1]
    
    f, ax = plt.subplots(figsize=(8, 6))
    
    # Contour plot of the exploration bias energy
    cf = ax.contourf(X, Y, exploration_bias_energy, levels=30, cmap='viridis', alpha=0.8)
    plt.colorbar(cf, ax=ax, label='Exploratory Bias Energy (kcal/mol)')
    
    # Downsample grid for readability in quiver plot
    step = 3
    X_q = X[::step, ::step]
    Y_q = Y[::step, ::step]
    U_q = fx_push[::step, ::step]
    V_q = fy_push[::step, ::step]
    
    # Quiver plot for the pushing forces
    ax.quiver(X_q, Y_q, U_q, V_q, color='white', edgecolor='k', linewidth=0.5, zorder=5)
    
    ax.set_title(f'Exploratory Bias (iter {iteration})')
    ax.set_xlabel('CV x')
    ax.set_ylabel('CV y')
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/exploratory_bias_{iteration}.png", dpi=150, bbox_inches='tight')
    plt.close(f)


def plot_total_force_2d(cv, fx_total, fy_total, total_energy, iteration):
    """Plot the total energy as a contour and the total forces as a quiver plot."""
    plot_dir = "./figs/gpr_plots"
    os.makedirs(plot_dir, exist_ok=True)
    
    X = cv[0]
    Y = cv[1]
    
    f, ax = plt.subplots(figsize=(8, 6))
    
    # Contour plot of the total energy
    cf = ax.contourf(X, Y, total_energy, levels=30, cmap='viridis', alpha=0.8)
    plt.colorbar(cf, ax=ax, label='Total Energy (kcal/mol)')
    
    # Downsample grid for readability in quiver plot
    step = 3
    X_q = X[::step, ::step]
    Y_q = Y[::step, ::step]
    U_q = fx_total[::step, ::step]
    V_q = fy_total[::step, ::step]
    
    # Quiver plot for the total forces
    ax.quiver(X_q, Y_q, U_q, V_q, color='white', edgecolor='k', linewidth=0.5, zorder=5)
    
    ax.set_title(f'Total Force and Energy (iter {iteration})')
    ax.set_xlabel('CV x')
    ax.set_ylabel('CV y')
    
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/total_force_{iteration}.png", dpi=150, bbox_inches='tight')
    plt.close(f)


def plot_biglog_histogram_contour(
    big_log,
    iteration,
    plot_dir="./figs/gpr_plots",
    cmap='viridis',
    levels=25,
    color_norm: str = "linear",
    power_gamma: float = 0.5,
):
    """Plot `big_log.histogram` as a contour (2D) and overlay sampled bins as black points.

    Args:
        big_log: object with attributes `histogram` and either `cv` (tuple of X,Y grids)
                 or `cv_x`/`cv_y` (1D arrays used to construct a meshgrid).
        iteration: int or str used for filename/title.
        plot_dir: where to save the figure.
        cmap: colormap for contourf.
        levels: number of contour levels (or sequence of levels). When `color_norm="log"`
            and `levels` is an int, log-spaced levels are used.
        color_norm: "linear" (default), "log" (enhances low counts), or "power" (gamma correction).
        power_gamma: gamma used when `color_norm` is "power" (smaller -> more contrast for low counts).
    """
    os.makedirs(plot_dir, exist_ok=True)

    counts = np.asarray(big_log.histogram)

    # 2D case
    if counts.ndim == 2:
        # Try to read cv grid
        try:
            cv = big_log.cv
            X = np.asarray(cv[0])
            Y = np.asarray(cv[1])
            if X.T.shape == counts.shape:
                X, Y = X.T, Y.T
        except Exception:
            if hasattr(big_log, 'cv_x') and hasattr(big_log, 'cv_y'):
                X, Y = np.meshgrid(np.asarray(big_log.cv_x), np.asarray(big_log.cv_y), indexing='xy')
            else:
                X, Y = np.meshgrid(np.arange(counts.shape[1]), np.arange(counts.shape[0]), indexing='xy')

        f, ax = plt.subplots(figsize=(6, 5))

        counts_to_plot = counts
        norm = None
        levels_to_use = levels

        if color_norm is not None and str(color_norm).lower() != "linear":
            mode = str(color_norm).lower()
            if mode == "log":
                positive = counts[counts > 0]
                if positive.size > 0:
                    vmin = float(np.min(positive))
                    vmax = float(np.max(positive))
                    if vmax > vmin:
                        norm = LogNorm(vmin=vmin, vmax=vmax)
                        counts_to_plot = np.ma.masked_less_equal(counts, 0)
                        if np.isscalar(levels):
                            n_levels = int(levels)
                            if n_levels < 2:
                                n_levels = 2
                            levels_to_use = np.geomspace(vmin, vmax, n_levels)
            elif mode in ("power", "pow", "gamma"):
                vmax = float(np.max(counts))
                if vmax > 0:
                    norm = PowerNorm(gamma=float(power_gamma), vmin=0.0, vmax=vmax)

        cf = ax.contourf(X, Y, counts_to_plot, levels=levels_to_use, cmap=cmap, norm=norm)
        cbar = plt.colorbar(cf, ax=ax, label='Counts')
        if norm is not None and isinstance(norm, LogNorm):
            cbar.locator = tck.LogLocator(base=10)
            cbar.formatter = tck.LogFormatter(base=10)
            cbar.update_ticks()

        rows, cols = np.nonzero(counts)
        if rows.size > 0:
            ax.scatter(X[rows, cols], Y[rows, cols], c='k', s=5, zorder=6)

        ax.set_title(f'big_log histogram (iter {iteration})')
        ax.set_xlabel('CV x')
        ax.set_ylabel('CV y')
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/gpr_histogram_contour_{iteration}.png", dpi=150, bbox_inches='tight')
        plt.close(f)

    else:
        # 1D fallback
        cv = np.asarray(getattr(big_log, 'cv', big_log.cv))
        f, ax = plt.subplots(figsize=(6, 3))
        width = (cv[1] - cv[0]) if len(cv) > 1 else 1.0
        ax.bar(cv, counts, width=width, align='center', color='lightgrey')
        nz = np.nonzero(counts)[0]
        if nz.size > 0:
            ax.scatter(cv[nz], counts[nz], c='k', s=20, zorder=5)
        ax.set_title(f'big_log histogram (1D) (iter {iteration})')
        ax.set_xlabel('CV')
        ax.set_ylabel('Counts')
        plt.tight_layout()
        plt.savefig(f"{plot_dir}/gpr_histogram_1d_{iteration}.png", dpi=150, bbox_inches='tight')
        plt.close(f)


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


class PotentialGradGP(gpytorch.models.ExactGP):
    """
    A GPyTorch ExactGP model for potential energy surfaces.

    This version allows a custom `mean_module` to be passed during initialization.
    If no mean module is provided, it defaults to a `ZeroMean`. This enables
    the use of a prior mean (like one learned from a previous iteration)
    instead of always assuming a zero-mean function.
    """
    def __init__(self, train_x, train_y, likelihood, kernel, mean_module=None):
        super().__init__(train_x, train_y, likelihood)
        
        # If a custom mean_module is passed, use it. 
        # Otherwise, default to the standard ZeroMean.
        if mean_module is None:
            self.mean_module = gpytorch.means.ZeroMean()
        else:
            self.mean_module = mean_module
            
        self.covar_module = kernel

    def forward(self, x):
        # This part doesn't need to change. It will automatically use whatever
        # mean_module and covar_module were set in the constructor.
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def get_loggers():
    """Retrieve all logger objects from the current directory."""
    loggers = []
    logger_dir = './loggers'

    for file in os.listdir(logger_dir):
        if file.startswith('logger_') and file.endswith('.pkl'):
            file_path = os.path.join(logger_dir, file)
            with open(file_path, 'rb') as f:
                log = pickle.load(f)
            loggers.append(log)

    # print(f"Found {len(loggers)} logger(s) for GPR training.")
    return loggers


def stack_mean_forces(loggers):
    """Stack mean forces from multiple loggers into a single array."""
    all_xmean_forces = np.vstack([log.mean_forces for log in loggers])
    
    return all_xmean_forces


def build_gp_training_data_1d(loggers, dim2 = False, min_count=1):
    """
    Combine all loggers into per-bin GP training data.
    Uses histogram counts as weights and a min_count threshold.
    """
    template = loggers[0]
    
    if dim2:
        # 2D case: cv should be a meshgrid or 2D array
        cv = np.asarray(template.cv)      # shape (n_bins_x, n_bins_y, 2) or similar
        
        # Stack mean forces and histograms: (n_loggers, n_bins_x, n_bins_y)
        xforces = np.stack([log.xmean_forces for log in loggers])
        yforces = np.stack([log.ymean_forces for log in loggers])
        counts = np.stack([log.histogram for log in loggers])
        
        # Total samples per bin across loggers
        total_counts = counts.sum(axis=0)    # shape (n_bins_x, n_bins_y)
        
        
        # Only use bins with enough samples
        mask = total_counts >= min_count
        
        # Weighted average force in each bin (using counts as weights) for both x and y
        weighted_xforce_num = (xforces * counts).sum(axis=0)
        weighted_yforce_num = (yforces * counts).sum(axis=0)
        weighted_force_den = np.maximum(total_counts, 1.0)
        mean_xforce_per_bin = weighted_xforce_num / weighted_force_den
        mean_yforce_per_bin = weighted_yforce_num / weighted_force_den
        
        # Reshape cv for training: flatten spatial dimensions
        if cv.ndim == 3 and cv.shape[-1] == 2:
            # cv is already in format (n_bins_x, n_bins_y, 2)
            cv_reshaped = cv.reshape(-1, 2)  # shape (n_bins_x*n_bins_y, 2)
        else:
            # Assume cv needs to be constructed from grid
            raise ValueError("2D cv must be shape (n_bins_x, n_bins_y, 2)")
        
        # Flatten mask and forces
        mask_flat = mask.ravel()
        mean_xforce_flat = mean_xforce_per_bin.ravel()
        mean_yforce_flat = mean_yforce_per_bin.ravel()
        
        # Training inputs and targets for two independent GPs
        x = cv_reshaped[mask_flat]           # shape (n_used_bins, 2) - CV coordinates
        y_x = mean_xforce_flat[mask_flat]    # shape (n_used_bins,) - x-forces
        y_y = mean_yforce_flat[mask_flat]    # shape (n_used_bins,) - y-forces

        return x, y_x, y_y
    
    else:
        # 1D case
        cv = np.asarray(template.cv)          # shape (n_bins,)
        n_bins = len(cv)

        # Stack data from all loggers
        counts = np.stack([log.histogram for log in loggers])   # shape (n_loggers, n_bins)
        forces = np.stack([log.Fsum for log in loggers])        # shape (n_loggers, n_bins)

        # Total samples per bin across loggers
        # Sum across all loggers (axis=0)
        total_counts = counts.sum(axis=0)    # shape (n_bins,)
        total_forces = forces.sum(axis=0)    # shape (n_bins,)

        # Compute mean force per bin (safe division)
        mean_force_per_bin = np.zeros_like(total_forces, dtype=float)
        mask = total_counts > 0
        mean_force_per_bin[mask] = total_forces[mask] / total_counts[mask]

        # Filter bins with enough samples
        final_mask = total_counts >= min_count

        # Training inputs and targets
        x = cv[final_mask].reshape(-1, 1)      # shape (n_used_bins, 1)
        y = mean_force_per_bin[final_mask]     # shape (n_used_bins,)

        # --- DEBUG: build_gp_training_data summary ---
        print(f'[build_gp] n_loggers={len(loggers)}, n_bins={n_bins}, min_count={min_count}')
        print(f'[build_gp] total_counts: min={total_counts.min()}, max={total_counts.max()}, '
              f'nonzero={np.count_nonzero(total_counts)}/{n_bins}')
        print(f'[build_gp] total_forces (Fsum): min={total_forces.min():.6f}, max={total_forces.max():.6f}')
        if np.any(mask):
            print(f'[build_gp] mean_force_per_bin (where sampled): '
                  f'min={mean_force_per_bin[mask].min():.6f}, max={mean_force_per_bin[mask].max():.6f}, '
                  f'mean={mean_force_per_bin[mask].mean():.6f}')
        else:
            print(f'[build_gp] mean_force_per_bin: NO sampled bins yet (all counts=0)')
        print(f'[build_gp] bins passing filter: {final_mask.sum()}/{n_bins}')
        if final_mask.sum() > 0:
            print(f'[build_gp] training x range: [{x.min():.4f}, {x.max():.4f}]')
            print(f'[build_gp] training y (mean force) range: [{y.min():.6f}, {y.max():.6f}]')
        else:
            print(f'[build_gp] training data: EMPTY (no bins pass filter)')
        # Print per-bin detail for first 10 sampled bins
        sampled_idx = np.where(final_mask)[0]
        for idx in sampled_idx[:10]:
            print(f'  bin[{idx}] cv={cv[idx]:.4f}  counts={total_counts[idx]:.0f}  '
                  f'Fsum={total_forces[idx]:.6f}  mf={mean_force_per_bin[idx]:.6f}')
        if len(sampled_idx) > 10:
            print(f'  ... ({len(sampled_idx) - 10} more bins)')

        return x, y, None, total_counts


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


def reconstruct_A_from_gp_samples_2D_no_norm_fast(
    xv, yv,
    samples_phys,   # torch or np array, shape (S, 2, nx, ny): [Fx, Fy]
    ell,            # length-2 array, RBF ARD lengthscales (same units as xv,yv)
    chi2,           # scalar, RBF outputscale (σ_f^2)
    noise,          # noise on forces; same semantics as before
    noise_is_std=True,
    y_is_force=True,
    jitter_rel=1e-10,
    center="min"):
    """
    Reconstruct A(x,y) from multiple GP force samples in one shot (no normalization).

    Inputs
    ------
    xv, yv       : (nx, ny) grids
    samples_phys : (S, 2, nx, ny), GP samples of forces or gradients
                   samples_phys[s, 0] = Fx^(s)(x,y), samples_phys[s, 1] = Fy^(s)(x,y)
    ell          : length-2 array (ℓ_x, ℓ_y), same units as xv,yv
    chi2         : scalar kernel outputscale (σ_f^2)
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
    q = 1.0 / (ell**2)  # (2,)

    if chi2 is None or chi2 <= 0.0:
        # fallback if needed
        chi2 = max(1e-6, np.var(samples_np))

    # ----- kernel K (derivative–derivative) -----
    # dx[i,j,d] = X[i,d] - X[j,d]
    dx = X[:, None, :] - X[None, :, :]                # (N,N,2)
    sqdist = np.sum(dx**2 * q[None, None, :], axis=2) # (N,N)
    Kbase = chi2 * np.exp(-0.5 * sqdist)              # (N,N)

    diagQ = np.diag(q)                                # (2,2)
    term1 = Kbase[:, :, None, None] * diagQ[None, None, :, :]        # (N,N,2,2)
    dxq = dx * q[None, None, :]                       # (N,N,2)
    term2 = Kbase[:, :, None, None] * (dxq[:, :, :, None] * dxq[:, :, None, :])  # (N,N,2,2)
    K_blocks = term1 - term2                          # (N,N,2,2)

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

    # ----- Kstar: derivative–function cross covariances at same grid -----
    Xm = X
    dxm = X[:, None, :] - Xm[None, :, :]                    # (N,M,2), here M=N
    sqdist_star = np.sum(dxm**2 * q[None, None, :], axis=2) # (N,M)
    Kbase_star = chi2 * np.exp(-0.5 * sqdist_star)          # (N,M)
    # ∂/∂x_i,a k(x_i, x_m) = - q[a] * (x_i[a] - x_m[a]) * k
    Kstar_blocks = - Kbase_star[:, :, None] * (dxm * q[None, None, :])  # (N,M,2)

    Kstar = np.empty((ND, Xm.shape[0]), dtype=float)        # (2N, N)
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


def reconstruct_A_from_gp_outputs_1D_no_norm(
    xv,
    fx_pred,
    ell,
    chi2,
    noise,
    jitter_rel=1e-10):
    """
    1D derivative-GPR reconstruction of free energy A(x), WITHOUT normalization.

    Inputs
    ------
    xv       : array shape (nx,) coordinate grid (physical units)
    fx_pred  : array shape (nx,) predicted forces (= -dA/dx, physical units)
    ell      : GP lengthscale (scalar, in SAME units as xv)
    chi2     : GP kernel outputscale (σ_f^2)
    noise    : GP noise variance on forces (scalar, in physical units)
    jitter_rel : relative jitter term for numerical stability

    Returns
    -------
    A_grid   : (nx,) reconstructed free energy (shifted so min==0)
    A_std    : (nx,) posterior standard deviation
    A_trap   : (nx,) trapezoidal integrated free energy (cheap check)
    """

    xv = np.asarray(xv, dtype=float).ravel()
    fx_pred = np.asarray(fx_pred, dtype=float).ravel()
    nx = xv.shape[0]
    N = nx

    # Coordinates in physical units
    X_obs = xv.reshape(-1, 1)   # (N, 1)
    X_test = X_obs.copy()       # same grid

    # Forces are already dA/dx in physical units (actually -dA/dx)
    f_obs = fx_pred.copy()
    f_stack = f_obs.reshape(N)  # (N,)

    # Ensure ell scalar
    ell = float(np.asarray(ell).ravel()[0])

    # Fallback for chi2 if needed
    if chi2 is None or chi2 <= 1e-12:
        chi2 = max(1e-3, np.var(f_obs))

    # --- Build covariance matrix K (N x N) ---
    K = np.zeros((N, N), dtype=float)
    for i in range(N):
        xi = X_obs[i, 0]
        for j in range(N):
            xj = X_obs[j, 0]
            dx = xi - xj
            k_ij = chi2 * np.exp(-0.5 * (dx / ell)**2)
            # second derivative wrt xi and xj in 1D:
            # ∂²k/∂xi∂xj = (1/ell^2 - dx^2/ell^4) * k_ij
            K[i, j] = (1.0 / ell**2 - (dx**2) / (ell**4)) * k_ij

    # Noise variance on forces (already in physical units)
    var_phys = float(noise)
    Sigma = np.diag(np.full(N, var_phys))

    # Add jitter
    jitter = jitter_rel * np.trace(K) / max(1, K.shape[0])
    K_plus = K + Sigma + jitter * np.eye(N)

    # --- Build Kstar (N x M), here M = N same grid ---
    M = N
    Kstar = np.zeros((N, M), dtype=float)
    for i in range(N):
        xi = X_obs[i, 0]
        for m in range(M):
            xm = X_test[m, 0]
            dx = xi - xm
            k_im = chi2 * np.exp(-0.5 * (dx / ell)**2)
            # ∂k/∂xi in 1D = -(dx/ell^2) * k
            Kstar[i, m] = -(dx / ell**2) * k_im

    # Solve linear system for alpha
    try:
        L = la.cholesky(K_plus, lower=True)
        y = la.solve_triangular(L, f_stack, lower=True)
        alpha = la.solve_triangular(L.T, y, lower=False)
    except la.LinAlgError:
        jitter2 = max(1e-8, 1e-6 * np.mean(np.diag(K_plus)))
        L = la.cholesky(K_plus + jitter2 * np.eye(N), lower=True)
        y = la.solve_triangular(L, f_stack, lower=True)
        alpha = la.solve_triangular(L.T, y, lower=False)

    # Posterior mean of A(x)
    A_mean = - (Kstar.T @ alpha)  # (M,)

    # Posterior variance
    v = la.solve_triangular(L, Kstar, lower=True)
    v = la.solve_triangular(L.T, v, lower=False)
    kxx = chi2
    A_var = np.empty(M, dtype=float)
    for m in range(M):
        A_var[m] = kxx - (Kstar[:, m] @ v[:, m])
        if A_var[m] < 0 and A_var[m] > -1e-12:
            A_var[m] = 0.0

    A_grid = A_mean
    A_std = np.sqrt(A_var)

    # --- sanity check: trapezoidal integration of forces ---
    dx_phys = np.mean(np.diff(xv))
    A_trap = np.zeros_like(fx_pred)
    for i in range(1, N):
        A_trap[i] = A_trap[i-1] - 0.5 * (fx_pred[i-1] + fx_pred[i]) * dx_phys
    A_trap -= A_trap.min()

    return A_grid, A_std, A_trap


def gpr_pytorch_1d(iteration, big_log):
    print(f'starting gp model training for iteration {iteration}')
    limit=10
    loggers = get_loggers()
    logger = loggers[0]  # Use the first logger as a template

    # Determine which CV to use
    active_cv = logger.dist_cv if logger.dist_cv is not None else logger.dihedral_cv

    x, y, _, total_counts = build_gp_training_data_1d(loggers)
    # etract data from big_log
    counts = big_log.histogram
    abf_mf = np.zeros_like(big_log.Fsum, dtype=float)
    mask_nz = counts > 0
    abf_mf[mask_nz] = big_log.Fsum[mask_nz] / counts[mask_nz]
    cv = big_log.cv
    # determine dampening factor and dampen data
    dampening = np.minimum(1, counts / limit)
    xmeanforces_damped = dampening * abf_mf
    # remove bins without counts
    rows = np.nonzero(counts)
    xnonzero_mf = xmeanforces_damped[rows]
    xcv_data = cv[rows]
    print(f'there are {len(xcv_data)} training points')

    # --- DEBUG: big_log data going into GP ---
    print(f'[gpr_1d] big_log histogram: nonzero={np.count_nonzero(counts)}/{len(counts)}, '
          f'total_samples={counts.sum():.0f}')
    print(f'[gpr_1d] big_log Fsum range: [{big_log.Fsum.min():.6f}, {big_log.Fsum.max():.6f}]')
    nz_idx = np.where(mask_nz)[0]
    if np.any(mask_nz):
        print(f'[gpr_1d] raw mean force (Fsum/counts) range (sampled bins): '
              f'[{abf_mf[mask_nz].min():.6f}, {abf_mf[mask_nz].max():.6f}]')
        print(f'[gpr_1d] dampening factor range (sampled bins): '
              f'[{dampening[mask_nz].min():.4f}, {dampening[mask_nz].max():.4f}]  (limit={limit})')
        print(f'[gpr_1d] damped mean force range: [{xnonzero_mf.min():.6f}, {xnonzero_mf.max():.6f}]')
        # Per-bin detail
        print(f'[gpr_1d] Per-bin detail (first 15 sampled bins):')
        for idx in nz_idx[:15]:
            print(f'  bin[{idx}] cv={cv[idx]:.4f}  counts={counts[idx]:.0f}  '
                  f'Fsum={big_log.Fsum[idx]:.6f}  raw_mf={abf_mf[idx]:.6f}  '
                  f'damp={dampening[idx]:.4f}  damped_mf={xmeanforces_damped[idx]:.6f}')
        if len(nz_idx) > 15:
            print(f'  ... ({len(nz_idx) - 15} more bins)')
    else:
        print(f'[gpr_1d] NO sampled bins yet — skipping per-bin detail')

    # if x is None:
    #     print("Not enough data to update GP yet.")
    #     return
    
    print('created training arrays')
    X_fx = torch.from_numpy(xcv_data.astype(np.float32))
    y_fx = torch.from_numpy(xnonzero_mf.astype(np.float32))

    # Likelihoods
    lik_fx = gpytorch.likelihoods.GaussianLikelihood()

    # RBF kernel with Automatic Relevance Determination (separate lengthscales per dimension)
    ls_max = 0.4  
    outputscale_min = 0.001 
    outputscale_max = 10.0
    base_kernel =  gpytorch.kernels.RBFKernel(ard_num_dims=2, lengthscale_constraint=gpytorch.constraints.Interval(lower_bound=0.05, upper_bound=ls_max))
    kernel = gpytorch.kernels.ScaleKernel(base_kernel,
        outputscale_constraint=gpytorch.constraints.Interval(
        lower_bound=outputscale_min, upper_bound=outputscale_max))
    model_fx = ForceGP(X_fx, y_fx, lik_fx, kernel)

    print('defined model and likelihood')
    # Put in train mode
    model_fx.train(); lik_fx.train()

    # Joint optimizer over both models + likelihoods
    optimizer = torch.optim.Adam(list(model_fx.parameters()) + list(lik_fx.parameters()), lr=0.05)
    
    mll_fx = gpytorch.mlls.ExactMarginalLogLikelihood(lik_fx, model_fx)

    num_iters = 300
    for i in range(num_iters):
        optimizer.zero_grad()
        out_fx = model_fx(X_fx)
        loss = -mll_fx(out_fx, y_fx)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        print("fx lengthscale:", model_fx.covar_module.base_kernel.lengthscale)
        print("fx noise:", lik_fx.noise.item())
        print(f"fx outputscale: {model_fx.covar_module.outputscale.item():.6f}")
        print(f"fx mean constant: {model_fx.mean_module.constant.item():.6f}")

    # Switch to eval mode
    model_fx.eval(); lik_fx.eval()

    grid_shape = cv.shape
    X_test_np = cv
    X_test = torch.tensor(X_test_np, dtype=torch.float32)

    num_samples = 25
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        pred_fx = lik_fx(model_fx(X_test))  # N_test

        # latent function posterior (no observation noise)
        posterior_fx = model_fx(X_test)

        # mean and std on grid
        fx_mean = posterior_fx.mean          # [nx]
        fx_std  = posterior_fx.stddev        # [nx]

        # SAMPLE THE GP num_samples TIMES: shape [num_samples, nx]
        fx_samples = posterior_fx.rsample(torch.Size([num_samples]))

    print('model finished training')
    fx_mean_grid = pred_fx.mean.cpu().numpy().reshape(grid_shape)
    fx_std_grid  = pred_fx.stddev.cpu().numpy().reshape(grid_shape)
    fx_samples_grid = fx_samples.cpu().numpy()

    # --- DEBUG: GP prediction vs training data comparison ---
    print(f'[gpr_1d] GP prediction on full grid:')
    print(f'  fx_mean range: [{fx_mean_grid.min():.6f}, {fx_mean_grid.max():.6f}]')
    print(f'  fx_std range:  [{fx_std_grid.min():.6f}, {fx_std_grid.max():.6f}]')
    # Compare GP prediction at training points vs training targets
    if len(xcv_data) > 0:
        with torch.no_grad():
            pred_at_train = lik_fx(model_fx(X_fx))
            pred_train_mean = pred_at_train.mean.cpu().numpy()
        residuals = pred_train_mean - y_fx.numpy()
        print(f'[gpr_1d] GP at training points: pred range=[{pred_train_mean.min():.6f}, {pred_train_mean.max():.6f}]')
        print(f'[gpr_1d] training target range: [{y_fx.numpy().min():.6f}, {y_fx.numpy().max():.6f}]')
        print(f'[gpr_1d] residual (pred-target): mean={residuals.mean():.6f}, '
              f'abs_max={np.abs(residuals).max():.6f}, rmse={np.sqrt((residuals**2).mean()):.6f}')
        # Show first 10 training point comparisons
        for j in range(min(10, len(xcv_data))):
            print(f'  train[{j}] cv={xcv_data[j]:.4f}  target={y_fx.numpy()[j]:.6f}  '
                  f'pred={pred_train_mean[j]:.6f}  resid={residuals[j]:.6f}')
    else:
        print(f'[gpr_1d] No training points — skipping prediction comparison')

    iter_num = logger.interval
    with torch.no_grad():
        # Each is [1, 2] because ARD over 2 inputs
        ell_fx = model_fx.covar_module.base_kernel.lengthscale.cpu().numpy().ravel()
        chi2 = 1.0

    noise_for_recon = np.mean(fx_std_grid**2)
    
    # Create data directory if it doesn't exist
    os.makedirs('./data', exist_ok=True)
    
    with open('./data/hp.txt', 'a') as f:
        f.write(f'{ell_fx} {np.mean(fx_std_grid)}'+ '\n')

    print('reconstructing energy surface...')
    A_samples, A_mean, A_std, trapz = reconstruct_A_from_gp_samples_1D_old(cv, fx_samples_grid, ell_fx, chi2, noise_for_recon, None)
    print('energy calculated')
    data_a = np.column_stack(A_samples)
    np.savetxt(f'./data/data_asummary{iteration}.txt', data_a, fmt="%.6f")
    # For 2D, flatten each grid into 1D arrays and save columns
    x_flat = cv[0].ravel()
    y_flat = cv[1].ravel()
    fx_mean_flat = fx_mean_grid.ravel()
    fx_std_flat = fx_std_grid.ravel()
    A_mean_flat = A_mean.ravel()
    A_std_flat = A_std.ravel()
    counts_flat = counts.ravel()
    data_fes = np.column_stack((x_flat, y_flat, fx_mean_flat, fx_std_flat, A_mean_flat, A_std_flat, counts_flat))
    np.savetxt(f'./data/data_summary{iteration}.txt', data_fes, fmt="%.6f", header=f"x y f_mean f_std A_mean A_std hist", comments='')
        
    plot_data(X_fx, y_fx, cv, fx_mean_grid, loggers, iteration, counts)
    plot_all_mf_data(X_fx, y_fx, cv, fx_mean_grid, loggers, iteration)
    plot_energy(cv, A_mean, A_samples, A_std, iteration)

    logger.model = model_fx
    logger.likelihood = lik_fx
        
        # torch.save({'model_state_dict': model.state_dict(), 'likelihood_state_dict': likelihood.state_dict()}, 'model_state.pth')
        # print(model.state_dict())
        # print(likelihood.state_dict())
        # torch.save(likelihood.state_dict(), 'model_state.pth')

    # Log key parameters for debugging
    print("Saved Lengthscale:", model_fx.covar_module.base_kernel.lengthscale)
    print("Saved Noise:", lik_fx.noise)

        
    # here alter the bias magnitude based on the visit density
    alpha = limit/2
    temp = []
    f = open("model_data.txt", "w")
    print(f'[gpr_1d] Writing model_data.txt ({len(cv)} bins):')
    for i in range(len(cv)):
        
        bias_og = round(fx_mean_grid[i], 8)
        if logger.model_old is None:
            bias_old = 0
        else: 
            bias_old = round(logger.model_old[i], 8)
        bias_current = round(fx_mean_grid[i], 8)
        temp.append(bias_og)

        cv1 = X_test[i].detach().item()

        # index = np.where(logger.cv > np.array(cv1))[0][0]

        if logger.model_current == None:
            
            f.write(str(cv1) + " 0 " + str(bias_og) + '\n')  # old bias is zero in this case
            
        else:
            f.write(str(cv1) + " " + str(bias_old) + " " + str(bias_current) +'\n')
        
        # DEBUG: print model_data entries for bins with data
        if counts[i] > 0 or i % 10 == 0:
            print(f'  model_data[{i}] cv={cv1:.4f}  bias_old={bias_old:.6f}  '
                  f'bias_current={bias_current:.6f}  counts={counts[i]:.0f}  '
                  f'raw_mf={abf_mf[i]:.6f}')
    f.close()

    # --- DEBUG: model_data.txt summary ---
    temp_arr = np.array(temp)
    print(f'[gpr_1d] model_data.txt written: bias range=[{temp_arr.min():.6f}, {temp_arr.max():.6f}]')
    # Compare with raw ABF mean forces at sampled bins
    if np.any(mask_nz):
        gp_at_sampled = np.array([fx_mean_grid[i] for i in nz_idx])
        raw_at_sampled = abf_mf[mask_nz]
        diff = gp_at_sampled - raw_at_sampled
        print(f'[gpr_1d] GP bias vs raw ABF mf (sampled bins): '
              f'mean_diff={diff.mean():.6f}, max_abs_diff={np.abs(diff).max():.6f}')

    logger.model_current = temp


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


def gpr_pytorch_2d(iteration, big_log):

    limit=100
    loggers = get_loggers()
    logger = loggers[0]  # Use the first logger as a template
    dim2 = logger.dim2

    print(f'[gpr_pytorch_2D] this is iteration {iteration}')

    nx = len(big_log.cv_x)
    ny = len(big_log.cv_y)
    
    # etract data from big_log
    counts = big_log.histogram
    xabf_mf = np.zeros_like(big_log.xFsum, dtype=float)
    yabf_mf = np.zeros_like(big_log.yFsum, dtype=float)
    mask_nz = counts > 0
    xabf_mf[mask_nz] = big_log.xFsum[mask_nz] / counts[mask_nz]
    yabf_mf[mask_nz] = big_log.yFsum[mask_nz] / counts[mask_nz]
    cv = big_log.cv
    # --- DEBUG: shapes and basic diagnostics ---
    try:
        print('DEBUG gpr_pytorch_2d: big_log attributes:')
        print(f"  big_log.cv type: {type(big_log.cv)}")
        if hasattr(big_log, 'cv_x') and hasattr(big_log, 'cv_y'):
            print(f"  big_log.cv_x len: {len(big_log.cv_x)}  big_log.cv_y len: {len(big_log.cv_y)}")
        print(f"  cv[0].shape: {getattr(cv[0], 'shape', None)}  cv[1].shape: {getattr(cv[1], 'shape', None)}")
        print(f"  counts.shape: {getattr(counts, 'shape', None)}  counts.ndim: {getattr(counts, 'ndim', None)}  counts.size: {getattr(counts, 'size', None)}")
        if hasattr(logger, 'histogram'):
            print(f"  logger.histogram.shape: {getattr(logger.histogram, 'shape', None)}")
        print(f"  big_log.xFsum.shape: {getattr(big_log.xFsum, 'shape', None)}  big_log.yFsum.shape: {getattr(big_log.yFsum, 'shape', None)}")
        print(f"  mask_nz sum: {np.count_nonzero(mask_nz)}")
    except Exception as _e:
        print('DEBUG: failed to print debug info:', _e)
    # determine dampening factor and dampen data
    dampening = np.minimum(1, counts / limit)
    xmeanforces_damped = dampening * xabf_mf
    ymeanforces_damped = dampening * yabf_mf
    # remove bins without counts
    rows, cols = np.nonzero(counts)
    xnonzero_mf = xmeanforces_damped[mask_nz]
    ynonzero_mf = ymeanforces_damped[mask_nz]
    # Use the same boolean mask to select CV grid points that have counts.
    # `counts` has shape (size_x, size_y) while `cv[0]`/`cv[1]` are (size_y, size_x),
    # so transpose them to align axes before boolean indexing.
    xcv_data = cv[0].T[mask_nz]
    ycv_data = cv[1].T[mask_nz]
    
    print(f'there are {len(xcv_data)} training points')

    num_nonzero = rows.size
    total_bins = logger.histogram.size          # or np.prod(log.hist.shape)
    frac_nonzero = num_nonzero / total_bins

    # x, y_x, y_y = build_gp_training_data_2d(loggers, dim2, min_count=1)
    
    # if x is None:
    #     print("Not enough data to update GP yet.")
    #     return
    
    X_fx, y_fx = make_training_data(xcv_data, ycv_data, xnonzero_mf)
    X_fy, y_fy = make_training_data(xcv_data, ycv_data, ynonzero_mf)

    xls_max = 0.4
    yls_max = 18
    os_min = 0.001
    os_max = 10.0

    base_kernel = RBFDerivativeKernel2D(ard_num_dims=2, 
                lengthscale_constraint=gpytorch.constraints.Interval(
                    lower_bound=torch.tensor([0.05, 0.05]),
                    upper_bound=torch.tensor([xls_max, yls_max])
                ))

    kernel = gpytorch.kernels.ScaleKernel(base_kernel, 
                outputscale_constraint=gpytorch.constraints.Interval(lower_bound=os_min, upper_bound=os_max))

    train_x, train_y = stack_force_training_data(X_fx, y_fx, X_fy, y_fy, y_is_force=True)

    if int(iteration) > 1:
        print('using previous model - warm starting from checkpoint')

        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = PotentialGradGP(train_x, train_y, likelihood, kernel)

        checkpoint = torch.load(f"./data/gp_checkpoint.pth", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        likelihood.load_state_dict(checkpoint["likelihood"])

        # update training data (keep hyperparameters)
        model.set_train_data(inputs=train_x, targets=train_y, strict=False)
        num_iters = 300
    else:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = PotentialGradGP(train_x, train_y, likelihood, kernel)
        num_iters = 500

     # ============ STEP 3: Model Training ============
    model.train(); likelihood.train()

    # Use different learning rates for different parameters:
    optimizer = torch.optim.Adam([
        {'params': model.covar_module.parameters(), 'lr': 0.01},
        {'params': model.mean_module.parameters(), 'lr': 0.01},
        {'params': likelihood.parameters(), 'lr': 0.001}  # Lower for noise
    ], lr=0.01)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    # Add scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_iters)

    for i in range(num_iters):
        optimizer.zero_grad()
        out = model(train_x)
        loss = -mll(out, train_y)
        loss.backward()
        
        # Gradient clipping helps stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
        os_ = model.covar_module.outputscale.item()
        nz = likelihood.noise.item()
        print(f"  Iter {i}/{num_iters}: loss = {loss.item():.4f}, ls={ls}, os={os_:.4f}, noise={nz:.4e}")

    with torch.no_grad():
        ls = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()
        os_ = model.covar_module.outputscale.item()
        nz = model.likelihood.noise.item()
        print("\nTraining Results (single model):")
        print(f"  lengthscales (ell): {ls}")
        print(f"  outputscale (chi2): {os_:.6f}")
        print(f"  noise: {nz:.4e}")

     # ============ STEP 4: Posterior Predictions ============
    model.eval(); likelihood.eval()

    grid_shape = cv[0].shape
    X_test_np = np.stack([cv[0].ravel(), cv[1].ravel()], axis=-1).astype(np.float32)
    X_test_xy = torch.tensor(X_test_np, dtype=torch.float32)

    # Augment with comp=0/1 to query dA/dx and dA/dy
    X_test_dx = make_aug_test(X_test_xy, 0.0)
    X_test_dy = make_aug_test(X_test_xy, 1.0)

    num_samples = 10
    print(f'generating {num_samples} posterior predictions and samples')

    with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.fast_pred_samples():
        post_dx = model(X_test_dx)   # predicts dA/dx
        post_dy = model(X_test_dy)   # predicts dA/dy

        dAdx_mean_flat = post_dx.mean
        dAdx_std_flat  = post_dx.stddev
        dAdy_mean_flat = post_dy.mean
        dAdy_std_flat  = post_dy.stddev

        dAdx_samples_flat = post_dx.rsample(torch.Size([num_samples]))  # (S, N_test)
        dAdy_samples_flat = post_dy.rsample(torch.Size([num_samples]))  # (S, N_test)

    # Convert to force predictions for your existing push logic/plots: F = -∇A
    fx_mean_flat = -dAdx_mean_flat
    fy_mean_flat = -dAdy_mean_flat
    fx_std_flat  = dAdx_std_flat     # std unaffected by sign
    fy_std_flat  = dAdy_std_flat

    fx_samples_flat = -dAdx_samples_flat
    fy_samples_flat = -dAdy_samples_flat

    # reshape to grids
    fx_mean_grid = fx_mean_flat.cpu().numpy().reshape(grid_shape)
    fx_std_grid  = fx_std_flat.cpu().numpy().reshape(grid_shape)
    fy_mean_grid = fy_mean_flat.cpu().numpy().reshape(grid_shape)
    fy_std_grid  = fy_std_flat.cpu().numpy().reshape(grid_shape)

    # fx_samples_grid = fx_samples_flat.cpu().numpy().reshape(num_samples, *grid_shape)
    # fy_samples_grid = fy_samples_flat.cpu().numpy().reshape(num_samples, *grid_shape)

    # keep ALSO the gradient samples for your energy reconstructor (it wants gradients if y_is_force=False)
    dAdx_samples_grid = dAdx_samples_flat.cpu().numpy().reshape(num_samples, *grid_shape)
    dAdy_samples_grid = dAdy_samples_flat.cpu().numpy().reshape(num_samples, *grid_shape)

    # print(f"X_fx range: {X_fx.min().item():.2f} to {X_fx.max().item():.2f}")
    # print(f"X_fy range: {X_fy.min().item():.2f} to {X_fy.max().item():.2f}")
    # print(f"y_fx range: {y_fx.min().item():.4f} to {y_fx.max().item():.4f}")
    # print(f"y_fy range: {y_fy.min().item():.4f} to {y_fy.max().item():.4f}")
    # print(f"Total training points: {len(train_x)}")

    # print(f"\nPrediction Results:")
    # print(f"  Fx prediction range: [{fx_mean_flat.min().item():.4f}, {fx_mean_flat.max().item():.4f}]")
    # print(f"  Fy prediction range: [{fy_mean_flat.min().item():.4f}, {fy_mean_flat.max().item():.4f}]")
    # print(f"  Fx std range: [{fx_std_flat.min().item():.4f}, {fx_std_flat.max().item():.4f}]")
    # print(f"  Fy std range: [{fy_std_flat.min().item():.4f}, {fy_std_flat.max().item():.4f}]")

    del X_test_xy, X_test_dx, X_test_dy
    del dAdx_mean_flat, dAdx_std_flat, dAdy_mean_flat, dAdy_std_flat
    del dAdx_samples_flat, dAdy_samples_flat
    del fx_mean_flat, fx_std_flat, fy_mean_flat, fy_std_flat
    del fx_samples_flat, fy_samples_flat

    # ============ STEP 5: Uncertainty-Based Exploration Bias ============
    # Compute overall uncertainty as magnitude of force uncertainty vector
    uncertainty_grid = np.sqrt(fx_std_grid**2 + fy_std_grid**2)
    nx, ny = uncertainty_grid.shape

    # Exclude grid edges to avoid boundary artifacts
    margin_x = 4  # grid points to discard on each side
    margin_y = 4

    # Create mask: True in interior, False on borders
    mask = np.zeros_like(uncertainty_grid, dtype=bool)
    mask[margin_x:nx-margin_x, margin_y:ny-margin_y] = True

    # Apply mask: set edge uncertainties to -inf so they're never selected as maximum
    masked_uncertainty = np.where(mask, uncertainty_grid, -np.inf)

    # Find the grid point with maximum uncertainty
    flat_idx = np.argmax(masked_uncertainty)
    i_max, j_max = np.unravel_index(flat_idx, masked_uncertainty.shape)

    dist_max = cv[0][i_max, j_max]  # target distance coordinate (Å)
    dih_max  = cv[1][i_max, j_max]  # target dihedral coordinate (rad, range [0, pi])

    # Boltzmann constant and temperature
    kb = 8.62e-5  # kcal/mol/K
    T  = 300       # K

    # Separate spring constants per CV to account for different units (Å vs radians)
    # Tune k_push_dist (kcal/mol/Å²) and k_push_dih (kcal/mol/rad²) independently.
    k_push_dist = kb * T * 2 # spring constant for distance CV
    k_push_dih  = kb * T  * 2 # use same spring constant for dihedral natively in radians

    # ── Distance CV: plain difference (linear, non-periodic) ──────────────────────
    d_dist = cv[0] - dist_max  # (Å)

    # ── Dihedral CV [0, π]: plain difference (half-range, non-periodic) ─────────
    # No wrapping needed — 0 and π are distinct endpoints, not periodic images
    d_dih = cv[1] - dih_max    # (radians)

    # Raw push forces (negative gradient of harmonic well)
    fx_push_raw = -k_push_dist * d_dist
    fy_push_raw = -k_push_dih  * d_dih

    # ── Time-dependent ramp: gradually turn on exploration over iterations ─────────
    iter_idx   = int(iteration)
    ramp_start = 2    # iterations with zero push
    ramp_iters = 50   # iterations to ramp from 0 → 1

    if iter_idx <= ramp_start:
        time_ramp = 0.0
    else:
        time_ramp = (iter_idx - ramp_start) / ramp_iters
        time_ramp = min(time_ramp, 1.0)  # cap at 1.0

    # ── Coverage-dependent scaling: reduce push as sampling coverage increases ─────
    frac_start = 0.85  # start reducing at 85% coverage
    frac_end   = 0.95  # zero push at 95% coverage

    if frac_nonzero <= frac_start:
        coverage_scale = 1.0  # full strength
    elif frac_nonzero >= frac_end:
        coverage_scale = 0.0  # zero strength
    else:
        coverage_scale = (frac_end - frac_nonzero) / (frac_end - frac_start)  # linear interpolation

    # Overall exploration scaling: product of time ramp and coverage scaling
    exploration_scale = time_ramp * coverage_scale

    # Apply scaling to push forces
    fx_push = fx_push_raw * exploration_scale
    fy_push = fy_push_raw * exploration_scale

    # ── Exploration bias energy surface (harmonic potential, mixed units) ──────────
    # Note: each term uses its own k to keep contributions physically balanced
    exploration_bias_energy_raw = (0.5 * k_push_dist * np.square(d_dist)
                                   + 0.5 * k_push_dih  * np.square(d_dih))
    exploration_bias_energy = exploration_scale * exploration_bias_energy_raw

    print(f"exploration_scale:          {exploration_scale}")
    print(f"Max energy (single point):  {np.max(exploration_bias_energy)}")
    print(f"Min energy (single point):  {np.min(exploration_bias_energy)}")
    print(f"Are there any NaNs?         {np.any(np.isnan(exploration_bias_energy))}")
    print(f"Max uncertainty location:   dist_max={dist_max:.4f} Å, dih_max={dih_max:.4f} deg")

# ============ STEP 7: Energy Surface Reconstruction ============
    with torch.no_grad():
        ell_shared = model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().ravel()  # (2,)
        chi2_shared = float(model.covar_module.outputscale.detach().cpu())                      # scalar

    # your old noise heuristic can remain (this is force std -> same as grad std)
    noise_for_recon = 0.5 * (fx_std_grid**2 + fy_std_grid**2)

    # IMPORTANT: reconstructor expects gradients if y_is_force=False
    samples_phys = np.stack([dAdx_samples_grid, dAdy_samples_grid], axis=1)  # (S,2,nx,ny)

    print(f'Learned hyperparameters - ell: {ell_shared} chi2: {chi2_shared}')
    print('reconstructing energy surface from GP samples')

    A_samples, A_mean, A_std, trap = reconstruct_A_from_gp_samples_2D_no_norm_fast(
        xv=cv[0], yv=cv[1],
        samples_phys=samples_phys,      # gradients (dA/dx, dA/dy)
        ell=ell_shared,
        chi2=chi2_shared,
        noise=noise_for_recon,
        noise_is_std=False,
        y_is_force=False,               # correct: we are passing gradients
        jitter_rel=1e-10,
        center="min")

    # Plot GP results (forces/uncertainty) and reconstructed energy surface
    try:
        # Build NaN-masked 2D training grids aligned with cv (transpose counts-indexed
        # arrays from (size_x, size_y) to (size_y, size_x) to match cv layout)
        train_fx_grid = np.where(mask_nz.T, xmeanforces_damped.T, np.nan)
        train_fy_grid = np.where(mask_nz.T, ymeanforces_damped.T, np.nan)
        plot_gp_contours(cv, fx_mean_grid, fy_mean_grid, fx_std_grid, fy_std_grid, iteration,
                         train_fx=train_fx_grid, train_fy=train_fy_grid)
    except Exception:
        print('Warning: plot_gp_contours failed')

    try:
        plot_energy_2d(cv, A_mean, A_std, A_samples, iteration)
    except Exception:
        print('Warning: plot_energy_2d failed')

    plot_biglog_histogram_contour(big_log, iteration, color_norm="log")

    # keep A_samples for saving below; only discard trap
    del trap
    
    # ============ STEP 8: Save Results ============
    # Save trained model checkpoints for next iteration (warm-start)
    checkpoint = {
        "model": model.state_dict(),
        "likelihood": likelihood.state_dict(),}
    # save checkpoint in a local data directory (cluster root '/' is not writable)
    os.makedirs('./data', exist_ok=True)
    torch.save(checkpoint, os.path.join('data', 'gp_checkpoint.pth'))
    print("Saved GP checkpoint for next iteration")

    # Compute total forces: GP predictions + exploration bias
    fx_total = fx_mean_grid + fx_push
    fy_total = fy_mean_grid + fy_push
    total_energy = A_mean + exploration_bias_energy

    # ============ STEP 6/8: Visualization ============
    try:
        plot_exploratory_bias_2d(cv, fx_push, fy_push, exploration_bias_energy, iteration)
    except Exception as e:
        print('Warning: plot_exploratory_bias_2d failed:', e)

    try:
        plot_total_force_2d(cv, fx_total, fy_total, total_energy, iteration)
    except Exception as e:
        print('Warning: plot_total_force_2d failed:', e)

    del fx_push_raw, fy_push_raw
    del uncertainty_grid, mask, masked_uncertainty

    # Flatten grids for saving to file
    x_flat = cv[0].ravel()
    y_flat = cv[1].ravel()
    fx_flat = fx_total.ravel()
    fy_flat = fy_total.ravel()

    # Save energy surface summary (x, y, A_mean, A_std, histogram)
    from pathlib import Path
    os.makedirs('./data', exist_ok=True)

    # Log hyperparameters to tracking file
    with open(f'./data/hp.txt', 'a') as f:
        f.write(f'{ell_shared} {chi2_shared} {np.mean(A_std)} {frac_nonzero}\n')

    data_a = np.column_stack(A_samples)
    np.savetxt(f'./data/data_asummary{iteration}.txt', data_a, fmt="%.6f")
    # Flatten 2D grids to 1D arrays for consistent column stacking
    fx_mean_flat = fx_mean_grid.ravel()
    fx_std_flat = fx_std_grid.ravel()
    A_mean_flat = A_mean.ravel()
    A_std_flat = A_std.ravel()
    counts_flat = counts.ravel()
    data_fes = np.column_stack((x_flat, y_flat, fx_mean_flat, fx_std_flat, A_mean_flat, A_std_flat, counts_flat))
    np.savetxt(f'./data/data_summary{iteration}.txt', data_fes, fmt="%.6f", comments='')
        
    print(f'[gpr_2d] Writing model_data.txt ({len(cv)} bins)')
    # Save GP force predictions (for external analysis or plotting)
    data_gpr = np.column_stack((x_flat, y_flat, fx_flat, fy_flat))
    np.savetxt(f'model_data.txt', data_gpr, fmt="%.6f", comments='')
    
          

