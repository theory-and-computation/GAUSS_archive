"""logger.py — Simulation state container for the ABF + GP benchmark.

The ``Logger`` class holds all mutable state shared between the Langevin
integrators (main.py) and the GP training / reconstruction code
(gpr_model.py):

  * Circular trajectory buffers — last 3 steps only (enough for the
    3-point backward difference used in the ABF force estimate)
  * ABF bin accumulators — force sums, visit counts, bin-mean forces
  * Per-bin force *block* statistics — consumed by the
    autocorrelation-corrected noise estimate in
    gpr_model._estimate_global_block_noise ([NOISE 2/3] in gpr_model.py)
  * The trained GP model + likelihood (refreshed every interval)
  * Row buffers for the periodic file flushes in main.py

The normalizer classes at the bottom (MinMaxNormalizer, StandardScaler)
are small self-contained scaling utilities; they are NOT referenced by
main.py or gpr_model.py (kept for analysis scripts).

The GP *kernel* and *noise model* definitions are NOT in this file —
see the [KERNEL x/3] / [NOISE x/3] markers in gpr_model.py.
"""
import numpy as np


class Logger:
    """Mutable simulation state shared by the integrator and the GP code.

    Parameters
    ----------
    total_steps : int
        Total integration steps (from SimConfig.total_steps).
    x_bins, y_bins : int
        Number of CV bins along each coordinate.
    x_cv, y_cv : ndarray
        CV grids (a meshgrid; in 1D mode only row 0 of x_cv is used).
    dim2D : bool
        True → 2D CV (x, y); False → 1D CV (x).
    start_bias, interval : int
        ABF/GP schedule (also mirrored from SimConfig in main.init_log).
    """

    def __init__(self, total_steps, x_bins, y_bins, x_cv, y_cv,
                 dim2D, start_bias, interval):
        self.dim2 = dim2D
        self.init_variables(total_steps, x_bins, y_bins, x_cv, y_cv,
                            dim2D, start_bias, interval)

    def init_variables(self, total_steps, x_bins, y_bins, x_cv, y_cv,
                       dim2D, start_bias, interval):
        # ════════════════════════════════════════════════════════════════
        # Circular trajectory buffers (shared by 1D and 2D)
        # ════════════════════════════════════════════════════════════════
        # Only the last 3 steps are kept: the integrator needs the previous
        # step's (x, v, F_total), and the ABF force estimate uses a 3-point
        # backward difference (steps n, n-1, n-2). The slot for step n is
        # get_buffer_index(n) = n % buffer_size, so each write overwrites
        # the state from 3 steps earlier.
        self.buffer_size = 3
        self.atom1_xpositions = np.zeros(self.buffer_size)      # x position
        self.atom1_xvelocities = np.zeros(self.buffer_size)     # x velocity
        self.energies = np.zeros(self.buffer_size)              # potential energy
        self.xforces = np.zeros(self.buffer_size)               # model-potential force, x
        self.xwp_array = np.zeros(self.buffer_size)             # CV momentum W_p = m·v, x
        self.total_xforce_array = np.zeros(self.buffer_size)    # total (bias-included) force, x

        # ════════════════════════════════════════════════════════════════
        # GP model holders (populated by gpr_model.gp_model1D_simple)
        # ════════════════════════════════════════════════════════════════
        # self.model : trained force GP  (DerivativeGPModel1D)
        # self.lh    : its Gaussian likelihood
        # Storage only — the kernel and noise model themselves are defined
        # in gpr_model.py ([KERNEL x/3] / [NOISE x/3] markers).
        self.lh = None
        self.model = None

        # ════════════════════════════════════════════════════════════════
        # Scalar settings (mirrors of SimConfig, set in main.init_log)
        # ════════════════════════════════════════════════════════════════
        self.stddev = np.zeros(x_bins)      # (unused by main/gpr code — kept for analysis)
        self.max_steps = 0
        self.kbT = 0
        self.start_bias = 0
        self.interval = 0
        self.step = 0                       # global step counter (bumped in run_sim)
        self.xnormalizer = None             # (unused by main/gpr code — kept for analysis)
        self.ynormalizer = None
        self.max_unce = None
        self.current_step = 0               # (legacy — the live counter is self.step)

        # Row buffers flushed to ./data/ every 1000 steps (see main.py)
        self.vel_buffer = []
        self.force_buffer = []

        match self.dim2:
            case True:  # ── 2D CV (x, y) ────────────────────────────────
                # y-direction circular buffers (x-direction ones above)
                self.atom1_ypositions = np.zeros(self.buffer_size)
                self.atom1_yvelocities = np.zeros(self.buffer_size)
                self.yforces = np.zeros(self.buffer_size)
                self.ywp_array = np.zeros(self.buffer_size)
                self.total_yforce_array = np.zeros(self.buffer_size)

                # ── ABF bin accumulators (x_bins × y_bins) ──────────────
                # xfsum / yfsum   : running sum of ABF force samples per bin
                #                   (raw before start_bias; bias-subtracted
                #                   afterwards — see Langevin_int2D in main.py)
                # count_array     : total visits per bin since step 0 — used
                #                   for the training-data damping and to
                #                   select visited bins for GP training
                # new_counts      : visits per bin since the LAST GP retrain
                #                   (zeroed at each retrain; plotting only)
                # xmf / ymf_array : bin-mean ABF force = fsum / count_array
                self.xfsum = np.zeros((x_bins, y_bins))
                self.yfsum = np.zeros((x_bins, y_bins))
                self.hist = np.zeros((x_bins, y_bins))      # (unused by main/gpr code)
                self.x_cv_grid = x_cv                       # 2D meshgrid of CV x values
                self.y_cv_grid = y_cv                       # 2D meshgrid of CV y values
                self.count_array = np.zeros((x_bins, y_bins))
                self.new_counts = np.zeros((x_bins, y_bins))
                self.xmf_array = np.zeros((x_bins, y_bins))
                self.ymf_array = np.zeros((x_bins, y_bins))

                # Domain bounds of the CV grids (grid centres, periodic)
                self.x1 = self.x_cv_grid.min()
                self.x2 = self.x_cv_grid.max()
                self.y1 = self.y_cv_grid.min()
                self.y2 = self.y_cv_grid.max()

            case False:  # ── 1D CV (x) ─────────────────────────────────
                # ── ABF bin accumulators (x_bins,) ──────────────────────
                # xfsum_abf       : running sum of ABF force samples per bin
                # xmf_array_abf   : bin-mean ABF force — this is what the
                #                   force GP is trained on (damped by
                #                   min(1, count/limit) in gpr_model.py)
                # xmf_array_us    : bin-mean *unbiased* reference force
                #                   (−fx from the potential), recorded each
                #                   step for comparison
                # xfsum_us        : (unused by main/gpr code — kept for analysis)
                self.xfsum_abf = np.zeros(x_bins)
                self.xfsum_us = np.zeros(x_bins)
                self.hist = np.zeros(x_bins)                # (unused by main/gpr code)
                self.x_cv_grid = x_cv[0, :]                 # 1D array of bin centre values
                self.count_array = np.zeros(x_bins)
                self.new_counts = np.zeros(x_bins)
                self.xmf_array_abf = np.zeros(x_bins)
                self.xmf_array_us = np.zeros(x_bins)

                # ── Per-bin force blocks (autocorrelation analysis) ─────
                # When SimConfig.autocorrelation_correction is on, every ABF
                # sample is also routed through record_force_sample(); each
                # bin stores the mean of every completed block of
                # `force_block_size` *ordered* samples. Those block means are
                # consumed by gpr_model._estimate_global_block_noise
                # ([NOISE 2/3]) to build the autocorrelation-corrected
                # observation noise of the likelihood.
                self.force_block_sums = np.zeros(x_bins)
                self.force_block_counts = np.zeros(x_bins)
                self.force_block_means = [[] for _ in range(x_bins)]
                self.force_block_size = 50  # overridden by SimConfig in main.init_log

            case _:  # unreachable for a bool dim2D — kept as a safety net
                print('You did not choose a dimension')

    # ── Circular-buffer access ────────────────────────────────────────────

    def get_buffer_index(self, step):
        """Map a global step number to its circular buffer slot (n % 3)."""
        return step % self.buffer_size

    # ── Force-block statistics (feeds the [NOISE 2/3] estimator) ─────────

    def record_force_sample(self, bin_index, force_sample):
        """Accumulate one ordered ABF force sample for the block-means
        diagnostic (consumed by gpr_model._estimate_global_block_noise).

        Once a bin has collected `force_block_size` samples, the block mean
        is appended to force_block_means[bin_index] and the running
        sum/count are reset.
        """
        self.force_block_sums[bin_index] += force_sample
        self.force_block_counts[bin_index] += 1
        if self.force_block_counts[bin_index] >= self.force_block_size:
            self.force_block_means[bin_index].append(
                float(
                    self.force_block_sums[bin_index]
                    / self.force_block_counts[bin_index]
                )
            )
            self.force_block_sums[bin_index] = 0.0
            self.force_block_counts[bin_index] = 0.0

    def finalize_force_blocks(self):
        """Store the *incomplete* block (if any) of every visited bin.

        Call this after the simulation ends so samples recorded since the
        last block completion are not lost before computing the
        autocorrelation-corrected noise estimate.
        """
        visited = self.force_block_counts > 0
        for bin_index in np.flatnonzero(visited):
            mean = (
                self.force_block_sums[bin_index]
                / self.force_block_counts[bin_index]
            )
            self.force_block_means[bin_index].append(float(mean))
        self.force_block_sums.fill(0.0)
        self.force_block_counts.fill(0.0)


# ════════════════════════════════════════════════════════════════════════════
# Data scaling utilities
#   Not referenced by main.py or gpr_model.py — kept for analysis scripts.
#   Both share the same affine form  x_norm = x·scale + min_  so that
#   normalize/recover are inverses of each other.
# ════════════════════════════════════════════════════════════════════════════

class MinMaxNormalizer:
    """Scale each feature to a fixed range via its min and max."""

    def __init__(self, data, feature_range=(0.0, 1.0)):
        """
        data : array-like, shape [N, D]
            The dataset used to compute min and max for each feature.
        feature_range : tuple (min, max), default (0, 1)
            Desired range of transformed data.
        """
        self.data_min = np.min(data, axis=0)
        self.data_max = np.max(data, axis=0)
        self.range_min, self.range_max = feature_range

        # avoid divide by zero if some feature is constant
        self.scale = (self.range_max - self.range_min) / np.where(
            (self.data_max - self.data_min) != 0,
            (self.data_max - self.data_min),
            1.0)
        self.min_ = self.range_min - self.data_min * self.scale

    def normalize(self, x):
        """Transform to [range_min, range_max]."""
        return x * self.scale + self.min_

    def recover(self, x_norm):
        """Undo normalization (recover original scale)."""
        if x_norm.ndim == 1:
            x_norm = x_norm.reshape(-1, 1)
        return ((x_norm - self.min_) / self.scale)


class StandardScaler:
    """Standardize each feature to zero mean / unit variance."""

    def __init__(self, data, feature_range=None, eps=1e-12):
        """
        data : array-like, shape [N, D] or [N]
            Dataset used to compute per-feature mean and std.
        feature_range : unused (kept for API compatibility with MinMaxNormalizer)
        eps : small floor to avoid divide-by-zero for constant features
        """
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        # Per-feature statistics
        self.mean = np.mean(data, axis=0)           # shape [D]
        self.std  = np.std(data, axis=0)            # shape [D]
        self.std  = np.maximum(self.std, eps)       # avoid zeros

        # Reuse the same affine form as MinMaxNormalizer:
        # standardization (x - mean) / std  ==  x*(1/std) + (-mean/std)
        self.scale = 1.0 / self.std                 # shape [D]
        self.min_  = - self.mean * self.scale       # shape [D]

    def normalize(self, x):
        """Standardize each feature: z = (x - mean) / std. Returns same shape as input."""
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return (x * self.scale + self.min_).astype(np.float32)

    def recover(self, x_norm):
        """Inverse transform: x = std * x_norm + mean."""
        x_norm = np.asarray(x_norm, dtype=np.float32)
        if x_norm.ndim == 1:
            x_norm = x_norm.reshape(-1, 1)
            return ((x_norm - self.min_) / self.scale).flatten().astype(np.float32)
        return ((x_norm - self.min_) / self.scale).astype(np.float32)