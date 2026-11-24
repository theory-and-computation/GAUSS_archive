"""
State logging, checkpointing, and collective variable (CV) evaluation.

This module provides:
1. `Logger`: Primary state-tracking class that accumulates bin counts,
   generalized momenta, and mean force estimators across simulation iterations.
2. Analytic geometry routines for collective variable calculations (dihedrals, distances).
3. Coordinate and feature normalization utilities (`MinMaxNormalizer`, `StandardNormalizer`).
"""

from __future__ import annotations

import os
from pathlib import Path
import pickle
from typing import Callable, Tuple

import numpy as np

# Safe import for JAX (optional dependency for automatic differentiation of CVs)
try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jnp = None
    HAS_JAX = False


class Logger:
    """
    Tracks trajectory state, bin statistics, and running mean force estimates.
    
    Parameters
    ----------
    cv1 : str
        Name/type of first collective variable (e.g., 'dihedral').
    cv2 : str or None
        Name/type of second collective variable (None for 1D simulations).
    program : str
        Simulation engine identifier ('namd', 'vasp').
    num_gpr_iters : int
        Total number of scheduled iterations.
    tag : str
        Unique run label.
    dirtag : str
        Target output directory.
    x_cv : np.ndarray
        2D grid of CV1 coordinate centers.
    y_cv : np.ndarray
        2D grid of CV2 coordinate centers.
    """

    def __init__(
        self,
        cv1: str,
        cv2: str | None,
        program: str,
        num_gpr_iters: int,
        tag: str,
        dirtag: str,
        x_cv: np.ndarray,
        y_cv: np.ndarray,
    ):
        self.dim2: bool = cv2 is not None
        self.cv1: str = cv1
        self.cv2: str = cv2

        self.tag: str = tag
        self.dirtag: str = dirtag
        self.program: str = program
        self.iter: int = 0
        self.step: int = 0
        self.num_iters: int = num_gpr_iters
        self.dt: float = 2.0  # Simulation timestep / logging stride in ps

        # Coordinate grids
        self.x_cv_grid: np.ndarray = x_cv
        self.y_cv_grid: np.ndarray = y_cv

        # Running trajectory projections (projected momenta W_p)
        self.xwp_array: list[float] = []
        self.ywp_array: list[float] = []

        # Force accumulators (sum of observed forces per bin)
        self.xfsum_mf: np.ndarray = np.zeros_like(y_cv, dtype=float)
        self.yfsum_mf: np.ndarray = np.zeros_like(y_cv, dtype=float)
        self.xfsum_us: np.ndarray = np.zeros_like(y_cv, dtype=float)
        self.yfsum_us: np.ndarray = np.zeros_like(y_cv, dtype=float)

        # Histogram visit counts
        self.hist: np.ndarray = np.zeros_like(y_cv, dtype=int)
        self.hist_new: np.ndarray = np.zeros_like(y_cv, dtype=int)

        # Running mean force estimators: <F> = sum(F) / hist
        self.xmf_array_mf: np.ndarray = np.zeros_like(y_cv, dtype=float)
        self.ymf_array_mf: np.ndarray = np.zeros_like(y_cv, dtype=float)
        self.xmf_array_us: np.ndarray = np.zeros_like(y_cv, dtype=float)
        self.ymf_array_us: np.ndarray = np.zeros_like(y_cv, dtype=float)

    # =========================================================================
    # Checkpointing & Serialization
    # =========================================================================

    def save_state(self, filepath: str | Path) -> None:
        """Serialize current logger instance state to disk via pickle."""
        dest_path = Path(filepath)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, "wb") as f:
            pickle.dump(self, f)
        print(f"Logger state saved to: {dest_path}")

    @staticmethod
    def load_state(filepath: str | Path) -> Logger:
        """Load and return serialized Logger instance from disk."""
        src_path = Path(filepath)
        if not src_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {src_path}")
        with open(src_path, "rb") as f:
            logger = pickle.load(f)
        print(f"Logger state successfully restored from: {src_path}")
        return logger

    # =========================================================================
    # Collective Variable Geometry Functions
    # =========================================================================

    @staticmethod
    def cv_function(cv_name: str) -> Callable:
        """Factory function returning the geometry evaluator for a named CV."""
        match cv_name.lower():
            case "dihedral":
                return Logger.dihedral_jax
            case "dist":
                return Logger.dist_jax
            case _:
                raise ValueError(f"Unsupported collective variable: {cv_name}")

    @staticmethod
    def dihedral_jax(atoms: list | np.ndarray):
        """Compute dihedral angle in degrees from four atomic position vectors using JAX."""
        if not HAS_JAX:
            raise ImportError("JAX is required for Logger.dihedral_jax.")
        
        atom1, atom2, atom3, atom4 = atoms
        b1 = atom2 - atom1
        b2 = atom3 - atom2
        b3 = atom4 - atom3

        b2_norm = b2 / jnp.linalg.norm(b2)
        n1 = jnp.cross(b1, b2)
        n2 = jnp.cross(b2, b3)
        m1 = jnp.cross(b2_norm, n1)

        x = jnp.dot(n1, n2)
        y = jnp.dot(m1, n2)
        return jnp.arctan2(y, x) * (180.0 / jnp.pi)

    @staticmethod
    def dist_jax(atoms: list | np.ndarray):
        """Compute Euclidean distance between two atoms using JAX."""
        if not HAS_JAX:
            raise ImportError("JAX is required for Logger.dist_jax.")
        atom1, atom2 = atoms
        return jnp.linalg.norm(atom2 - atom1)

    @staticmethod
    def dihedral_cartesian(atom1, atom2, atom3, atom4) -> float:
        """Pure NumPy analytic computation of a dihedral angle in degrees [-180, 180]."""
        x1, y1, z1 = atom1[0], atom1[1], atom1[2]
        x2, y2, z2 = atom2[0], atom2[1], atom2[2]
        x3, y3, z3 = atom3[0], atom3[1], atom3[2]
        x4, y4, z4 = atom4[0], atom4[1], atom4[2]

        dx1, dy1, dz1 = x2 - x1, y2 - y1, z2 - z1
        dx2, dy2, dz2 = x3 - x2, y3 - y2, z3 - z2
        dx3, dy3, dz3 = x4 - x3, y4 - y3, z4 - z3

        nx1 = dy1 * dz2 - dz1 * dy2
        ny1 = dz1 * dx2 - dx1 * dz2
        nz1 = dx1 * dy2 - dy1 * dx2

        nx2 = dy2 * dz3 - dz2 * dy3
        ny2 = dz2 * dx3 - dx2 * dz3
        nz2 = dx2 * dy3 - dy2 * dx3

        arg1 = (
            dx2 * (ny1 * nz2 - ny2 * nz1)
            + dy2 * (nz1 * nx2 - nz2 * nx1)
            + dz2 * (nx1 * ny2 - ny1 * nx2)
        )
        arg2 = np.sqrt(dx2**2 + dy2**2 + dz2**2) * (nx1 * nx2 + ny1 * ny2 + nz1 * nz2)

        return float(np.arctan2(arg1, arg2) * (180.0 / np.pi))

    @staticmethod
    def all_done() -> None:
        """Notify simulation completion."""
        print("Simulation workflow execution completed successfully.")


# =============================================================================
# Feature Normalization Utilities
# =============================================================================

class MinMaxNormalizer:
    """Min-Max Feature Scaler: transforms features linearly into [min, max]."""

    def __init__(self, data: np.ndarray, feature_range: Tuple[float, float] = (0.0, 1.0)):
        self.data_min = np.min(data, axis=0)
        self.data_max = np.max(data, axis=0)
        self.range_min, self.range_max = feature_range

        # Prevent division-by-zero for zero-variance constant features
        diff = np.where((self.data_max - self.data_min) != 0, (self.data_max - self.data_min), 1.0)
        self.scale = (self.range_max - self.range_min) / diff
        self.min_ = self.range_min - self.data_min * self.scale

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Map values from input space to normalized bounds."""
        return x * self.scale + self.min_

    def recover(self, x_norm: np.ndarray) -> np.ndarray:
        """Invert normalization mapping back to raw physical dimensions."""
        return (x_norm - self.min_) / self.scale


class StandardNormalizer:
    """Standard Feature Scaler: transforms features to zero mean and unit variance."""

    def __init__(self, data: np.ndarray):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # Prevent division-by-zero for zero-variance constant features
        self.std[self.std == 0.0] = 1.0

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Standardize to zero mean and unit variance."""
        return (x - self.mean) / self.std

    def recover(self, x_norm: np.ndarray) -> np.ndarray:
        """Invert standardization back to physical dimensions."""
        return x_norm * self.std + self.mean