"""
State Logging, Checkpointing, and Collective Variable (CV) Mechanics for VASP.

This module provides:
1. `Atom` and `CollectiveVariable` descriptors to define and index CV atom groups.
2. `Logger`: Primary state accumulator tracking 1D/2D bin counts, running force
   sums, running mean forces, and generalized conjugate momenta history (Wp).
3. PyTorch autograd CV definitions for dihedrals and radius of gyration (Rg).
"""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import List, Sequence, Tuple

import numpy as np
import torch

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jnp = None
    HAS_JAX = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class Atom:
    """Atomic descriptor specifying structural index, mass, and element label.

    Parameters
    ----------
    index : int
        0-based index of the atom in the structure/POSCAR.
    mass : float
        Atomic mass in atomic mass units (amu).
    atom_type : str
        Elemental symbol (e.g., 'C', 'H', 'O').
    """

    def __init__(self, index: int, mass: float, atom_type: str):
        self.index: int = index
        self.mass: float = mass
        self.type: str = atom_type

    def __repr__(self) -> str:
        return f"Atom(index={self.index}, mass={self.mass:.2f}, type='{self.type}')"


class CollectiveVariable:
    """Discretized collective variable definition.

    Parameters
    ----------
    cv_type : str
        Type identifier (e.g., 'distance', 'dihedral', 'rg').
    atoms : List[Atom]
        List of participating Atom objects.
    cv_min : float
        Lower boundary of the discretized CV domain.
    cv_max : float
        Upper boundary of the discretized CV domain.
    num_bins : int
        Number of discretization bins.
    """

    def __init__(
        self,
        cv_type: str,
        atoms: List[Atom],
        cv_min: float,
        cv_max: float,
        num_bins: int,
    ):
        self.type: str = cv_type
        self.atoms: List[Atom] = atoms
        self.min: float = cv_min
        self.max: float = cv_max
        self.num_bins: int = num_bins
        self.bin_size: float = (cv_max - cv_min) / max(1, num_bins)
        self.values: np.ndarray = np.linspace(cv_min, cv_max, num_bins)

    def __repr__(self) -> str:
        return (
            f"CollectiveVariable(type='{self.type}', atoms={len(self.atoms)}, "
            f"range=[{self.min:.3f}, {self.max:.3f}], bins={self.num_bins})"
        )


# =============================================================================
# LOGGER & RUNNING ACCUMULATOR
# =============================================================================

class Logger:
    """Tracks trajectory state, bin histograms, and running mean force estimators.

    Parameters
    ----------
    cv : np.ndarray or list of np.ndarray
        1D grid array or 2D meshgrid [X, Y] of collective variable bin centers.
    dim2 : bool
        True if operating on a 2D CV space, False for 1D.
    """

    def __init__(self, cv: np.ndarray | Sequence[np.ndarray], dim2: bool):
        self.dim2: bool = dim2
        self.iteration: int = 0
        self.step: int = 0
        self.dt: float | None = None
        self.cell: np.ndarray = np.zeros(3, dtype=float)

        # Generalized momentum history for 3-point backward finite differences
        self.Wp_prev: float | np.ndarray | None = None
        self.Wp_prevprev: float | np.ndarray | None = None

        # Collective variable objects (populated dynamically by workflow initializer)
        self.dist_cv: CollectiveVariable | None = None
        self.dihedral_cv: CollectiveVariable | None = None
        self.rg_cv: CollectiveVariable | None = None

        if not dim2:
            self.cv: np.ndarray = np.asarray(cv, dtype=float)
            self.size: int = len(self.cv)

            # 1D Accumulators
            self.histogram: np.ndarray = np.zeros(self.size, dtype=int)
            self.hist_new: np.ndarray = np.zeros(self.size, dtype=int)
            self.xFsum: np.ndarray = np.zeros(self.size, dtype=float)
            self.xmean_forces: np.ndarray = np.zeros(self.size, dtype=float)

        else:
            self.cv = cv
            cv_arr = np.asarray(cv)
            self.cv_x: np.ndarray = np.unique(cv_arr[0])
            self.cv_y: np.ndarray = np.unique(cv_arr[1])
            self.size_x: int = len(self.cv_x)
            self.size_y: int = len(self.cv_y)

            # 2D Grid Accumulators (shape: size_x, size_y)
            self.histogram: np.ndarray = np.zeros((self.size_x, self.size_y), dtype=int)
            self.hist_new: np.ndarray = np.zeros((self.size_x, self.size_y), dtype=int)
            self.xFsum: np.ndarray = np.zeros((self.size_x, self.size_y), dtype=float)
            self.yFsum: np.ndarray = np.zeros((self.size_x, self.size_y), dtype=float)
            self.xmean_forces: np.ndarray = np.zeros((self.size_x, self.size_y), dtype=float)
            self.ymean_forces: np.ndarray = np.zeros((self.size_x, self.size_y), dtype=float)

    # =========================================================================
    # Checkpointing & Serialization
    # =========================================================================

    def save_state(self, filepath: str | Path) -> None:
        """Serialize current Logger instance to disk."""
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
    # Differentiable Geometric Operators (PyTorch / JAX)
    # =========================================================================

    @staticmethod
    def dihedral_torch(
        atom1: torch.Tensor,
        atom2: torch.Tensor,
        atom3: torch.Tensor,
        atom4: torch.Tensor,
    ) -> torch.Tensor:
        """Compute dihedral angle in signed radians (-pi, pi] with PyTorch autograd support."""
        b1 = atom2 - atom1
        b2 = atom3 - atom2
        b3 = atom4 - atom3

        b2_norm = b2 / torch.norm(b2)
        n1 = torch.cross(b1, b2)
        n2 = torch.cross(b2, b3)
        m1 = torch.cross(b2_norm, n1)

        x = torch.dot(n1, n2)
        y = torch.dot(m1, n2)

        return torch.atan2(y, x)

    @staticmethod
    def rg_torch(
        atom_tensors: Sequence[torch.Tensor], masses: torch.Tensor
    ) -> torch.Tensor:
        """Compute radius of gyration (Rg) in Angstroms with PyTorch autograd support.

        Parameters
        ----------
        atom_tensors : Sequence[torch.Tensor]
            List of (3,) coordinate tensors for each atom in the group.
        masses : torch.Tensor
            1D tensor of atomic masses for each atom.

        Returns
        -------
        torch.Tensor
            Scalar radius of gyration.
        """
        total_mass = masses.sum()

        # Center of mass: r_com = sum(m_i * r_i) / M
        r_com = sum(masses[i] * atom_tensors[i] for i in range(len(atom_tensors))) / total_mass

        # Rg^2 = (1 / M) * sum(m_i * |r_i - r_com|^2)
        rg_sq = (
            sum(
                masses[i] * torch.sum((atom_tensors[i] - r_com) ** 2)
                for i in range(len(atom_tensors))
            )
            / total_mass
        )

        return torch.sqrt(rg_sq)

    @staticmethod
    def dihedral_jax(atoms: Sequence[jnp.ndarray]) -> jnp.ndarray:
        """Compute dihedral angle in degrees using JAX."""
        if not HAS_JAX:
            raise ImportError("JAX is required for dihedral_jax, but is not installed.")

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