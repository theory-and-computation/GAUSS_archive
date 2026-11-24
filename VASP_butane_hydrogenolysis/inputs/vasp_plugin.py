"""
VASP Interactive Force and Stress Python Plugin Interface.

This module is called directly by VASP during molecular dynamics ionic steps to:
1. Evaluate instantaneous collective variables (distance, signed dihedral angle).
2. Look up adaptive biasing forces (ABF / GPR) from `model_data.txt`.
3. Apply Cartesian bias forces and harmonic boundary walls directly to ionic forces.
4. Record applied forces (f_app.txt) to allow exact unbiased force recovery.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Tuple

import numpy as np


# =============================================================================
# CONFIGURABLE ATOM INDICES & INTERFACE PARAMETERS
# (These module-level variables are targeted and rewritten by update_vasp_plugin_atoms)
# =============================================================================

dist_ion1 = 1
dist_ion2 = 3

dih_ion1 = 3
dih_ion2 = 1
dih_ion3 = 2
dih_ion4 = 0

MODEL_DATA_PATH = "../../model_data.txt"


# =============================================================================
# BIAS DATA LOOKUP UTILITIES
# =============================================================================

def load_model_bias_data(model_path: str = MODEL_DATA_PATH) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load precomputed GPR/ABF grid forces from model_data.txt.

    Expected columns: x_cv, y_cv, f_cv1, f_cv2
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model data table not found at: {path}")

    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    if data.shape[1] < 4:
        raise ValueError(f"model_data.txt requires >= 4 columns; found {data.shape[1]}")

    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def get_model_bias_forces(
    dist_val: float,
    dih_val: float,
    model_path: str = MODEL_DATA_PATH,
) -> Tuple[float, float]:
    """Retrieve nearest-neighbor bias forces (f_dist, f_dih) from grid table."""
    x_arr, y_arr, dist_arr, dih_arr = load_model_bias_data(model_path)

    def _wrap_to_pi(a: float | np.ndarray) -> float | np.ndarray:
        out = np.arctan2(np.sin(a), np.cos(a))
        return np.where(out == np.pi, -np.pi, out)

    qx = float(dist_val)
    qy = float(_wrap_to_pi(float(dih_val)))

    dy = _wrap_to_pi(y_arr - qy)
    dx = x_arr - qx
    sq_dist = dx ** 2 + dy ** 2
    idx = int(np.argmin(sq_dist))

    return float(dist_arr[idx]), float(dih_arr[idx])


def _cartesian_vector(values: np.ndarray, ion: int, name: str) -> np.ndarray:
    """Extract a 3D Cartesian vector for a specified atom index."""
    vec = np.asarray(values[ion]).reshape(-1)
    if vec.size < 3:
        raise ValueError(f"{name}[{ion}] must have >= 3 components; got {vec.shape}")
    return vec[:3]


def _extract_box(constants) -> np.ndarray:
    """Extract orthorhombic lattice lengths from VASP constants object."""
    return np.array([
        constants.lattice_vectors[0][0],
        constants.lattice_vectors[1][1],
        constants.lattice_vectors[2][2],
    ], dtype=float)


# =============================================================================
# GEOMETRIC COLLECTIVE VARIABLE EVALUATION
# =============================================================================

def calc_dist(constants, ion1: int, ion2: int) -> float:
    """Compute distance between two atoms under periodic boundary conditions."""
    box = _extract_box(constants)

    p1 = _cartesian_vector(constants.positions, ion1, "positions") * box
    p2 = _cartesian_vector(constants.positions, ion2, "positions") * box

    r = p2 - p1
    r -= box * np.round(r / box)
    return float(np.linalg.norm(r))


def calc_dih(constants, ion1: int, ion2: int, ion3: int, ion4: int) -> float:
    """Compute signed dihedral angle in radians [-pi, pi] under periodic boundary conditions."""
    box = _extract_box(constants)

    p1 = _cartesian_vector(constants.positions, ion1, "positions") * box
    p2 = _cartesian_vector(constants.positions, ion2, "positions") * box
    p3 = _cartesian_vector(constants.positions, ion3, "positions") * box
    p4 = _cartesian_vector(constants.positions, ion4, "positions") * box

    b0 = p2 - p1
    b0 -= box * np.round(b0 / box)

    b1 = p3 - p2
    b1 -= box * np.round(b1 / box)

    b2 = p4 - p3
    b2 -= box * np.round(b2 / box)

    b1_norm = np.linalg.norm(b1)
    if b1_norm == 0.0:
        raise ValueError("Degenerate geometry: zero-length central bond in dihedral CV.")
    b1_unit = b1 / b1_norm

    v = np.cross(b0, b1)
    w = np.cross(b1, b2)

    nv = np.linalg.norm(v)
    nw = np.linalg.norm(w)
    if nv == 0.0 or nw == 0.0:
        return 0.0

    v_unit = v / nv
    w_unit = w / nw

    x = float(np.dot(v_unit, w_unit))
    y = float(np.dot(np.cross(v_unit, w_unit), b1_unit))
    return float(np.arctan2(y, x))


def dihedral_gradient(
    r1: np.ndarray, r2: np.ndarray, r3: np.ndarray, r4: np.ndarray, box: np.ndarray | None = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Analytical gradient of the signed dihedral angle with respect to Cartesian coordinates."""
    r12 = r2 - r1
    r23 = r3 - r2
    r34 = r4 - r3

    if box is not None:
        r12 -= box * np.round(r12 / box)
        r23 -= box * np.round(r23 / box)
        r34 -= box * np.round(r34 / box)

    m = np.cross(r12, r23)
    n = np.cross(r23, r34)

    d23 = np.linalg.norm(r23)
    m_sq = np.linalg.norm(m) ** 2
    n_sq = np.linalg.norm(n) ** 2

    if m_sq == 0.0 or n_sq == 0.0 or d23 == 0.0:
        return np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)

    grad1 = -d23 * m / m_sq
    grad4 = d23 * n / n_sq

    dot12_23 = np.dot(r12, r23) / (d23 ** 2)
    dot34_23 = np.dot(r34, r23) / (d23 ** 2)

    grad2 = (dot12_23 - 1.0) * grad1 - dot34_23 * grad4
    grad3 = (dot34_23 - 1.0) * grad4 - dot12_23 * grad1

    return grad1, grad2, grad3, grad4


# =============================================================================
# FORCE COMPUTATION & RESTRAINTS
# =============================================================================

def calc_soft_wall(
    constants, ion1: int, ion2: int, r_max: float = 4.0, k: float = 5.0, r_buffer: float = 0.2
) -> Tuple[np.ndarray, float]:
    """Apply one-sided harmonic restraint preventing atom pair distance from exceeding r_max.

    Potential: V(r) = k * (r - r_onset)^2 for r > r_onset, where r_onset = r_max - r_buffer.
    """
    wall_forces = np.zeros_like(constants.positions)
    f_wall_proj = 0.0

    box = _extract_box(constants)
    p1 = _cartesian_vector(constants.positions, ion1, "positions") * box
    p2 = _cartesian_vector(constants.positions, ion2, "positions") * box

    diff = p2 - p1
    diff -= box * np.round(diff / box)
    r = float(np.linalg.norm(diff))
    unit_vec = diff / r

    r_onset = r_max - r_buffer
    if r > r_onset:
        f_mag = -2.0 * k * (r - r_onset)  # Restoring force magnitude (negative)
        wall_forces[ion2] += f_mag * unit_vec
        wall_forces[ion1] -= f_mag * unit_vec

        # Projection convention: f_app = -f_mag so that dWp/dt + f_app = f_dft
        f_wall_proj = -f_mag

    return wall_forces, f_wall_proj


def calc_dist_forces(
    constants, ion1: int, ion2: int, bias_cv: float
) -> Tuple[np.ndarray, float]:
    """Distribute scalar distance bias force onto participating atoms along their unit vector."""
    box = _extract_box(constants)
    p1 = _cartesian_vector(constants.positions, ion1, "positions") * box
    p2 = _cartesian_vector(constants.positions, ion2, "positions") * box

    f1_og = _cartesian_vector(constants.forces, ion1, "forces")
    f2_og = _cartesian_vector(constants.forces, ion2, "forces")

    r = p2 - p1
    r -= box * np.round(r / box)
    dist = float(np.linalg.norm(r))
    unit_vec = r / dist

    df = (f2_og - f1_og) / 2.0
    f_dft = float(np.dot(df, unit_vec))

    # Apply restoring/biasing forces along interatomic vector
    bias = np.zeros((constants.number_ions, 3))
    bias[ion1] = unit_vec * bias_cv
    bias[ion2] = -unit_vec * bias_cv

    with open("plugin_bias.txt", "a") as f:
        f.write(f"dist={dist:.6f} f_dft={f_dft:.6f} bias_cv={bias_cv:.6f}\n")

    return bias, f_dft


def calc_dihedral_forces(
    constants, ion1: int, ion2: int, ion3: int, ion4: int, bias_cv: float
) -> Tuple[np.ndarray, float]:
    """Project scalar dihedral bias force onto atomic Cartesian coordinates via analytical gradient."""
    box = _extract_box(constants)
    p1 = _cartesian_vector(constants.positions, ion1, "positions") * box
    p2 = _cartesian_vector(constants.positions, ion2, "positions") * box
    p3 = _cartesian_vector(constants.positions, ion3, "positions") * box
    p4 = _cartesian_vector(constants.positions, ion4, "positions") * box

    dih_cv = calc_dih(constants, ion1, ion2, ion3, ion4)
    g1, g2, g3, g4 = dihedral_gradient(p1, p2, p3, p4, box=box)

    # Atomic force: F_i = -bias_cv * (d phi / d r_i)
    bias = np.zeros((constants.number_ions, 3))
    bias[ion1] = -bias_cv * g1
    bias[ion2] = -bias_cv * g2
    bias[ion3] = -bias_cv * g3
    bias[ion4] = -bias_cv * g4

    return bias, bias_cv


def next_step() -> int:
    """Internal step counter tracking plugin invocations."""
    if not hasattr(next_step, "stepper"):
        next_step.stepper = 0
    next_step.stepper += 1
    return next_step.stepper


# =============================================================================
# PRIMARY VASP PLUGIN ENTRY POINT
# =============================================================================

def force_and_stress(constants, additions) -> None:
    """Interface hook called by VASP to augment atomic forces and stress tensor."""
    step = next_step() - 1

    # 1. Evaluate distance soft wall boundary
    wall_forces, f_wall_proj = calc_soft_wall(constants, dist_ion1, dist_ion2, r_max=4.0, k=5.0)
    additions.forces += wall_forces
    wall_active = (f_wall_proj != 0.0)

    # 2. Evaluate CV coordinates and model biases
    is_2d = (dih_ion4 != 0)

    if not is_2d:
        dist_cv = calc_dist(constants, dist_ion1, dist_ion2)
        dist_f_app, _ = get_model_bias_forces(dist_cv, 0.0, model_path=MODEL_DATA_PATH)
        dist_bias, _ = calc_dist_forces(constants, dist_ion1, dist_ion2, dist_f_app)
        dih_f_app = 0.0
    else:
        dist_cv = calc_dist(constants, dist_ion1, dist_ion2)
        dih_cv = calc_dih(constants, dih_ion1, dih_ion2, dih_ion3, dih_ion4)

        dist_f_app, gp_dih_force = get_model_bias_forces(dist_cv, dih_cv, model_path=MODEL_DATA_PATH)
        dist_bias, _ = calc_dist_forces(constants, dist_ion1, dist_ion2, dist_f_app)
        dih_bias, dih_f_app = calc_dihedral_forces(
            constants, dih_ion1, dih_ion2, dih_ion3, dih_ion4, gp_dih_force
        )
        additions.forces += dih_bias

    # 3. Suppress model bias if wall boundary restraint is active
    if wall_active:
        dist_f_app = 0.0
    else:
        additions.forces += dist_bias

    total_dist_f_app = f_wall_proj + dist_f_app

    # 4. Record applied forces for ABF force recovery (dWp/dt + f_app = f_dft)
    with open("f_app.txt", "a") as f:
        if is_2d:
            f.write(f"{total_dist_f_app:.8f} {dih_f_app:.8f}\n")
        else:
            f.write(f"{total_dist_f_app:.8f}\n")