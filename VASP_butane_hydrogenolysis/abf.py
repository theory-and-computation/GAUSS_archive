"""
Adaptive Biasing Force (ABF) Mechanics and Generalized Force Accumulators.

This module provides routines for:
1. Periodic boundary condition (PBC) distance, dihedral, and radius of gyration (Rg) evaluation.
2. Analytic and PyTorch-autograd-driven Jacobian evaluations (J = d(CV)/dr).
3. Coordinate projection of atomic conjugate momenta onto collective variables:
   Wp = (J * J^T)^(-1) * J * p
4. Finite difference time-derivative evaluation (dWp/dt) to extract the generalized
   mean force field, compensating for applied model bias forces.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import py4vasp
import torch

import logger


# =============================================================================
# GEOMETRY & MINIMUM IMAGE CONVENTIONS
# =============================================================================

def calc_dist(pos1: np.ndarray, pos2: np.ndarray, box: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Calculate Euclidean distance and minimum-image wrapped coordinates under orthorhombic PBC.

    Parameters
    ----------
    pos1, pos2 : np.ndarray
        Cartesian coordinates of atoms 1 and 2.
    box : np.ndarray
        Simulation cell vector lengths [Lx, Ly, Lz].

    Returns
    -------
    dist : float
        Minimum image distance.
    pos1_out, pos2_wrapped : np.ndarray
        Coordinates with pos2 shifted to the nearest periodic image of pos1.
    """
    p1 = np.asarray(pos1, dtype=float)
    p2 = np.asarray(pos2, dtype=float)
    box = np.asarray(box, dtype=float)

    delta = p2 - p1
    delta -= box * np.round(delta / box)
    dist = float(np.linalg.norm(delta))

    return dist, p1, p1 + delta


def wrap_to_pi(angle: float) -> float:
    """Map an angle in radians to the periodic domain [-pi, pi)."""
    val = math.atan2(math.sin(angle), math.cos(angle))
    return -math.pi if val == math.pi else val


# =============================================================================
# BIAS FORCE LOOKUP
# =============================================================================

def get_bias_forces_at_cv(
    cv1: float,
    cv2: float,
    model_path: str | Path = "../../model_data.txt",
    cv2_type: str = "dihedral",
) -> Tuple[float, float]:
    """Query nearest-neighbor bias forces (f_cv1, f_cv2) from a grid table (model_data.txt).

    Parameters
    ----------
    cv1, cv2 : float
        Query collective variable coordinates.
    model_path : str or Path
        Path to space-delimited table with columns: x, y, f_cv1, f_cv2.
    cv2_type : str
        Collective variable type for cv2 ('dihedral' wraps along [-pi, pi)).

    Returns
    -------
    Tuple[float, float]
        Interpolated/nearest bias forces (f_cv1, f_cv2) in eV/Å (or eV/rad).
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model data file not found at: {path}")

    data = np.loadtxt(path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise ValueError(f"model_data.txt requires >= 4 columns; found {data.shape[1]}")

    x_arr = data[:, 0]
    y_arr = data[:, 1]
    f1_arr = data[:, 2]
    f2_arr = data[:, 3]

    qx = float(cv1)
    qy = wrap_to_pi(float(cv2)) if cv2_type == "dihedral" else float(cv2)

    dx = x_arr - qx
    if cv2_type == "dihedral":
        dy = np.arctan2(np.sin(y_arr - qy), np.cos(y_arr - qy))
    else:
        dy = y_arr - qy

    sq_dist = dx ** 2 + dy ** 2
    idx = int(np.argmin(sq_dist))

    return float(f1_arr[idx]), float(f2_arr[idx])


# =============================================================================
# HISTOGRAM & RUNNING FORCE SUM ACCUMULATORS
# =============================================================================

def update_hists_1d(
    d: float,
    dwp_dt: float | np.ndarray | None,
    log: logger.Logger,
    f_app: float,
) -> None:
    """Accumulate 1D visit counts and generalized force running sums."""
    indices = np.where(log.cv > np.array(d))[0]
    if len(indices) == 0 or indices[0] == 0:
        return

    index = indices[0] - 1

    if dwp_dt is not None:
        dwp_scalar = dwp_dt.item() if isinstance(dwp_dt, np.ndarray) else float(dwp_dt)

        log.histogram[index] += 1
        log.hist_new[index] += 1

        # Recover unbiased mean force: F_true = dWp/dt + F_applied
        total_force = dwp_scalar + float(f_app)
        log.xFsum[index] += round(total_force, 8)

        if log.histogram[index] > 0:
            log.xmean_forces[index] = round(log.xFsum[index] / log.histogram[index], 8)


def update_hists_2d(
    cv1_val: float,
    cv2_val: float,
    dwp_dt_cv1: float | np.ndarray | None,
    dwp_dt_cv2: float | np.ndarray | None,
    log: logger.Logger,
    cv1_name: str = "dist",
    cv2_name: str = "dihedral",
    f_app_cv1: float = 0.0,
    f_app_cv2: float = 0.0,
) -> None:
    """Accumulate 2D visit counts and generalized force running sums across grid bins."""
    if cv1_name == "dihedral":
        cv1_val = wrap_to_pi(cv1_val)
    if cv2_name == "dihedral":
        cv2_val = wrap_to_pi(cv2_val)

    grid_cv1 = np.asarray(getattr(log, f"{cv1_name}_cv").values)
    grid_cv2 = np.asarray(getattr(log, f"{cv2_name}_cv").values)

    above_x = np.where(grid_cv1 > cv1_val)[0]
    above_y = np.where(grid_cv2 > cv2_val)[0]

    if above_x.size == 0 or above_x[0] == 0 or above_y.size == 0 or above_y[0] == 0:
        return

    index_x = int(above_x[0]) - 1
    index_y = int(above_y[0]) - 1

    if dwp_dt_cv1 is None or dwp_dt_cv2 is None:
        return

    dt_cv1 = dwp_dt_cv1.item() if isinstance(dwp_dt_cv1, np.ndarray) else float(dwp_dt_cv1)
    dt_cv2 = dwp_dt_cv2.item() if isinstance(dwp_dt_cv2, np.ndarray) else float(dwp_dt_cv2)

    log.histogram[index_x, index_y] += 1
    tot_f1 = dt_cv1 + f_app_cv1
    tot_f2 = dt_cv2 + f_app_cv2

    log.xFsum[index_x, index_y] += round(tot_f1, 8)
    log.yFsum[index_x, index_y] += round(tot_f2, 8)

    n_counts = log.histogram[index_x, index_y]
    if n_counts > 0:
        log.xmean_forces[index_x, index_y] = round(log.xFsum[index_x, index_y] / n_counts, 8)
        log.ymean_forces[index_x, index_y] = round(log.yFsum[index_x, index_y] / n_counts, 8)


# =============================================================================
# GENERALIZED MOMENTUM (Wp) & FORCE TRAJECTORY PROJECTIONS
# =============================================================================

def fcv_1d(
    dist_cv: List[float],
    moms: List[np.ndarray],
    bmat: List[torch.Tensor],
    log: logger.Logger,
    iteration: int,
    cv1_name: str = "dist",
) -> None:
    """Project atomic momenta onto 1D CV and compute dWp/dt via finite differences."""
    n_frames = len(dist_cv)

    # Extract applied biasing forces
    if iteration == 0:
        f_app_list = np.zeros(n_frames, dtype=float)
    else:
        f_app_path = Path("f_app.txt")
        if f_app_path.exists():
            with open(f_app_path, "r") as f:
                lines = [line.strip() for line in f if line.strip()]
            picked = lines[::2] if len(lines) == 2 * n_frames else lines
            f_app_list = [float(line.split()[0]) for line in picked]
            f_app_path.unlink()
        else:
            f_app_list = np.zeros(n_frames, dtype=float)

    for i in range(n_frames):
        jxi = bmat[i].cpu().numpy()
        p = np.concatenate([moms[j][i] for j in range(len(moms))])

        # Project momentum: Wp = (J * (J * J^T)^(-1))^T * p
        jjt_inv = np.linalg.pinv(jxi @ jxi.T)
        w = jxi.T @ jjt_inv
        wp = float((w.T @ p).item())

        dwp_dt_eva = None
        if log.Wp_prevprev is not None:
            # 3-point backward finite difference: (1.5*Wp_n - 2*Wp_{n-1} + 0.5*Wp_{n-2}) / dt
            dwp_dt = (1.5 * wp - 2.0 * log.Wp_prev + 0.5 * log.Wp_prevprev) / log.dt
            dwp_dt_eva = dwp_dt * 103.679  # Conversion factor to eV/Å

        log.Wp_prevprev = log.Wp_prev
        log.Wp_prev = wp

        update_hists_1d(dist_cv[i], dwp_dt_eva, log, f_app_list[i])


def fcv_2d(
    cv1_vals: List[float],
    cv2_vals: List[float],
    cv1_moms: List[np.ndarray],
    cv2_moms: List[np.ndarray],
    cv1_jacobians: List[torch.Tensor],
    cv2_jacobians: List[torch.Tensor],
    log: logger.Logger,
    iteration: int,
    cv1_name: str = "dist",
    cv2_name: str = "dihedral",
    model_path: str | Path = "../../model_data.txt",
) -> None:
    """Project atomic momenta onto 2D CV space and accumulate ABF mean forces."""
    n_frames = len(cv1_vals)

    # Temperature history for diagnostics
    temp_pairs = [(-1, float("nan"))] * n_frames
    temp_path = Path(f"temp{iteration - 1}.txt")
    if temp_path.exists():
        with open(temp_path, "r") as f:
            lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        picked = lines[::2] if len(lines) == 2 * n_frames else lines
        parsed = []
        for line in picked:
            parts = line.split()
            if len(parts) == 2 and parts[0].isdigit():
                parsed.append((int(parts[0]), float(parts[1])))
        if len(parsed) == n_frames:
            temp_pairs = parsed

    grid_cv1 = np.asarray(getattr(log, f"{cv1_name}_cv").values)
    grid_cv2 = np.asarray(getattr(log, f"{cv2_name}_cv").values)
    wp_records = []

    for i in range(n_frames):
        jxi_1 = cv1_jacobians[i].cpu().numpy()
        jxi_2 = cv2_jacobians[i].cpu().numpy()

        p_1 = np.concatenate([cv1_moms[j][i] for j in range(len(cv1_moms))])
        p_2 = np.concatenate([cv2_moms[j][i] for j in range(len(cv2_moms))])

        v1 = wrap_to_pi(cv1_vals[i]) if cv1_name == "dihedral" else cv1_vals[i]
        v2 = wrap_to_pi(cv2_vals[i]) if cv2_name == "dihedral" else cv2_vals[i]

        # Retrieve applied bias forces from model lookup table
        if iteration == 0:
            f_app_1, f_app_2 = 0.0, 0.0
        else:
            try:
                f_app_1, f_app_2 = get_bias_forces_at_cv(v1, v2, model_path=model_path, cv2_type=cv2_name)
            except Exception:
                f_app_1, f_app_2 = 0.0, 0.0

        # Generalized momentum projections
        w1 = jxi_1.T @ np.linalg.pinv(jxi_1 @ jxi_1.T)
        wp_1 = float((w1.T @ p_1).item())

        w2 = jxi_2.T @ np.linalg.pinv(jxi_2 @ jxi_2.T)
        wp_2 = float((w2.T @ p_2).item())

        dwp_1_eva = None
        dwp_2_eva = None

        if log.Wp_prevprev is not None:
            prev_1, prev_2 = log.Wp_prev[0], log.Wp_prev[1]
            pprev_1, pprev_2 = log.Wp_prevprev[0], log.Wp_prevprev[1]

            dwp_1 = (1.5 * wp_1 - 2.0 * prev_1 + 0.5 * pprev_1) / log.dt
            dwp_2 = (1.5 * wp_2 - 2.0 * prev_2 + 0.5 * pprev_2) / log.dt

            dwp_1_eva = dwp_1 * 103.679
            dwp_2_eva = dwp_2 * 103.679

        log.Wp_prevprev = log.Wp_prev
        log.Wp_prev = np.array([wp_1, wp_2])

        if dwp_1_eva is not None and dwp_2_eva is not None:
            update_hists_2d(
                v1, v2, dwp_1_eva, dwp_2_eva, log,
                cv1_name=cv1_name, cv2_name=cv2_name,
                f_app_cv1=f_app_1, f_app_cv2=f_app_2
            )

            idx_x = np.where(grid_cv1 > v1)[0]
            idx_y = np.where(grid_cv2 > v2)[0]
            if idx_x.size > 0 and idx_x[0] != 0 and idx_y.size > 0 and idx_y[0] != 0:
                step, temp = temp_pairs[i]
                wp_records.append((step, v1, v2, int(idx_x[0]) - 1, int(idx_y[0]) - 1, dwp_1_eva, dwp_2_eva, temp))

    # Save trajectory diagnostic log
    with open(f"abf_wp{iteration}.txt", "w") as f:
        f.write(f"# step {cv1_name} {cv2_name} bin_x bin_y force_{cv1_name} force_{cv2_name} temp\n")
        for rec in wp_records:
            f.write(f"{rec[0]} {rec[1]:.6f} {rec[2]:.6f} {rec[3]} {rec[4]} {rec[5]:.8f} {rec[6]:.8f} {rec[7]:.4f}\n")


# =============================================================================
# PYTORCH AUTOGRAD JACOBIAN ROUTINES
# =============================================================================

def calculate_distance_cv_and_jacobian(
    structures: np.ndarray, a1: int, a2: int, box: np.ndarray, num_frames: int
) -> Tuple[List[torch.Tensor], List[float]]:
    """Compute distance CV and its analytical Jacobian via PyTorch Autograd."""
    distances_cv = []
    jxi = []

    for i in range(num_frames):
        pos1 = structures[i, a1]
        pos2 = structures[i, a2]
        dist, p1_corr, p2_corr = calc_dist(pos1, pos2, box)
        distances_cv.append(dist)

        p1_t = torch.tensor(p1_corr, dtype=torch.float32, requires_grad=True)
        p2_t = torch.tensor(p2_corr, dtype=torch.float32, requires_grad=True)

        dist_t = torch.sqrt(torch.sum((p2_t - p1_t) ** 2))
        dist_t.backward()

        jac = torch.zeros(1, 6)
        jac[0, :3] = p1_t.grad
        jac[0, 3:] = p2_t.grad
        jxi.append(jac)

    return jxi, distances_cv


def calculate_dihedral_cv_and_jacobian(
    structures: np.ndarray, da1: int, da2: int, da3: int, da4: int, box: np.ndarray, num_frames: int
) -> Tuple[List[torch.Tensor], List[float]]:
    """Compute dihedral angle CV and its analytical Jacobian via PyTorch Autograd."""
    dihedral_angles = []
    jxi = []

    for i in range(num_frames):
        p1 = structures[i, da1]
        p2 = structures[i, da2]
        p3 = structures[i, da3]
        p4 = structures[i, da4]

        # Fix p2 as origin reference; wrap neighboring atoms sequentially
        p2_corr = np.array(p2, dtype=float)
        _, _, p1_corr = calc_dist(p2, p1, box)
        _, _, p3_corr = calc_dist(p2_corr, p3, box)
        _, _, p4_corr = calc_dist(p3_corr, p4, box)

        p1_t = torch.tensor(p1_corr, dtype=torch.float32, requires_grad=True)
        p2_t = torch.tensor(p2_corr, dtype=torch.float32, requires_grad=True)
        p3_t = torch.tensor(p3_corr, dtype=torch.float32, requires_grad=True)
        p4_t = torch.tensor(p4_corr, dtype=torch.float32, requires_grad=True)

        dih_t = logger.dihedral_torch(p1_t, p2_t, p3_t, p4_t)
        dihedral_angles.append(wrap_to_pi(float(dih_t.item())))

        dih_t.backward()

        jac = torch.zeros(1, 12)
        jac[0, :3] = p1_t.grad
        jac[0, 3:6] = p2_t.grad
        jac[0, 6:9] = p3_t.grad
        jac[0, 9:12] = p4_t.grad
        jxi.append(jac)

    return jxi, dihedral_angles


def calculate_rg_cv_and_jacobian(
    structures: np.ndarray, atom_indices: List[int], atom_masses: List[float], box: np.ndarray, num_frames: int
) -> Tuple[List[torch.Tensor], List[float]]:
    """Compute Radius of Gyration (Rg) and its analytical Jacobian via PyTorch Autograd."""
    rg_values = []
    jxi = []
    n_atoms = len(atom_indices)
    masses_t = torch.tensor(atom_masses, dtype=torch.float32)

    for i in range(num_frames):
        # Sequential minimum-image wrapping around running center of mass
        corrected = [np.array(structures[i, atom_indices[0]], dtype=float)]
        running_com = corrected[0].copy()
        running_mass = atom_masses[0]

        for k in range(1, n_atoms):
            raw = structures[i, atom_indices[k]]
            _, _, wrapped = calc_dist(running_com, raw, box)
            corrected.append(wrapped)

            running_mass += atom_masses[k]
            running_com = (
                sum(atom_masses[j] * corrected[j] for j in range(k + 1)) / running_mass
            )

        tensors = [torch.tensor(p, dtype=torch.float32, requires_grad=True) for p in corrected]
        rg = logger.rg_torch(tensors, masses_t)
        rg_values.append(float(rg.item()))

        rg.backward()

        jac = torch.zeros(1, n_atoms * 3)
        for k, t in enumerate(tensors):
            jac[0, k * 3: k * 3 + 3] = t.grad
        jxi.append(jac)

    return jxi, rg_values


# =============================================================================
# TRAJECTORY PARSING & ABF PREPARATION
# =============================================================================

def manual_abf(log: logger.Logger, cv1_name: str = "dist", cv2_name: str | None = None):
    """Load VASP trajectory, extract atomic velocities, and compute CV Jacobians.

    Returns
    -------
    1D: (jxi, moms, cv_vals)
    2D: (jxi1, jxi2, moms1, moms2, cv1_vals, cv2_vals)
    """
    calc = py4vasp.Calculation.from_path(".")
    structures = calc.structure[:].cartesian_positions()
    all_velocities = calc.velocity[:].read()["velocities"]
    n_frames = int(all_velocities.shape[0])
    box = log.cell

    def _extract_cv(name: str):
        cv_obj = getattr(log, f"{name}_cv")
        indices = [atom.index for atom in cv_obj.atoms]
        masses = [atom.mass for atom in cv_obj.atoms]

        vels = [all_velocities[:, idx, :] for idx in indices]
        moms = [vel * m for vel, m in zip(vels, masses)]

        match name:
            case "dist":
                jxi, cv_vals = calculate_distance_cv_and_jacobian(structures, indices[0], indices[1], box, n_frames)
            case "dihedral":
                jxi, cv_vals = calculate_dihedral_cv_and_jacobian(structures, *indices[:4], box, n_frames)
            case "rg":
                jxi, cv_vals = calculate_rg_cv_and_jacobian(structures, indices, masses, box, n_frames)
            case _:
                raise ValueError(f"Unsupported CV identifier: {name}")

        return jxi, moms, cv_vals

    jxi1, moms1, cv1_vals = _extract_cv(cv1_name)

    if cv2_name is not None:
        jxi2, moms2, cv2_vals = _extract_cv(cv2_name)
        return jxi1, jxi2, moms1, moms2, cv1_vals, cv2_vals

    return jxi1, moms1, cv1_vals