"""
NAMD Simulation Engine Interface.

This module provides the `NAMD` wrapper class to:
1. Execute NAMD MD trajectories and post-process coordinates/velocities.
2. Read trajectory data from Colvars output files.
3. Compute CV Jacobians via automatic differentiation (JAX).
4. Compute generalized conjugate atomic momenta along CV degrees of freedom.
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
from typing import TYPE_CHECKING, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from alad_wats.extr_cv_coor import extract_cv_coor

if TYPE_CHECKING:
    import logger as modular_log

# Configurable VMD binary path via environment variable with a standard fallback
VMD_EXEC = os.getenv("VMD_EXEC", "vmd")


class NAMD:
    """Wrapper interface for running NAMD simulations and extracting ABF observables.

    Parameters
    ----------
    cv1 : str
        Name/type of first collective variable (e.g., 'dihedral').
    cv2 : str
        Name/type of second collective variable (e.g., 'dihedral').
    """

    def __init__(self, cv1: str, cv2: str):
        self.cv1: str = cv1
        self.cv2: str = cv2
        # Unit conversion scalar from internal NAMD/Charmm units to kcal/(mol*deg) or equivalent
        self.unit_scalar: float = 2390.06

    def runner(self, logger: modular_log.Logger) -> None:
        """Execute NAMD MD simulation and invoke coordinate/velocity extraction scripts.

        Parameters
        ----------
        logger : modular_log.Logger
            Active simulation state logger.
        """
        print(f"Iteration {logger.iter}: Running NAMD MD engine...", flush=True)

        job_str = f"{logger.iter:05d}"
        run_dir = Path(logger.dirtag).resolve()
        script_dir = run_dir / "alad_wats"

        # Execute MD run script: eq_exe.sh for step 0, run_exe.sh for subsequent iterations
        exe_script = script_dir / ("run_exe.sh" if logger.iter != 0 else "eq_exe.sh")
        subprocess.run(["bash", str(exe_script), job_str, str(run_dir)], check=True)

        print(f"Iteration {logger.iter}: NAMD execution complete. Extracting CV coordinates...", flush=True)
        extract_cv_coor(str(script_dir), job_str, str(run_dir))

        # Extract velocities using VMD script
        vmd_script = script_dir / "get_vel.tcl"
        subprocess.run(
            [
                VMD_EXEC,
                "-dispdev", "text",
                "-e", str(vmd_script),
                "-args", str(script_dir), job_str, str(run_dir),
            ],
            check=True,
        )

    def collect_data(
        self, logger: modular_log.Logger
    ) -> Tuple[jnp.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Gather CV values, applied forces, Jacobians, and conjugate momenta.

        Parameters
        ----------
        logger : modular_log.Logger
            Active simulation state logger.

        Returns
        -------
        jxi : jnp.ndarray
            Jacobian matrices along the trajectory, shape (N_steps, 2, 24).
        p : np.ndarray
            Conjugate momentum vectors along the trajectory, shape (N_steps, 24, 1).
        cv1, cv2 : np.ndarray
            Instantaneous values of the collective variables.
        f_applied1, f_applied2 : np.ndarray or None
            Biasing forces applied by the colvars module (None during equilibration).
        """
        job_str = f"{logger.iter:05d}"
        job_path = Path(logger.dirtag).resolve()

        print(f"Iteration {logger.iter}: Evaluating CV Jacobians via JAX...", flush=True)
        jxi = self.calc_jac(job_str, logger, str(job_path))

        print(f"Iteration {logger.iter}: Calculating conjugate atomic momenta...", flush=True)
        p = self.calc_mom(job_str, str(job_path))

        print(f"Iteration {logger.iter}: Loading trajectory CVs and applied forces...", flush=True)
        cv1, cv2, f_applied1, f_applied2 = self.grab_cvs(job_str, str(job_path))

        # Disk cleanup: delete output directories from 2 iterations ago to save quota
        if logger.iter > 1:
            prev_job_str = f"{logger.iter - 2:05d}"
            old_output_dir = job_path / "output" / prev_job_str
            if old_output_dir.is_dir():
                try:
                    shutil.rmtree(old_output_dir)
                except OSError as e:
                    print(f"Warning: Could not remove old output directory {old_output_dir}: {e}")

        return jxi, p, cv1, cv2, f_applied1, f_applied2

    @staticmethod
    def grab_cvs(
        job: str, job_path: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Read collective variable coordinates and applied bias forces from Colvars trajectory."""
        traj_file = Path(job_path) / "output" / job / f"{job}.colvars.traj"
        df = pd.read_csv(traj_file, sep=r"\s+", comment="#", header=None)

        cv1_vals = df[df.columns[1]].to_numpy()
        cv2_vals = df[df.columns[4]].to_numpy()

        if int(job) == 0:
            return cv1_vals, cv2_vals, None, None

        f_applied1 = df[df.columns[3]].to_numpy()
        f_applied2 = df[df.columns[6]].to_numpy()
        return cv1_vals, cv2_vals, f_applied1, f_applied2

    @staticmethod
    def calc_mom(job: str, job_path: str) -> np.ndarray:
        """Compute atomic momenta from velocity logs and atomic masses.

        Returns
        -------
        np.ndarray
            Momenta array of shape (N_steps, 24, 1) in units of A*amu/fs.
        """
        output_dir = Path(job_path) / "output" / job
        df1 = pd.read_csv(output_dir / f"{job}.cv1_vel.dat", sep=r"\s+")
        df2 = pd.read_csv(output_dir / f"{job}.cv2_vel.dat", sep=r"\s+")

        natoms = {
            "cv1": ["004", "006", "008", "014"],
            "cv2": ["006", "008", "014", "016"],
        }

        # Conversion factor for NAMD velocity units to internal momentum units
        vel_scale = np.float32(0.0204548)

        def _extract_velocities(df: pd.DataFrame, cv_key: str) -> np.ndarray:
            cols = []
            for atom_id in natoms[cv_key]:
                cols.extend([f"{{atom_{atom_id}_vx}}", f"{{atom_{atom_id}_vy}}", f"{{atom_{atom_id}_vz}}"])
            return df[cols].to_numpy(dtype=np.float32)

        vels_cv1 = _extract_velocities(df1, "cv1") * vel_scale
        vels_cv2 = _extract_velocities(df2, "cv2") * vel_scale
        vels_all = np.hstack([vels_cv1, vels_cv2])  # Shape: (N_steps, 24)

        # Atomic masses for each atom coordinate (C: 12.01, N: 14.0)
        mass_cv1 = [12.01, 12.01, 12.01, 14.00, 14.00, 14.00, 12.01, 12.01, 12.01, 12.01, 12.01, 12.01]
        mass_cv2 = [14.00, 14.00, 14.00, 12.01, 12.01, 12.01, 12.01, 12.01, 12.01, 14.00, 14.00, 14.00]
        mass_all = np.array(mass_cv1 + mass_cv2, dtype=np.float32)

        # Vectorized momentum calculation: p = m * v
        moms_all = vels_all * mass_all[None, :]  # Shape: (N_steps, 24)
        return moms_all[:, :, None]              # Reshape to (N_steps, 24, 1)

    @staticmethod
    def calc_jac(job: str, logger: modular_log.Logger, job_path: str) -> jnp.ndarray:
        """Compute the Jacobian of collective variables with respect to Cartesian coordinates."""
        output_dir = Path(job_path) / "output" / job
        df_cv1 = pd.read_csv(output_dir / f"{job}.cv1_coor.dat", sep=r"\s+")
        df_cv2 = pd.read_csv(output_dir / f"{job}.cv2_coor.dat", sep=r"\s+")

        natoms = {
            "cv1": ["004", "006", "008", "014"],
            "cv2": ["006", "008", "014", "016"],
        }

        atoms_cv1 = jnp.stack(
            [df_cv1[[f"{{atom_{a}_x}}", f"{{atom_{a}_y}}", f"{{atom_{a}_z}}"]].to_numpy() for a in natoms["cv1"]],
            axis=1,
        )
        atoms_cv2 = jnp.stack(
            [df_cv2[[f"{{atom_{a}_x}}", f"{{atom_{a}_y}}", f"{{atom_{a}_z}}"]].to_numpy() for a in natoms["cv2 daylight" if "cv2 daylight" in natoms else "cv2"]],
            axis=1,
        )

        def flat_jac(atom_coords, cv_type: str):
            cv_func = logger.cv_function(cv_type)
            jac = jax.jacrev(cv_func)(atom_coords)
            return jac.ravel()

        # Vectorized reverse-mode automatic differentiation over the trajectory
        vmap_jac_cv1 = jax.vmap(lambda a: flat_jac(a, logger.cv1))
        vmap_jac_cv2 = jax.vmap(lambda a: flat_jac(a, logger.cv2))

        jac_cv1 = vmap_jac_cv1(atoms_cv1)
        jac_cv2 = vmap_jac_cv2(atoms_cv2)

        # Assemble into block-diagonal structure for 2D collective variables
        zeros1 = jnp.zeros_like(jac_cv1)
        zeros2 = jnp.zeros_like(jac_cv2)

        jacs_all_steps = jnp.stack(
            [
                jnp.concatenate([jac_cv1, zeros2], axis=-1),
                jnp.concatenate([zeros1, jac_cv2], axis=-1),
            ],
            axis=1,
        )

        return jnp.array(jacs_all_steps)