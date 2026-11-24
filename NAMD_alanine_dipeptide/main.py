"""
Main entry point for running iterative Gaussian Process Regression (GPR)-driven
enhanced sampling simulations (e.g., adaptive biasing force / ABF).

This script orchestrates:
1. Initial equilibration molecular dynamics simulation.
2. Data collection and generalized momentum/mean force accumulation along collective variables.
3. GPR model retraining and iterative bias field updates.
"""

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import shutil
import time

import numpy as np

# Local application imports
import logger as modular_log
from gpr_model import gp_model2D_gpder
from namd_func import NAMD
from vasp_func import VASP


# Default path template for input coordinate/topology files.
# Can be overridden via the TEMPLATE_DIR environment variable.
DEFAULT_TEMPLATE_DIR = Path(
    os.getenv("TEMPLATE_DIR", "/path/to/template/alad_wats")
)


class JobRunner:
    """Context manager to ensure simulation clean-up and state finalization."""

    def __init__(self, logger: modular_log.Logger):
        self.logger = logger

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.all_done()


def log_timing(
    timing_path: Path,
    tag: str,
    iteration: str | int,
    namd_seconds: float,
    accumulation_seconds: float,
    gp_seconds: float,
) -> None:
    """Append execution time benchmarks for simulation stages to a CSV file.

    Parameters
    ----------
    timing_path : Path
        Path to the output CSV timing file.
    tag : str
        Unique simulation or job identifier.
    iteration : str | int
        Current iteration index or stage label (e.g., 'equilibration').
    namd_seconds : float
        Time spent executing the MD engine.
    accumulation_seconds : float
        Time spent gathering coordinates and calculating generalized forces.
    gp_seconds : float
        Time spent training/evaluating the GPR model.
    """
    timing_path.parent.mkdir(parents=True, exist_ok=True)
    has_content = timing_path.exists() and timing_path.stat().st_size > 0

    with timing_path.open("a", newline="") as timing_file:
        writer = csv.writer(timing_file)
        if not has_content:
            writer.writerow([
                "timestamp",
                "tag",
                "iteration",
                "namd_seconds",
                "data_accumulation_seconds",
                "gp_training_seconds",
            ])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            tag,
            iteration,
            f"{namd_seconds:.6f}",
            f"{accumulation_seconds:.6f}",
            f"{gp_seconds:.6f}",
        ])


def generate_simulation(logger: modular_log.Logger, my_program, timing_path: Path) -> None:
    """Run the initial equilibration and subsequent iterative GPR sampling loop.

    Parameters
    ----------
    logger : modular_log.Logger
        Logger tracking simulation state, histograms, and CV grids.
    my_program : NAMD or VASP
        Simulation engine interface instance.
    timing_path : Path
        Destination path for timing logs.
    """
    print("Starting simulation workflow...", flush=True)

    # --- Equilibration Stage ---
    start_eq = time.time()
    
    namd_start = time.perf_counter()
    my_program.runner(logger)
    namd_seconds = time.perf_counter() - namd_start

    accum_start = time.perf_counter()
    jxi, p, cv1, cv2, f_app1, f_app2 = my_program.collect_data(logger)
    cv1_last, cv2_last = fcv(jxi, p, cv1, cv2, logger, my_program, f_app1, f_app2)
    accumulation_seconds = time.perf_counter() - accum_start

    print(f"Iteration {logger.iter}: Training initial GPR model", flush=True)
    gp_start = time.perf_counter()
    gp_model2D_gpder(logger, cv1_last, cv2_last)
    gp_seconds = time.perf_counter() - gp_start

    log_timing(
        timing_path,
        logger.tag,
        "equilibration",
        namd_seconds,
        accumulation_seconds,
        gp_seconds,
    )
    
    end_eq = time.time()
    with open("time.txt", "a") as f:
        f.write(f"eq step: {end_eq - start_eq:.4f}s\n")

    # --- Iterative Sampling Loop ---
    for i in range(logger.num_iters):
        start_step = time.time()
        logger.iter = i + 1
        gp_iterations(logger, my_program, timing_path)
        end_step = time.time()

        with open("time.txt", "a") as f:
            f.write(f"step {i}: {end_step - start_step:.4f}s\n")


def gp_iterations(logger: modular_log.Logger, my_program, timing_path: Path) -> None:
    """Execute a single production iteration: MD run, force accumulation, and GPR update."""
    logger.hist_new = np.zeros_like(logger.hist_new)

    namd_start = time.perf_counter()
    my_program.runner(logger)
    namd_seconds = time.perf_counter() - namd_start

    accum_start = time.perf_counter()
    jxi, p, cv1, cv2, f_app1, f_app2 = my_program.collect_data(logger)
    cv1_last, cv2_last = fcv(jxi, p, cv1, cv2, logger, my_program, f_app1, f_app2)
    accumulation_seconds = time.perf_counter() - accum_start

    gp_start = time.perf_counter()
    gp_model2D_gpder(logger, cv1_last, cv2_last)
    gp_seconds = time.perf_counter() - gp_start

    log_timing(
        timing_path,
        logger.tag,
        logger.iter,
        namd_seconds,
        accumulation_seconds,
        gp_seconds,
    )


def fcv(bmat, p, cv1, cv2, logger: modular_log.Logger, my_program, f_app1, f_app2):
    """Compute generalized momenta and mean force contributions along collective variables.

    Parameters
    ----------
    bmat : list or np.ndarray
        Jacobian matrices of collective variables with respect to Cartesian coordinates.
    p : list or np.ndarray
        Atomic momenta.
    cv1, cv2 : list or np.ndarray
        Collective variable values along the trajectory.
    logger : modular_log.Logger
        State tracking object.
    my_program : NAMD or VASP
        MD engine interface holding unit conversion scalars.
    f_app1, f_app2 : list or np.ndarray
        Applied biasing forces at each step.

    Returns
    -------
    tuple
        Final collective variable values (cv1[-1], cv2[-1]).
    """
    num_data = len(bmat)

    for i in range(num_data):
        jxi = bmat[i]       # Jacobian of CVs
        p_temp = p[i]       # Atomic momentum

        # Project momentum onto CV subspace: Wp = (J * J^T)^(-1) * J * p
        wp = np.linalg.solve(jxi @ jxi.T, jxi @ p_temp)
        xwp, ywp = wp[0], wp[1]
        logger.xwp_array.append(xwp)
        logger.ywp_array.append(ywp)

        xdwp_dt_converted, ydwp_dt_converted = None, None

        # 3-point backward finite difference for dWp/dt
        if len(logger.xwp_array) >= 3:
            n = len(logger.xwp_array) - 1
            xdwp_dt = (
                -1.5 * logger.xwp_array[n]
                + 2.0 * logger.xwp_array[n - 1]
                - 0.5 * logger.xwp_array[n - 2]
            ) / logger.dt
            xdwp_dt_converted = xdwp_dt * my_program.unit_scalar

            if logger.dim2:
                ydwp_dt = (
                    -1.5 * logger.ywp_array[n]
                    + 2.0 * logger.ywp_array[n - 1]
                    - 0.5 * logger.ywp_array[n - 2]
                ) / logger.dt
                ydwp_dt_converted = ydwp_dt * my_program.unit_scalar

        update_hists(
            cv1[i],
            cv2[i],
            xdwp_dt_converted,
            ydwp_dt_converted,
            logger,
            i,
            f_app1,
            f_app2,
        )

    return cv1[-1], cv2[-1]


def abf_getindex(x: float, y: float, log: modular_log.Logger) -> tuple[int, int]:
    """Find the 2D grid index of the bin center nearest to coordinate (x, y)."""
    dx = log.x_cv_grid[0, 1] - log.x_cv_grid[0, 0]
    dy = log.y_cv_grid[1, 0] - log.y_cv_grid[0, 0]

    x0 = log.x_cv_grid[0, 0]
    y0 = log.y_cv_grid[0, 0]

    x_index = int(np.round((x - x0) / dx))
    y_index = int(np.round((y - y0) / dy))

    # Clamp indices to grid bounds
    nx = log.x_cv_grid.shape[1]
    ny = log.y_cv_grid.shape[0]
    x_index = np.clip(x_index, 0, nx - 1)
    y_index = np.clip(y_index, 0, ny - 1)

    return y_index, x_index


def update_hists(
    cv1_val: float,
    cv2_val: float,
    xdwp_dt,
    ydwp_dt,
    log: modular_log.Logger,
    i: int,
    f_app1,
    f_app2,
) -> None:
    """Update histogram visit counts and running average mean force estimators."""
    index = abf_getindex(cv1_val, cv2_val, log)

    if xdwp_dt is not None:
        log.hist[index] += 1
        log.hist_new[index] += 1

        if log.iter == 0:
            log.xfsum_mf[index] += round(float(xdwp_dt[0]), 8)
            if ydwp_dt is not None:
                log.yfsum_mf[index] += round(float(ydwp_dt[0]), 8)
        else:
            xtot_force = float(xdwp_dt[0]) + float(f_app1[i])
            log.xfsum_mf[index] += round(xtot_force, 8)
            if ydwp_dt is not None:
                ytot_force = float(ydwp_dt[0]) + float(f_app2[i])
                log.yfsum_mf[index] += round(ytot_force, 8)

        # Average mean force calculation
        log.xmf_array_mf[index] = log.xfsum_mf[index] / log.hist[index]
        if ydwp_dt is not None:
            log.ymf_array_mf[index] = log.yfsum_mf[index] / log.hist[index]

        # Log diagonal slices for tracking/diagnostic purposes
        for diag_idx in range(log.x_cv_grid.shape[0]):
            if index[0] == diag_idx and index[1] == diag_idx:
                diag_log_path = Path(log.dirtag) / f"wp{diag_idx}.txt"
                with open(diag_log_path, "a") as f:
                    f.write(
                        f"{float(log.xmf_array_mf[diag_idx][diag_idx]):.6f}  "
                        f"{float(log.xmf_array_us[diag_idx][diag_idx]):.6f}  "
                        f"{log.hist[diag_idx][diag_idx]}\n"
                    )


def process_input() -> tuple[str, str]:
    """Parse CLI arguments and configure directory structure for the simulation run."""
    parser = argparse.ArgumentParser(
        description="Iterative GPR-driven enhanced sampling orchestrator."
    )
    parser.add_argument("--tag", required=True, help="Unique run tag (e.g., run_01, s123).")
    parser.add_argument("--dirtag", help="Custom output directory path (optional).", default=None)
    parser.add_argument(
        "--template-dir",
        type=Path,
        default=DEFAULT_TEMPLATE_DIR,
        help="Path to initial configuration and parameter templates.",
    )
    args = parser.parse_args()

    tag = args.tag
    today_str = datetime.today().strftime("%Y-%m-%d")
    dirtag = args.dirtag if args.dirtag else f"{today_str}/{tag}"

    out_dir = Path(dirtag)
    if out_dir.exists():
        raise FileExistsError(f"Directory '{dirtag}' already exists. Choose a unique tag.")

    out_dir.mkdir(parents=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)

    # Copy template input files into the destination execution folder
    dst_template = out_dir / "alad_wats"
    if args.template_dir.exists():
        shutil.copytree(args.template_dir, dst_template)
    else:
        print(f"Warning: Template directory {args.template_dir} not found. Skipping copy.")

    print(f"Run directory successfully created at: {dirtag}\n")
    return tag, dirtag


def program_factory(program_name: str, cv1: str = None, cv2: str = None):
    """Instantiate the corresponding simulation engine interface."""
    match program_name.lower():
        case "vasp":
            return VASP()
        case "namd":
            return NAMD(cv1, cv2)
        case _:
            raise ValueError(f"Unsupported simulation program: {program_name}")


def move_slurm_logs(dirtag: str) -> None:
    """Move Slurm standard output/error files into the run output directory."""
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        print("SLURM_JOB_ID not detected; skipping Slurm log transfer.", flush=True)
        return

    submit_dir = Path(os.environ.get("SLURM_SUBMIT_DIR", os.getcwd()))
    dest_dir = Path(dirtag)
    dest_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        submit_dir / f"slurm-{job_id}.out",
        submit_dir / f"slurm-{job_id}.err",
    ]

    for src in candidates:
        if src.exists():
            dst = dest_dir / src.name
            try:
                shutil.move(str(src), str(dst))
                print(f"Archived Slurm log: {src.name} -> {dst}", flush=True)
            except OSError as e:
                print(f"Warning: Failed to move {src.name}: {e}", flush=True)


def main():
    """Main orchestration routine."""
    num_gpr_iters = 5000

    # Grid boundaries and bin sizes (degrees or applicable units)
    cv1_min, cv1_max, cv1_step = -180, 180, 10
    cv2_min, cv2_max, cv2_step = -180, 180, 10

    # Center-bin coordinates (e.g., [-175, -165, ..., 175])
    x_1d = np.arange(cv1_min, cv1_max, cv1_step) + 0.5 * cv1_step
    y_1d = np.arange(cv2_min, cv2_max, cv2_step) + 0.5 * cv2_step
    x_cv, y_cv = np.meshgrid(x_1d, y_1d)

    cv1_type = "dihedral"
    cv2_type = "dihedral"
    program_name = "namd"

    my_program = program_factory(program_name, cv1_type, cv2_type)
    print(f"Initialized {program_name.upper()} with CVs: ({cv1_type}, {cv2_type})", flush=True)

    tag, dirtag = process_input()
    timing_path = Path(dirtag) / "data" / "timing.log"
    logger_path = Path(dirtag) / "logger_state.pkl"

    if logger_path.exists():
        logger = modular_log.Logger.load_state(str(logger_path))
        print("Resuming run from existing checkpoint.")
    else:
        logger = modular_log.Logger(
            cv1_type, cv2_type, program_name, num_gpr_iters, tag, dirtag, x_cv, y_cv
        )
        print("Starting a new simulation run.")

    # Initialize dummy bias force file
    x_flat = x_cv.flatten()
    y_flat = y_cv.flatten()
    zeros_flat = np.zeros_like(x_flat)
    initial_model_data = np.column_stack((x_flat, y_flat, zeros_flat, zeros_flat))

    model_data_file = Path(dirtag) / "alad_wats" / "model_data.txt"
    if model_data_file.parent.exists():
        np.savetxt(
            model_data_file,
            initial_model_data,
            fmt="%.6f",
            header="x_cv y_cv fx_pred fy_pred",
            comments="",
        )

    start_total_time = time.time()
    with JobRunner(logger):
        generate_simulation(logger, my_program, timing_path)

    move_slurm_logs(dirtag)
    elapsed_time = time.time() - start_total_time
    print(f"Simulation workflow completed in {elapsed_time:.2f} seconds.", flush=True)


if __name__ == "__main__":
    main()