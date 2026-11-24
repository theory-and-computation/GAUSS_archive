"""
VASP Workflow Orchestrator for GPR-Enhanced Sampling.

This script coordinates multiple concurrent VASP walkers, managing:
1. Directory setup, file archiving, and POSCAR updates.
2. Concurrent parallel execution of VASP walkers via MPI (ibrun / mpirun).
3. Temperature monitoring and automated re-equilibration when thermal spikes occur.
4. Data collection, ABF mean force accumulation, and iterative GPR retraining.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import pickle
import re
import shutil
import subprocess
import time
from typing import List, Tuple

import imageio.v2 as imageio
import numpy as np

# Application imports
import abf
import gpr_model as gpr
import logger


# =============================================================================
# ENVIRONMENT & RUNTIME CONFIGURATION
# =============================================================================

# System execution limits and hardware mapping
TEMP_SPIKE_THRESHOLD: float = 1000.0
MAX_REEQUIL_ATTEMPTS: int = 5
DEFAULT_CORES_PER_JOB: int = int(os.getenv("CORES_PER_JOB", "47"))

# Binary paths configured via environment variables with standard fallbacks
VASP_BINARY: str = os.getenv("VASP_BINARY", "vasp_std")
VASP_LIB_PATH: str = os.getenv("VASP_LIB_PATH", "")
MPI_RUN_CMD: str = os.getenv("MPI_RUN_CMD", "ibrun")  # 'ibrun', 'mpirun', or 'srun'


# =============================================================================
# FILE & DIRECTORY MANAGEMENT
# =============================================================================

def save_files(folder_name: str | Path, iteration: int, number_of_runs: int, tag: str | None = None) -> None:
    """Archive VASP output files and promote CONTCAR to POSCAR for the next iteration.

    Parameters
    ----------
    folder_name : str or Path
        Base directory containing walker subfolders (e.g., 'runs').
    iteration : int
        Current simulation iteration index.
    number_of_runs : int
        Number of active parallel walkers.
    tag : str or None
        Optional suffix for re-equilibration archives (e.g., 'r1').
    """
    label = f"{iteration}_{tag}" if tag else str(iteration)
    subdirs = ["vaspxml", "xdatcar", "contcar", "oszicar"]
    base_dir = Path(folder_name)

    for run_i in range(1, number_of_runs + 1):
        run_dir = base_dir / f"run_{run_i:02d}"
        for d in subdirs:
            (run_dir / d).mkdir(parents=True, exist_ok=True)

        # Archive primary trajectory and output logs
        moves = [
            ("vasprun.xml", run_dir / f"vaspxml/vasprun_{label}.xml"),
            ("XDATCAR", run_dir / f"xdatcar/XDATCAR_{label}"),
        ]
        copies = [
            ("CONTCAR", run_dir / f"contcar/CONTCAR_{label}"),
            ("OSZICAR", run_dir / f"oszicar/OSZICAR_{label}"),
        ]

        for src, dst in moves:
            src_path = run_dir / src
            if src_path.exists():
                shutil.move(str(src_path), str(dst))

        for src, dst in copies:
            src_path = run_dir / src
            if src_path.exists():
                shutil.copy2(str(src_path), str(dst))

        # Promote latest coordinates for next iteration
        contcar = run_dir / "CONTCAR"
        poscar = run_dir / "POSCAR"
        if contcar.exists():
            shutil.move(str(contcar), str(poscar))


def setup_directories(folder_name: str | Path) -> None:
    """Reset and initialize clean working directories for a fresh run."""
    directories = [Path(folder_name), Path("./figs/gpr_plots"), Path("./data"), Path("./loggers")]
    for path in [Path(folder_name), Path("./figs"), Path("./data"), Path("./loggers")]:
        if path.exists():
            shutil.rmtree(path)

    for path in directories:
        path.mkdir(parents=True, exist_ok=True)


def save_logger(log: logger.Logger, run_i: int) -> None:
    """Serialize walker logger instance to disk."""
    out_path = Path(f"./loggers/logger_{run_i:02d}.pkl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(log, f)


# =============================================================================
# TEMPERATURE DIAGNOSTICS & CHECKS
# =============================================================================

def parse_temperatures(oszicar_path: str | Path) -> List[Tuple[int, float]]:
    """Parse instantaneous temperatures and ionic steps from an OSZICAR file."""
    temperatures: List[Tuple[int, float]] = []
    step = 0
    temp_pattern = re.compile(r"T=\s*([\d.]+(?:E[+-]?\d+)?)")
    ionic_step_pattern = re.compile(r"^\s*\d+\s+F=")

    osz_file = Path(oszicar_path)
    if not osz_file.exists():
        return temperatures

    with open(osz_file, "r") as f:
        for line in f:
            if ionic_step_pattern.match(line):
                step += 1
            match = temp_pattern.search(line)
            if match:
                current_step = step if step > 0 else len(temperatures) + 1
                temperatures.append((current_step, float(match.group(1))))
    return temperatures


def write_temperatures(temperatures: List[Tuple[int, float]], output_path: str | Path) -> None:
    """Save temperature history to a formatted text log."""
    with open(output_path, "w") as f:
        f.write(f"{'Step':>10}  {'Temperature (K)':>18}\n")
        f.write("-" * 32 + "\n")
        for step, temp in temperatures:
            f.write(f"{step:>10}  {temp:>18.4f}\n")


def all_runs_spiked(folder_name: str | Path, number_of_runs: int) -> bool:
    """Return True only if every walker exceeded TEMP_SPIKE_THRESHOLD.

    Missing OSZICAR files are treated as healthy to prevent false positives
    caused by asynchronous file-system delays.
    """
    base_dir = Path(folder_name)
    for run_i in range(1, number_of_runs + 1):
        oszicar = base_dir / f"run_{run_i:02d}" / "OSZICAR"
        if not oszicar.exists():
            print(f"[all_runs_spiked] OSZICAR missing for run_{run_i:02d} — treating as stable.")
            return False

        temperatures = parse_temperatures(oszicar)
        max_temp = max((t for _, t in temperatures), default=0.0)
        if max_temp <= TEMP_SPIKE_THRESHOLD:
            return False
    return True


# =============================================================================
# LOGGER & STATE INITIALIZATION
# =============================================================================

def initialize_logger(
    cv_grid: np.ndarray | List[np.ndarray],
    iconst: list,
    dim2: bool,
    cv1_name: str,
    cv2_name: str | None = None,
) -> logger.Logger:
    """Instantiate and populate a Logger with CV descriptors and unit cell dimensions."""
    cv_atom_indices = {
        "dist": [1, 3],
        "dihedral": [3, 1, 2, 0],
        "rg": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    }
    carbon_mass = 12.01

    def _get_timestep() -> float:
        incar_path = Path("./inputs/INCAR_1")
        with open(incar_path, "r") as f:
            for line in f:
                if "POTIM" in line:
                    return float(line.split("=")[1].strip())
        raise ValueError("POTIM tag not found in ./inputs/INCAR_1")

    def _get_cell() -> np.ndarray:
        poscar_path = Path("./inputs/POSCAR_1")
        with open(poscar_path, "r") as f:
            lines = f.read().splitlines()
        vecs = [[float(x) for x in lines[i].split()] for i in range(2, 5)]
        return np.array([vecs[0][0], vecs[1][1], vecs[2][2]])

    def _build_cv(name: str, cv_min: float, cv_max: float, num_bins: int):
        clean_name = name.lower()
        indices = cv_atom_indices[clean_name]
        atoms = [logger.Atom(index=idx, mass=carbon_mass, atom_type="C") for idx in indices]

        match clean_name:
            case "dist":
                return logger.CollectiveVariable("distance", atoms, cv_min, cv_max, num_bins)
            case "dihedral":
                return logger.CollectiveVariable("dihedral", atoms, cv_min, cv_max, num_bins)
            case "rg":
                return logger.CollectiveVariable("rg", atoms, cv_min, cv_max, num_bins)
            case _:
                raise ValueError(f"Unsupported CV identifier: '{name}'")

    log = logger.Logger(cv_grid, dim2)
    log.dim2 = dim2
    log.dt = _get_timestep()

    if not dim2:
        cv_obj = _build_cv(cv1_name, float(cv_grid[0]), float(cv_grid[-1]), len(cv_grid))
        setattr(log, f"{cv1_name}_cv", cv_obj)
    else:
        if cv2_name is None:
            raise ValueError("dim2=True requires cv2_name.")
        cv1_mesh, cv2_mesh = cv_grid
        setattr(
            log,
            f"{cv1_name}_cv",
            _build_cv(cv1_name, float(cv1_mesh.min()), float(cv1_mesh.max()), cv1_mesh.shape[1]),
        )
        setattr(
            log,
            f"{cv2_name}_cv",
            _build_cv(cv2_name, float(cv2_mesh.min()), float(cv2_mesh.max()), cv2_mesh.shape[0]),
        )

    log.cell = _get_cell()
    return log


def initialize_big_log(
    warm_start: bool,
    big_log: logger.Logger,
    dim2: bool,
    data_dir: str | Path = "./data",
) -> Tuple[logger.Logger, int]:
    """Restore ABF accumulated state from disk if warm-starting."""
    if warm_start:
        state_path = Path(data_dir) / "abf_state.npz"
        if not state_path.exists():
            raise FileNotFoundError(f"Checkpoint abf_state.npz not found at {state_path}")

        state = np.load(state_path)
        big_log.histogram = state["histogram"].copy()
        big_log.xFsum = state["xFsum"].copy()

        with np.errstate(invalid="ignore", divide="ignore"):
            big_log.xmean_forces = np.where(
                big_log.histogram > 0, big_log.xFsum / big_log.histogram, 0.0
            )

        if dim2:
            big_log.yFsum = state["yFsum"].copy()
            with np.errstate(invalid="ignore", divide="ignore"):
                big_log.ymean_forces = np.where(
                    big_log.histogram > 0, big_log.yFsum / big_log.histogram, 0.0
                )

        iteration = int(state["iteration"]) + 1
        big_log.Wp_prev = None
        big_log.Wp_prevprev = None

        print(f"[initialize_big_log] Restored state from {state_path} (resuming iter {iteration})")
        return big_log, iteration

    print("[initialize_big_log] Cold start: zeroing all accumulator grids.")
    big_log.histogram = np.zeros_like(big_log.histogram)
    big_log.xFsum = np.zeros_like(big_log.xFsum)
    big_log.xmean_forces = np.zeros_like(big_log.xmean_forces)
    if dim2:
        big_log.yFsum = np.zeros_like(big_log.yFsum)
        big_log.ymean_forces = np.zeros_like(big_log.ymean_forces)
    return big_log, 0


# =============================================================================
# RUN DIRECTORY PREPARATION
# =============================================================================

def expand_walkers(folder_name: str | Path, current_count: int, target_count: int) -> None:
    """Create additional walker directories cloned from run_01."""
    if target_count <= current_count:
        return

    base_dir = Path(folder_name)
    source_dir = base_dir / "run_01"
    if not source_dir.exists():
        raise FileNotFoundError(f"Clone template directory {source_dir} does not exist.")

    print(f"[expand_walkers] Cloning walkers {current_count} -> {target_count}")
    ignore_archives = shutil.ignore_patterns("vaspxml", "xdatcar", "contcar", "wavecar", "chgcar", "oszicar")

    for i in range(current_count + 1, target_count + 1):
        dst_dir = base_dir / f"run_{i:02d}"
        shutil.copytree(source_dir, dst_dir, ignore=ignore_archives, dirs_exist_ok=True)


def prepare_initial_dirs(txt, folder_name: str | Path, number_of_runs: int, original_inputs: Path) -> None:
    """Configure walker folders for iteration 0 (unbiased equilibration)."""
    poscar_files = sorted(
        [f for f in original_inputs.iterdir() if f.name.startswith("POSCAR_")],
        key=lambda x: int(re.search(r"(\d+)", x.name).group(1)),
    )
    num_poscars = len(poscar_files)
    block_size = math.ceil(float(number_of_runs) / max(1, num_poscars))
    base_dir = Path(folder_name)

    for i in range(1, number_of_runs + 1):
        run_dir = base_dir / f"run_{i:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        for item in original_inputs.iterdir():
            dst = run_dir / item.name
            shutil.copytree(item, dst, dirs_exist_ok=True) if item.is_dir() else shutil.copy2(item, dst)

        # Distribute template POSCAR files across walkers
        source_idx = min((i - 1) // block_size + 1, num_poscars)
        poscar_src = run_dir / f"POSCAR_{source_idx}"
        if poscar_src.exists():
            shutil.move(str(poscar_src), str(run_dir / "POSCAR"))

        incar_1 = run_dir / "INCAR_1"
        if incar_1.exists():
            shutil.move(str(incar_1), str(run_dir / "INCAR"))

        incar_2 = run_dir / "INCAR_2"
        if incar_2.exists():
            incar_2.unlink()

        txt.write(f"  run_{i:02d} initialized -> POSCAR_{source_idx}, INCAR_1\n")
    txt.flush()


def prepare_production_dirs(txt, folder_name: str | Path, number_of_runs: int, incar_source_name: str) -> None:
    """Distribute specified INCAR variant (INCAR_1 or INCAR_2) across all walkers."""
    incar_template = Path("./inputs") / incar_source_name
    base_dir = Path(folder_name)

    for i in range(1, number_of_runs + 1):
        run_dir = base_dir / f"run_{i:02d}"
        shutil.copy2(incar_template, run_dir / "INCAR")

        # Fallback to CONTCAR if POSCAR is missing
        poscar = run_dir / "POSCAR"
        if not poscar.exists():
            contcar = run_dir / "CONTCAR"
            if contcar.exists():
                shutil.copy2(contcar, poscar)


# =============================================================================
# PARALLEL VASP EXECUTION
# =============================================================================

def run_vasp_jobs(txt, folder_name: str | Path, number_of_runs: int, cores_per_job: int = DEFAULT_CORES_PER_JOB) -> None:
    """Launch concurrent VASP MPI processes across all walker directories."""
    processes = []
    env = os.environ.copy()
    if VASP_LIB_PATH:
        env["LD_LIBRARY_PATH"] = f"{VASP_LIB_PATH}:" + env.get("LD_LIBRARY_PATH", "")

    base_dir = Path(folder_name)

    for i in range(1, number_of_runs + 1):
        run_dir = base_dir / f"run_{i:02d}"
        offset = (i - 1) * cores_per_job

        txt.write(f"  Launching VASP: {run_dir.name} (core offset={offset})\n")
        txt.flush()

        stdout_file = open(run_dir / "vasp_stdout.log", "a")
        stderr_file = open(run_dir / "vasp_stderr.log", "a")

        # Command construction supports ibrun (TACC) or generic mpirun
        if MPI_RUN_CMD == "ibrun":
            cmd = ["ibrun", "-o", str(offset), "-n", str(cores_per_job), VASP_BINARY]
        else:
            cmd = [MPI_RUN_CMD, "-n", str(cores_per_job), VASP_BINARY]

        start_time = time.perf_counter()
        proc = subprocess.Popen(
            cmd, cwd=str(run_dir), env=env, stdout=stdout_file, stderr=stderr_file
        )
        processes.append((run_dir, proc, stdout_file, stderr_file, start_time))

    # Synchronize and monitor child processes
    for run_dir, proc, stdout_file, stderr_file, start_time in processes:
        try:
            proc.wait()
            elapsed = time.perf_counter() - start_time
            txt.write(f"  {run_dir.name} completed (exit code {proc.returncode}) in {elapsed:.2f}s\n")
            if proc.returncode != 0:
                raise RuntimeError(f"VASP execution failed in {run_dir} (code {proc.returncode})")
        finally:
            stdout_file.close()
            stderr_file.close()
            txt.flush()


# =============================================================================
# RE-EQUILIBRATION & STABILITY RECOVERY
# =============================================================================

def handle_reequilibration(
    txt, folder_name: str | Path, number_of_runs: int, iteration: int, inputs_dir: Path
) -> bool:
    """Execute fallback unbiased MD steps if all walkers exhibit thermal instability."""
    for attempt in range(1, MAX_REEQUIL_ATTEMPTS + 1):
        txt.write(f"[Re-equilibration] Attempt {attempt}/{MAX_REEQUIL_ATTEMPTS} for iteration {iteration}\n")
        txt.flush()

        prepare_production_dirs(txt, folder_name, number_of_runs, "INCAR_1")
        run_vasp_jobs(txt, folder_name, number_of_runs)
        save_files(folder_name, iteration, number_of_runs, tag=f"r{attempt}")

        if not all_runs_spiked(folder_name, number_of_runs):
            txt.write(f"[Re-equilibration] Walkers stabilized on attempt {attempt}.\n")
            return True

    txt.write(f"[Re-equilibration] WARNING: Walkers unrecovered after {MAX_REEQUIL_ATTEMPTS} attempts.\n")
    return False


# =============================================================================
# DATA ACCUMULATION & ITERATION LOGIC
# =============================================================================

def update_single_logger(
    txt, folder_name: str | Path, run_i: int, log: logger.Logger, log_iter: int,
    iteration: int, cv1_name: str, cv2_name: str | None = None
) -> Tuple[logger.Logger, bool]:
    """Extract trajectory data, compute generalized forces, and update individual walker log."""
    log_file = Path(f"./loggers/logger_{run_i:02d}.pkl")
    if log_file.exists():
        with open(log_file, "rb") as f:
            log = pickle.load(f)

    run_dir = Path(folder_name) / f"run_{run_i:02d}"
    oszicar_path = run_dir / "OSZICAR"

    temperatures = parse_temperatures(oszicar_path)
    write_temperatures(temperatures, run_dir / f"temp{iteration - 1}.txt")

    max_temp = max((t for _, t in temperatures), default=0.0)
    temp_spiked = max_temp > TEMP_SPIKE_THRESHOLD

    # Navigate to run directory to parse VASP outputs (OUTCAR/XDATCAR)
    cwd = os.getcwd()
    os.chdir(run_dir)
    try:
        if cv2_name is not None:
            jxi_1, jxi_2, p1, p2, val1, val2 = abf.manual_abf(log, cv1_name, cv2_name)
            abf.fcv_2d(val1, val2, p1, p2, jxi_1, jxi_2, log, log_iter, None, cv1_name, cv2_name)
        else:
            jxi, p, val1 = abf.manual_abf(log, cv1_name)
            abf.fcv_1d(val1, p, jxi, log, log_iter, log_iter, cv1_name)
    finally:
        os.chdir(cwd)

    save_logger(log, run_i)
    return log, temp_spiked


def run_iteration(
    txt, folder_name: str | Path, iteration: int, number_of_runs: int,
    log: logger.Logger, big_log: logger.Logger, cv1_name: str, cv2_name: str | None
) -> logger.Logger:
    """Execute a single workflow cycle: MD execution -> archiving -> force aggregation -> GPR."""
    inputs_dir = Path("./inputs")
    txt.write(f"\n--- Iteration {iteration} ---\n")
    txt.flush()

    # 1. Prepare run folders
    if iteration == 0:
        prepare_initial_dirs(txt, folder_name, number_of_runs, inputs_dir)
    else:
        prepare_production_dirs(txt, folder_name, number_of_runs, "INCAR_2")

    # 2. Parallel VASP execution
    run_vasp_jobs(txt, folder_name, number_of_runs)

    # 3. Archive outputs
    save_files(folder_name, iteration, number_of_runs)

    # 4. Check for thermal instability
    if iteration > 0 and all_runs_spiked(folder_name, number_of_runs):
        recovered = handle_reequilibration(txt, folder_name, number_of_runs, iteration, inputs_dir)
        if not recovered:
            txt.write("Instability unrecovered; skipping GPR update for this iteration.\n")
            return big_log

    # 5. Accumulate forces across all healthy walkers
    log_iter = big_log.iteration
    for run_i in range(1, number_of_runs + 1):
        log, spiked = update_single_logger(
            txt, folder_name, run_i, log, log_iter, iteration, cv1_name, cv2_name
        )
        if not spiked:
            big_log.histogram += log.histogram
            big_log.xFsum += log.xFsum
            if big_log.dim2:
                big_log.yFsum += log.yFsum

        mask = big_log.histogram > 0
        big_log.xmean_forces = np.where(mask, big_log.xFsum / big_log.histogram, 0.0)
        if big_log.dim2:
            big_log.ymean_forces = np.where(mask, big_log.yFsum / big_log.histogram, 0.0)

    # 6. Model retraining
    txt.write("Retraining GPR free-energy force model...\n")
    txt.flush()
    start_gpr = time.perf_counter()

    if big_log.dim2:
        gpr.gpr_pytorch_2d(iteration, big_log, cv1_name, cv2_name)
    else:
        gpr.gpr_pytorch_1d(iteration, big_log, cv1_name)

    txt.write(f"GPR model updated in {time.perf_counter() - start_gpr:.2f}s.\n")
    txt.flush()
    return big_log


# =============================================================================
# CLI & WORKFLOW ORCHESTRATION
# =============================================================================

def build_grid(cv_name: str) -> np.ndarray:
    """Build standardized 1D discretization grid for collective variables."""
    match cv_name.lower():
        case "dist":
            return np.linspace(1.0, 4.0, 50)
        case "dihedral":
            return np.linspace(-np.pi, np.pi, 50, endpoint=False)
        case "rg":
            return np.linspace(1.0, 5.0, 50)
        case _:
            raise ValueError(f"Unknown CV: {cv_name}")


def main():
    parser = argparse.ArgumentParser(description="VASP GPR Workflow Orchestrator")
    parser.add_argument("iterations", type=int, help="Number of workflow iterations")
    parser.add_argument("walkers", type=int, help="Number of concurrent VASP walkers")
    parser.add_argument("dims", type=int, choices=[1, 2], help="Collective variable dimensionality (1 or 2)")
    parser.add_argument("--cv1", type=str, default="dist", help="First collective variable")
    parser.add_argument("--cv2", type=str, default="dihedral", help="Second collective variable")
    parser.add_argument("--folder", type=str, default="runs", help="Base directory for walkers")
    parser.add_argument("--cold-start", action="store_true", help="Wipe directories and start cold")
    args = parser.parse_args()

    warm_start = not args.cold_start
    dim2 = args.dims == 2

    if not warm_start:
        setup_directories(args.folder)

    # Setup CV Grids
    grid1 = build_grid(args.cv1)
    if dim2:
        grid2 = build_grid(args.cv2)
        cv_grid = np.meshgrid(grid1, grid2)
    else:
        cv_grid = grid1

    # Load ICONST template
    with open("./inputs/ICONST", "r") as f:
        iconst = [line.split() for line in f.read().strip().splitlines() if line.strip()]

    log = initialize_logger(cv_grid, iconst, dim2, args.cv1, args.cv2 if dim2 else None)
    big_log = initialize_logger(cv_grid, iconst, dim2, args.cv1, args.cv2 if dim2 else None)
    big_log, start_iteration = initialize_big_log(warm_start, big_log, dim2, data_dir="./data")
    big_log.iteration = start_iteration

    if warm_start:
        for f in Path("./loggers").glob("logger_*.pkl"):
            f.unlink()

        # Expand walker directory layout if scaling up
        current_dirs = sorted(Path(args.folder).glob("run_*"))
        if args.walkers > len(current_dirs):
            expand_walkers(args.folder, len(current_dirs), args.walkers)

    with open("job_details.txt", "a") as txt:
        txt.write("===== VASP Enhanced Sampling Execution Log =====\n")
        for iteration in range(start_iteration, start_iteration + args.iterations):
            big_log = run_iteration(
                txt, args.folder, iteration, args.walkers, log, big_log, args.cv1, args.cv2 if dim2 else None
            )
            big_log.iteration += 1

        # Generate animated GIF of learning progression
        plot_files = sorted(
            Path("./figs/gpr_plots").glob("_gpr_*.png"),
            key=lambda p: int(re.search(r"(\d+)", p.name).group(1)),
        )
        if plot_files:
            images = [imageio.imread(str(p)) for p in plot_files]
            imageio.mimsave("./figs/gpr_evolution.gif", images)
            txt.write("Generated ./figs/gpr_evolution.gif\n")


if __name__ == "__main__":
    main()