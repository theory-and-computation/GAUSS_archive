import csv
import time
print(">>> Entered Python script", time.ctime(), flush=True)
import modular_log
import os
print("finished importing os", flush=True)
import numpy as np
print("finished importing numpy", flush=True)
from datetime import datetime
from vasp_func import VASP
from namd_func import NAMD
import shutil
from modular_gprmodel_clean import gp_model2D_gpder
import argparse
from pathlib import Path



debug = False

class JobRunner():
    def __init__(self, logger):
        self.logger = logger

    def __enter__(self):
        print("Enter")
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.logger.all_done()


def log_timing(timing_path, tag, iteration, namd_seconds, accumulation_seconds,
               gp_seconds):
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


def generate_simulation(logger, my_program, timing_path):
    print('starting simulation', flush=True)

    # equilibration run
    start = time.time()
    namd_start = time.perf_counter()
    my_program.runner(logger)  # initial equilibration run
    namd_seconds = time.perf_counter() - namd_start

    accumulation_start = time.perf_counter()
    jxi, p, cv1, cv2, f_app1, f_app2 = my_program.collect_data(logger)
    cv1_last, cv2_last = fcv(jxi, p, cv1, cv2, logger, my_program, f_app1, f_app2)
    accumulation_seconds = time.perf_counter() - accumulation_start

    print(f"{logger.iter} training new GPR model", flush=True)
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
    end = time.time()

    f = open("time.txt", "a")  # move this to the report writing
    f.write('eq step: ' + str(end-start) + 's' + '\n')
    f.close()

    # iterate through n times while updating the gpr bias
    for i in range(logger.num_iters):
        start = time.time()
        logger.iter = i+1
        gp_iterations(logger, my_program, timing_path)  # start looping over gpr interations
        end = time.time()
        f = open("time.txt", "a")  # move this to the report writing
        f.write('step' + str(i) + ': ' + str(end-start) + 's' + '\n')
        f.close()
    

def gp_iterations(logger, my_program, timing_path):
    logger.hist_new = np.zeros_like(logger.hist_new)

    namd_start = time.perf_counter()
    my_program.runner(logger)  # run the program
    namd_seconds = time.perf_counter() - namd_start

    accumulation_start = time.perf_counter()
    jxi, p, cv1, cv2, f_app1, f_app2 = my_program.collect_data(logger)
    cv1_last, cv2_last = fcv(jxi, p, cv1, cv2, logger, my_program, f_app1, f_app2)
    accumulation_seconds = time.perf_counter() - accumulation_start

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


def fcv(bmat, p, cv1, cv2, logger, my_program, f_app1, f_app2):

    num_data = len(bmat)

    # Compute generalized momenta
    for i in range(num_data):

        Jxi = bmat[i]  # jacobian of cv's 
        p_temp = p[i]  # momentums in A*amu/fs

        Wp = np.linalg.solve(Jxi @ Jxi.T, Jxi @ p_temp)

        xwp = Wp[0]
        ywp = Wp[1]
        logger.xwp_array.append(xwp)
        logger.ywp_array.append(ywp)

        xdWp_dt_converted, ydWp_dt_converted = None, None

        if len(logger.xwp_array) >= 3 and logger.dim2:
            n = len(logger.xwp_array) - 1  # index of current
            xdWp_dt = (-1.5*logger.xwp_array[n] + 2.0*logger.xwp_array[n-1] - 0.5*logger.xwp_array[n-2]) / logger.dt
            ydWp_dt = (-1.5*logger.ywp_array[n] + 2.0*logger.ywp_array[n-1] - 0.5*logger.ywp_array[n-2]) / logger.dt

            xdWp_dt_converted = xdWp_dt*my_program.unit_scalar
            ydWp_dt_converted = ydWp_dt*my_program.unit_scalar
        
        if len(logger.xwp_array) >= 3 and not logger.dim2:
            n = len(logger.xwp_array) - 1  # index of current
            xdWp_dt = (-1.5*logger.xwp_array[n] + 2.0*logger.xwp_array[n-1] - 0.5*logger.xwp_array[n-2]) / logger.dt
            xdWp_dt_converted = xdWp_dt*my_program.unit_scalar
    

        update_hists(cv1[i], cv2[i], xdWp_dt_converted, ydWp_dt_converted, logger, i, f_app1, f_app2)
    
    return cv1[-1], cv2[-1]
    
        

def abf_getindex(x, y, log): 
    """Gives the index corresponding to the nearest bin center."""
    dx = log.x_cv_grid[0, 1] - log.x_cv_grid[0, 0]
    dy = log.y_cv_grid[1, 0] - log.y_cv_grid[0, 0]
    
    # log.x_cv_grid[0, 0] is now the first center (e.g., -175)
    x0 = log.x_cv_grid[0, 0]
    y0 = log.y_cv_grid[0, 0]
    
    # Find nearest center by rounding
    x_index = int(np.round((x - x0) / dx))
    y_index = int(np.round((y - y0) / dy))
    
    # Clip to valid grid boundaries (or wrap with modulo if strictly periodic)
    nx = log.x_cv_grid.shape[1]
    ny = log.y_cv_grid.shape[0]
    x_index = np.clip(x_index, 0, nx - 1)
    y_index = np.clip(y_index, 0, ny - 1)
    
    return y_index, x_index


def update_hists(cv1, cv2, xdWp_dt, ydWp_dt, log, i, f_app1, f_app2):
    index = abf_getindex(cv1, cv2, log)

    if xdWp_dt is not None:
        log.hist[index] += 1
        log.hist_new[index] += 1
        
        if log.iter == 0:
            log.xfsum_mf[index] += round(float(xdWp_dt[0]), 8)  # Fsum is with ABF
            log.yfsum_mf[index] += round(float(ydWp_dt[0]), 8)  # Fsum is with ABF
        else:
            fapp_1 = f_app1[i]
            fapp_2 = f_app2[i]
            xtot_force = float(xdWp_dt[0])+float(fapp_1)
            ytot_force = float(ydWp_dt[0])+float(fapp_2)
            log.xfsum_mf[index] += round(xtot_force, 8)  # Fsum is with ABF
            log.yfsum_mf[index] += round(ytot_force, 8)  # Fsum is with ABF
        # mf from abf
        log.xmf_array_mf[index] = log.xfsum_mf[index] / log.hist[index]
        log.ymf_array_mf[index] = log.yfsum_mf[index] / log.hist[index]
        
        for i in range(0,log.x_cv_grid.shape[0]):
            if index[0] == i and index[1] == i:
                f = open(log.dirtag + '/wp'+str(i)+".txt", "a")
                f.write(f' {str(float(log.xmf_array_mf[i][i]))}  {str(float(log.xmf_array_us[i][i]))}  {str(log.hist[i][i])}  \n')
                f.close
    
    
def process_input():

    p = argparse.ArgumentParser(description="GPR/VASP orchestrator")
    p.add_argument("--tag", help="Run tag (e.g., s123)")
    p.add_argument("--dirtag", help="Directory tag or case name", default=None)
    args = p.parse_args()
    if debug:
        tag = '0'
    else:
        tag = args.tag
    
    today_time = datetime.today().strftime("%Y-%m-%d")
    if not os.path.isdir(today_time):
        os.mkdir(today_time)
    dirtag = today_time + '/' + tag
    
    if debug:
        shutil.rmtree(dirtag)
    
    if os.path.isdir(dirtag):
        raise Exception(f"{dirtag} is already a taken tag or subdirectory name.")
    os.mkdir(dirtag)
    os.mkdir(f'{dirtag}/data')
    
    src = Path("/dfs7/elizaml4/dkamp/poly_catal/enhanced_sampling/4_gpr/4_namd/alad_wats/")
    # os.mkdir(f'{dirtag}/alad_wats')
    dst1 = Path(f'{dirtag}/alad_wats')
    shutil.copytree(src, dst1)

    print(f"Tag {dirtag} successful.\n\n")
    return tag, dirtag


def program_factory(program_name, cv1=None, cv2=None):
    match program_name:
        case "vasp":
            return VASP()
        case "namd":
            return NAMD(cv1, cv2)
        case _:
            raise ValueError(f"Unknown program: {program_name}")


def move_slurm_logs(dirtag: str):
    """
    Move Slurm stdout/stderr files for this job into `dirtag`.

    Parameters
    ----------
    dirtag : str
        Destination directory path (e.g. /dfs7/.../2026-01-21/0).
    """
    job_id = os.environ.get("SLURM_JOB_ID")
    if not job_id:
        # Not running under Slurm, or environment not set
        print("SLURM_JOB_ID not found; skipping Slurm log move.", flush=True)
        return

    submit_dir = Path(os.environ.get("SLURM_SUBMIT_DIR", os.getcwd()))
    dest_dir = Path(dirtag)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Default Slurm filenames (adjust if #SBATCH -o/-e are custom)
    candidates = [
        submit_dir / f"slurm-{job_id}.out",
        submit_dir / f"slurm-{job_id}.err",
    ]

    for src in candidates:
        if src.exists():
            dst = dest_dir / src.name
            try:
                shutil.move(str(src), str(dst))
                print(f"Moved {src} -> {dst}", flush=True)
            except Exception as e:
                print(f"Warning: failed to move {src}: {e}", flush=True)
        else:
            print(f"Slurm log file not found: {src}", flush=True)

def main():
    print('starting main function', flush=True)
    num_gpr_iters = 5000
    
    cv1_min, cv1_max = -180, 180
    cv2_min, cv2_max = -180, 180
    cv1_step_size = 10
    cv2_step_size = 10

    # setting up mesh grid for 2D
    cv1_range = cv1_max-cv1_min
    y_range = cv2_max-cv2_min
    x_bins = int((cv1_range/cv1_step_size))
    y_bins = int((y_range/cv2_step_size))
    # x_1D = np.linspace(cv1_min, cv1_max, x_bins)
    # x_1D = np.arange(cv1_min, cv1_max+cv1_step_size, cv1_step_size)
    # # y_1D = np.linspace(cv2_min, cv2_max, y_bins)
    # y_1D = np.arange(cv2_min, cv2_max+cv2_step_size, cv2_step_size)

    # Center-bin coordinates: [-175, -165, ..., 175]
    x_1D = np.arange(cv1_min, cv1_max, cv1_step_size) + 0.5 * cv1_step_size
    y_1D = np.arange(cv2_min, cv2_max, cv2_step_size) + 0.5 * cv2_step_size
    x_cv, y_cv = np.meshgrid(x_1D, y_1D)

    x_cv, y_cv = np.meshgrid(x_1D, y_1D)  # 2D arrays of all the x vals and all the y vals
    

    # dihedral, angle, dist, coord, rg
    cv1 = 'dihedral'
    cv2 = 'dihedral'
    
    # 'namd', 'lammps', 'vasp'
    program_name = 'namd'

    my_program = program_factory(program_name, cv1, cv2)

    print(f'simulation set up in {program_name} with cvs: {cv1} and {cv2}', flush=True)

    tag, dirtag = process_input() 
    timing_path = Path("data") / "timing.log"
    logger_path = os.path.join(dirtag, "logger_state.pkl")
    if os.path.exists(logger_path):
        logger = modular_log.Logger.load_state(logger_path)
        print("Resuming from a saved state.")
    else:
        logger = modular_log.Logger(cv1, cv2, program_name, num_gpr_iters, tag, dirtag, x_cv, y_cv)
        print("Starting a new simulation.")

    print(f"Beginning process of {num_gpr_iters} iterations.", flush=True)
    start = time.time()

    x_flat = x_cv.flatten()
    y_flat = y_cv.flatten()
    fx_flat = np.zeros_like(x_flat)
    fy_flat = np.zeros_like(x_flat)

    # Stack columns
    data_dummy = np.column_stack((x_flat, y_flat, fx_flat, fy_flat)) #scaled
    np.savetxt(f'{dirtag}/alad_wats/model_data.txt', data_dummy, fmt="%.6f", header="x_cv y_cv fx_pred fy_pred", comments='')


    with JobRunner(logger):
        generate_simulation(logger, my_program, timing_path)
    
    move_slurm_logs(dirtag)

    end = time.time()
    print(f"This process took {end-start} seconds.", flush=True)



if __name__ == "__main__":
    print('starting main', flush=True)
    main()
