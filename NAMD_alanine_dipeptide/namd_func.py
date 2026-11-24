import os
import numpy as np
import pandas as pd
import torch
import jax
import jax.numpy as jnp
from alad_wats.extr_cv_coor import extract_cv_coor


class NAMD:

    def __init__(self, cv1, cv2):
        self.cv1 = cv1
        self.cv2 = cv2
        self.unit_scalar = 2390.06 

    def runner(self, logger):
        """This function runs NAMD from the command line"""
        from pathlib import Path
        print(f"{logger.iter} running namd")
        dirtag = Path(logger.dirtag).resolve()
        job = format(logger.iter, '05d')
        # job_path = '/dfs7/elizaml4/dkamp/poly_catal/enhanced_sampling/4_gpr/4_namd/'+logger.dirtag
        # job_path = f'{logger.dirtag}'
        job_path = str(dirtag)
        # root = f'{logger.dirtag}/alad_wats/'
        root = dirtag / "alad_wats"

        # os.system(f"sed -i 's/^export jobstep=.*/export jobstep={job}/' exe.sh")
        
        if logger.iter != 0:
            os.system(f"bash {root}/run_exe.sh {job} {job_path}")
        else:
            os.system(f"bash {root}/eq_exe.sh {job} {job_path}")

        # os.system(f"bash {root}/run_exe.sh {job} {job_path}")

        print(f"{logger.iter} finished running NAMD")
        extract_cv_coor(root, job, job_path)  # write out the coordinated for each cv
        os.system(f"/dfs7/elizaml4/sinail/vmd/vmd-1.9.4a57/bin/vmd -dispdev text -e {logger.dirtag}/alad_wats/get_vel.tcl -args {root} {job} {job_path}")

    
    def collect_data(self, logger): 
        """This function collects NAMD data and formats for ABF"""
        job = format(logger.iter, '05d')
        colv_traj_file = str(job)+'.colvars.traj'
        cv_coord_file = str(job)+'.cv_coor.txt'
        job_path = '/dfs7/elizaml4/dkamp/poly_catal/enhanced_sampling/4_gpr/7_gauss_revisions_namd_gpr/'+logger.dirtag
        
        print(f"{logger.iter} calculating jacobians")
        jxi = self.calc_jac(job, logger, job_path)
        
        print(f"{logger.iter} calculating momentums")
        p = self.calc_mom(job, job_path)

        print(f"{logger.iter} grabbing cvs")
        cv1, cv2, f_applied1, f_applied2 = self.grab_cvs(job, job_path)
       
        import glob
        import shutil
        if logger.iter > 1:
            prevprev_job = format(logger.iter-2, '05d')
            output_dir = os.path.join(job_path, 'output', prevprev_job)
            keep_files = [] 
            if os.path.isdir(output_dir):
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        if file_path not in keep_files:
                            try:
                                os.remove(file_path)
                            except Exception as e:
                                print(f'Could not delete {file_path}: {e}')


        return jxi, p, cv1, cv2, f_applied1, f_applied2


    @staticmethod
    def grab_cvs(job, job_path):

        cv1, cv2 = 'phi', 'psi'

        if int(job) == 0:
            df = pd.read_csv(
            f"{job_path}/output/{job}/{job}.colvars.traj",
            sep='\s+',
            comment="#",
            header=None,)
            return df[df.columns[1]].to_numpy(), df[df.columns[4]].to_numpy(), None, None
        else:
            df = pd.read_csv(
            f"{job_path}/output/{job}/{job}.colvars.traj",
            sep='\s+',
            comment="#",
            header=None)
            return  df[df.columns[1]].to_numpy(), df[df.columns[4]].to_numpy(), df[df.columns[3]].to_numpy(), df[df.columns[6]].to_numpy()


    @staticmethod
    def calc_mom(job, job_path):
        """calculate the momentums at every timestep from the NAMD data"""
        # mom1, mom2 = [], []
        
        df1 = pd.read_csv(f"{job_path}/output/{job}/{job}.cv1_vel.dat", sep=r"\s+")
        df2 = pd.read_csv(f"{job_path}/output/{job}/{job}.cv2_vel.dat", sep=r"\s+")
        natoms = {
            "cv2": ['006', '008', '014', '016'],
            "cv1": ['004', '006', '008', '014']}
        
        def get_velocities(df, cv):
            cols = []
            for a in natoms[cv]:
                cols.extend([f'{{atom_{a}_vx}}', f'{{atom_{a}_vy}}', f'{{atom_{a}_vz}}'])
            # Convert directly to float32 to save memory
            return df[cols].to_numpy(dtype=np.float32)

            # shape: (nsteps, 12), float32
        vels_cv1 = get_velocities(df1, "cv1") * np.float32(0.0204548)
        vels_cv2 = get_velocities(df2, "cv2") * np.float32(0.0204548)

        # (nsteps, 24) velocities
        vels_all = np.hstack([vels_cv1, vels_cv2])  # float32

        # masses, broadcast over steps
        mass1 = np.array([12.01, 12.01, 12.01, 14, 14, 14, 12.01, 12.01, 12.01, 12.01, 12.01, 12.01],
                        dtype=np.float32)
        mass2 = np.array([14, 14, 14, 12.01, 12.01, 12.01, 12.01, 12.01, 12.01, 14, 14, 14],
                        dtype=np.float32)
        
        
        moms = []
        for step in vels_all:
            mom1 = step[:12]*mass1  # cv1 moms
            mom2 = step[12:]*mass2  # cv2 moms
            mom_tot = np.concatenate((mom1, mom2)).reshape(-1, 1)
            moms.append(mom_tot)

        return np.array(moms)

    @staticmethod
    def calc_jac(job, logger, job_path):
        """calculates the jacobian matrix at every timestep in the NAMD simulation"""

        natoms = {"cv2": ['006', '008', '014', '016'], "cv1": ['004', '006', '008', '014']}
        df_cv1 = pd.read_csv(f'{job_path}/output/{job}/{job}.cv1_coor.dat', sep='\s+')  # cv1
        df_cv2 = pd.read_csv(f'{job_path}/output/{job}/{job}.cv2_coor.dat', sep='\s+')  # cv2
        
        atoms_cv1 = jnp.stack(
        [df_cv1[[f'{{atom_{a}_x}}', f'{{atom_{a}_y}}', f'{{atom_{a}_z}}']].to_numpy() for a in natoms['cv1']],
        axis=1)
        atoms_cv2 = jnp.stack(
        [df_cv2[[f'{{atom_{a}_x}}', f'{{atom_{a}_y}}', f'{{atom_{a}_z}}']].to_numpy() for a in natoms['cv2']],
        axis=1)

        # flattened jacobian for jnp
        def flat_jac(atoms, cv_type):
            cv_fn = logger.cv_function(cv_type)   # get the CV function
            jac = jax.jacrev(cv_fn)(atoms)        # evaluate Jacobian at atoms
            return jac.ravel() 
        
        # vectorize both
        vmap_jac_cv1 = jax.vmap(lambda a: flat_jac(a, logger.cv1))
        vmap_jac_cv2 = jax.vmap(lambda a: flat_jac(a, logger.cv2))

        # evaluate
        jac_cv1 = vmap_jac_cv1(atoms_cv1)
        jac_cv2 = vmap_jac_cv2(atoms_cv2)

        # pad with zeros so cv1 & cv2 sit in separate blocks
        zeros1 = jnp.zeros_like(jac_cv1)
        zeros2 = jnp.zeros_like(jac_cv2)

        jacs_all_steps = jnp.stack([jnp.concatenate([jac_cv1, zeros2], axis=-1),   # cv1 in left block
                jnp.concatenate([zeros1, jac_cv2], axis=-1)], axis=1)

        return jnp.array(jacs_all_steps)
