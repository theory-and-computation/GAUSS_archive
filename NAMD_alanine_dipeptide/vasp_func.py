import os
import shutil
import subprocess
import pandas as pd
import torch
import jax
import jax.numpy as jnp


class VASP:


    def __init__(self):
        # self.cv1 = cv1
        # self.cv2 = cv2
        self.unit_scalar = 103.679  # convert force to eV/A

    def runner(self, logger):
        """This function runs VASP from the command line"""

        job = format(logger.iter, '05d')
        job_path = '/home/daisy/research/github/gpr_sampling/'+logger.dirtag
        root = '/home/daisy/research/github/gpr_sampling/butane_ru'
        
        # job_path = '/dfs7/elizaml4/dkamp/poly_catal/enhanced_sampling/4_gpr/2_butane/'+logger.dirtag
        # root = '/dfs7/elizaml4/dkamp/poly_catal/enhanced_sampling/4_gpr/2_butane/butane_ru'
        
        shutil.copytree(root, job_path, dirs_exist_ok=True)
        
        if logger.iter == 0:  # equilibration run, no biasing
            os.mkdir(f'{job_path}/{logger.iter}')
            os.rename(f'{job_path}/INCAR_eq', f'{job_path}/INCAR')
            os.rename(f'{job_path}/vasp_plugin_eq.py', f'{job_path}/vasp_plugin.py')
            
            subprocess.run(
                ["mpirun", "-np", "16", "/home/daisy/software/vasp.6.5.1/bin/vasp_std"], cwd=job_path)
            
            # subprocess.run(
            #     ["mpirun", "-np", "4", "-mca", "pml", "ucx", "/dfs7/elizaml4/jonis/01_vasp_compilations/vasp_plugin/bin/vasp_std"], cwd=job_path,)
            
            os.rename(f'{job_path}/INCAR', f'{job_path}/INCAR_eq')
            os.rename(f'{job_path}/vasp_plugin.py', f'{job_path}/vasp_plugin_eq.py')

            shutil.copyfile(f'{job_path}/XDATCAR', f'{job_path}/{logger.iter}')
            shutil.copyfile(f'{job_path}/vasprun.xml', f'{job_path}/{logger.iter}/')
            shutil.copyfile(f'{job_path}/OUTCAR', f'{job_path}/{logger.iter}/')
        
        else:  # production run, with biasing
            os.mkdir(f'{job_path}/{logger.iter}')
            os.rename(f'{job_path}/INCAR_run', f'{job_path}/INCAR')
            os.rename(f'{job_path}/vasp_plugin_run.py', f'{job_path}/vasp_plugin.py')
            
            subprocess.run(
                ["mpirun", "-np", "16", "/home/daisy/software/vasp.6.5.1/bin/vasp_std"], cwd=job_path,)
            
            os.rename(f'{job_path}/INCAR', f'{job_path}/INCAR_run')
            os.rename(f'{job_path}/vasp_plugin.py', f'{job_path}/vasp_plugin_run.py')

            shutil.copyfile(f'{job_path}/XDATCAR', f'{job_path}/{logger.iter}')
            shutil.copyfile(f'{job_path}/vasprun.xml', f'{job_path}/{logger.iter}')
            shutil.copyfile(f'{job_path}/OUTCAR', f'{job_path}/{logger.iter}')
        

        print(f'done running vasp job {logger.iter}')


    def collect_data(self, logger): 
        """This function collects NAMD data and formats for ABF"""
        job = format(logger.iter, '05d')
        colv_traj_file = str(job)+'.colvars.traj'
        cv_coord_file = str(job)+'.cv_coor.txt'
        job_path = '/home/daisy/research/github/gpr_sampling/'+logger.dirtag
        
        print(f"{logger.iter} calculating jacobians")
        jxi = self.calc_jac(job, logger, job_path)
        
        print(f"{logger.iter} calculating momentums")
        p = self.calc_mom(job, job_path)

        print(f"{logger.iter} grabbing cvs")
        cv1, cv2, f_applied1, f_applied2 = self.grab_cvs(job, job_path)
        
        return jxi, p, cv1, cv2, f_applied1, f_applied2


    @staticmethod
    def calc_jac(job, logger, job_path):
        """calculates the jacobian matrix at every timestep in the NAMD simulation"""

        box = [10.7662169169999995, 9.3308193240000001, 25.0782871245999992]

        f = open(f'/home/daisy/research/github/gpr_sampling/{logger.dirtag}/XDATCAR', 'r')
        list_all = []
        start = []
        sorted_all = []
        carbons = []
        hydrogen = []

        for x in f:
            temp = x.lstrip().split()
            list_all.append(temp)
        f.close()
        for x in range(len(list_all)):
            if list_all[0][0] == 'Direct':
                break
            start.append(list_all[0])
            del list_all[0]
        
        natoms = 0
        for x in range(len(start[6])):
            natoms = natoms + int(start[6][x])
        numberofsteps = int(len(list_all)/(natoms+1))
        for x in range(numberofsteps):
            temp = []
            del list[0]
            for y in range(natoms):
                list[0] = [float(i) for i in list[0]]
                temp.append(list[0])
                del list[0]
            sorted_all.append(temp)
            
        for x in range(len(sorted_all)):
            temp = []
            for y in range(int(start[6][0])):
                temp.append(sorted_all[x][0]) 
                del sorted_all[x][0]
            carbons.append(temp)



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
            jac = jax.jacrev(cv_fn)(atoms)           # evaluate Jacobian at atoms
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
