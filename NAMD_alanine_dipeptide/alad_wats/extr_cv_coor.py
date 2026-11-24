#-----------------#
# extr_cv_coor.py #
#-----------------#
import MDAnalysis as mda 
import numpy as np
import sys 
import os


def extract_cv_coor(root_dir, job_id, out_path):

    # write function
    def write(atoms_sel, out_file):
        with open(out_file, 'w') as f:
            print(atoms_sel.indices)
            header_string = ' '.join(['{atom_' + f'{i:03d}' + '_x} {atom_' + f'{i:03d}' + '_y} {atom_' f'{i:03d}' + '_z}' for i in atoms_sel.indices])
            f.write('  {timestep} ' + header_string + ' \n')
            for ts in u.trajectory:
                coor = atoms_sel.positions
                coor_string = '   '.join([f'{x:>10.5f}' for a in coor for x in a])
                f.write(f' {ts.frame:>10}   ' + coor_string + '\n')


    # define topology and trajectory files
    topo = f"{root_dir}/components/topo/alad-wats_cross-term.psf"
    traj = f"{out_path}/output/{job_id}/{job_id}.dcd"
	
	# create universe
    u = mda.Universe(topo, traj)
	
	# CV atom indices (VMD indexing (0-based))
    phi = "4 6 8 14"
    psi = "6 8 14 16"
	
	# make atom selection in MDA
    cv1_sel = u.select_atoms(f"index {phi}") 
    cv2_sel = u.select_atoms(f"index {psi}") 
	
	# extract coordinates
    write(cv1_sel, f"{out_path}/output/{job_id}/{job_id}.cv1_coor.dat")
    write(cv2_sel, f"{out_path}/output/{job_id}/{job_id}.cv2_coor.dat")



# main -----------------------------#
if __name__ == "__main__":
    root_dir = sys.argv[1]
    job_id = sys.argv[2]
    out_path = sys.argv[3]

    print("*** Extracting collective variable coordinates ***")
    extract_cv_coor(root_dir, job_id, out_path)
#-----------------------------------#
