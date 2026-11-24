import numpy as np
import os
import torch
import pickle
# import jax
import jax.numpy as jnp

class Logger:
    def __init__(self, cv1, cv2, program, num_gpr_iters, tag, dirtag, x_cv, y_cv):
        if cv2 is None:
            self.dim2 = False
        else:
            self.dim2 = True

        self.cv1 = cv1
        self.cv2 = cv2

        self.dirtag = dirtag
        self.tag = tag
        self.program = program
        self.iter = 0
        self.step = 0
        self.num_iters = num_gpr_iters

        self.xwp_array = []  #W_p at each step
        self.ywp_array = []
        
        self.xfsum_us = np.zeros_like(y_cv, dtype=float)  # xfsum in each bin from us
        self.xfsum_mf = np.zeros_like(y_cv, dtype=float)  # fsum in each bin from mf
        self.yfsum_us = np.zeros_like(y_cv, dtype=float)  # yfsum in each bin from us
        self.yfsum_mf = np.zeros_like(y_cv, dtype=float)  # yfsum in each bin from mf
        self.hist = np.zeros_like(y_cv)  # historgram of counts in each bin
        self.hist_new = np.zeros_like(y_cv)  # recent historgram of counts in each bin

        self.xmf_array_us = np.zeros_like(y_cv, dtype=float)  # Mean force in each bin from us
        self.xmf_array_mf = np.zeros_like(y_cv, dtype=float)  # Mean force in each bin from mf
        self.ymf_array_us = np.zeros_like(y_cv, dtype=float)  # Mean  force in each bin from us
        self.ymf_array_mf = np.zeros_like(y_cv, dtype=float)  # Mean  force in each bin from mf

        self.lh = None
        self.model = None

        self.dt = 2  # time step in ps

        self.x_cv_grid = x_cv  # 2D array of all the X values
        self.y_cv_grid = y_cv  # 2D array of all the y values
        
        # self.init_variables(total_steps,x_bins, y_bins, x_cv, y_cv, dim2D)
        # self.new_counts = np.zeros(x_bins)

        self.xnormalizer = None
        self.ynormalizer = None
        
    def save_state(self, filepath):
        """Saves the logger state to a file using pickle."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Logger state saved to {filepath}")

    @staticmethod
    def load_state(filepath):
        """Loads a logger state from a file."""
        with open(filepath, 'rb') as f:
            logger = pickle.load(f)
        print(f"Logger state loaded from {filepath}")
        return logger
    
    def init_variables(self,total_steps,x_bins, y_bins, x_cv, y_cv, dim2D):
        
        self.atom1_xpositions = np.zeros(total_steps+1)  #Position of the particle at each step
        self.atom1_xvelocities = np.zeros(total_steps+1)  #x Velocity of the particle at each step
        self.energies = np.zeros(total_steps+1)  #Energy of the particle at each step
        self.xforces = np.zeros(total_steps+1)  #x Force on the particle at each step
        self.time_array = np.zeros(total_steps+1)  #Time at each step
        self.xwp_array = np.zeros(total_steps+1)  #W_p at each step
        self.total_xforce_array = np.zeros(total_steps+1)  #x Total force on the particle at each step

        self.lh = None
        self.model = None
        self.stddev = np.zeros(x_bins)
        self.max_steps = 0
        self.kbT = 0

        self.iter = 0

        match self.dim2:
            case True:  # system is in 2D
                """time arrays"""
                self.atom1_ypositions = np.zeros(total_steps+1)  #Position of the particle at each step
                self.atom1_yvelocities = np.zeros(total_steps+1)  #y Velocity of the particle at each step
                self.yforces = np.zeros(total_steps+1)  #y Force on the particle at each step
                self.ywp_array = np.zeros(total_steps+1)  #W_p at each step
                self.total_yforce_array = np.zeros(total_steps+1)  #y Total force on the particle at each step

                """2D arrays"""
                self.xfsum_us = np.zeros((x_bins, y_bins))  # xfsum in each bin from us
                self.xfsum_mf = np.zeros((x_bins, y_bins))  # fsum in each bin from mf
                self.yfsum_us = np.zeros((x_bins, y_bins))  # yfsum in each bin from us
                self.yfsum_mf = np.zeros((x_bins, y_bins))  # yfsum in each bin from mf
                self.hist = np.zeros((x_bins, y_bins))  # counts for each bin
                self.x_cv_grid = x_cv  # 2D array of all the X values
                self.y_cv_grid = y_cv  # 2D array of all the y values
                self.count_array = np.zeros((x_bins, y_bins))  # Number of visits to each bin
                self.xmf_array_us = np.zeros((x_bins, y_bins))  # Mean force in each bin from us
                self.xmf_array_mf = np.zeros((x_bins, y_bins))  # Mean force in each bin from mf
                self.ymf_array_us = np.zeros((x_bins, y_bins))  # Mean  force in each bin from us
                self.ymf_array_mf = np.zeros((x_bins, y_bins))  # Mean  force in each bin from mf
            
            case False:  # system is in 1D
                
                self.xfsum = np.zeros(x_bins)  # fsum in each bin
                self.hist = np.zeros(x_bins)  # counts in each bin
                self.x_cv_grid = x_cv[0, :]  # 1D array of all possible cv values
                self.count_array = np.zeros(x_bins)  # Number of visits to each bin at each step
                self.xmf_array = np.zeros(x_bins)  # Mean force in each bin

            case _:

                print('You did not choose a dimension')


    def dihedral_torch(self, atom1, atom2, atom3, atom4):
        b1 = atom2 - atom1
        b2 = atom3 - atom2
        b2_norm = b2 / torch.norm(b2)
        b3 = atom4 - atom3
        
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        m1 = torch.cross(n1, b2_norm)

        x = torch.dot(n1, n2)
        y = torch.dot(m1, n2)
        angle = torch.atan2(y, x)  # in radians

        return angle

    @staticmethod
    def cv_function(cv):
        
        match cv:
            case "dihedral":
                return Logger.dihedral_jax
            case "dist":
                return Logger.dist_jax
            case _:
                raise ValueError(f"Unknown program: {cv}")


    @staticmethod
    def dihedral_jax(atoms):
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
        
        return jnp.arctan2(y, x) * 180 / jnp.pi


    @staticmethod
    def dist_jax(atoms):
        atom1, atom2 = atoms
        return jnp.linalg.norm(atom2 - atom1)


    @staticmethod
    def dihedral_sinai(atom1, atom2, atom3, atom4):
        x1, y1, z1 = atom1[0], atom1[1], atom1[2]
        x2, y2, z2 = atom2[0], atom2[1], atom2[2]
        x3, y3, z3 = atom3[0], atom3[1], atom3[2]
        x4, y4, z4 = atom4[0], atom4[1], atom4[2]

        dx1, dy1, dz1 = x2-x1, y2-y1, z2-z1
        dx2, dy2, dz2 = x3-x2, y3-y2, z3-z2
        dx3, dy3, dz3 = x4-x3, y4-y3, z4-z3

        nx1, ny1, nz1 = dy1*dz2 - dz1*dy2, dz1*dx2 - dx1*dz2, dx1*dy2 - dy1*dx2
        nx2, ny2, nz2 = dy2*dz3 - dz2*dy3, dz2*dx3 - dx2*dz3, dx2*dy3 - dy2*dx3

        arg1 = dx2*( ny1*nz2 - ny2*nz1 ) + dy2*( nz1*nx2 - nz2*nx1 ) + dz2*( nx1*ny2 - ny1*nx2 )
        arg2 = np.sqrt(dx2*dx2 + dy2*dy2 + dz2*dz2) * (nx1*nx2 + ny1*ny2 + nz1*nz2)
        
        num = nx1*nx2 + ny1*ny2 + nz1*nz2
        den = np.sqrt(nx1*nx1 + ny1*ny1 + nz1*nz1) * np.sqrt(nx2*nx2 + ny2*ny2 + nz2*nz2)
        
        return np.arctan2(arg1, arg2) * 180 / np.pi
    
    
    @staticmethod
    def all_done():
        print('the simulation is finished :)')
    

class MinMaxNormalizer:
    def __init__(self, data, feature_range=(0.0, 1.0)):
        """
        data : array-like, shape [N, D]
            The dataset used to compute min and max for each feature.
        feature_range : tuple (min, max), default (0, 1)
            Desired range of transformed data.
        """
        self.data_min = np.min(data, axis=0)
        self.data_max = np.max(data, axis=0)
        self.range_min, self.range_max = feature_range

        # avoid divide by zero if some feature is constant
        self.scale = (self.range_max - self.range_min) / np.where(
            (self.data_max - self.data_min) != 0,
            (self.data_max - self.data_min),
            1.0)
        self.min_ = self.range_min - self.data_min * self.scale

    def normalize(self, x):
        """Transform to [range_min, range_max]."""
        return x * self.scale + self.min_

    def recover(self, x_norm):
        """Undo normalization (recover original scale)."""
        return (x_norm - self.min_) / self.scale


class StandardNormalizer:
    def __init__(self, data):
        """
        data : array-like, shape [N, D]
            The dataset used to compute mean and std.
        """
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # avoid division by zero
        self.std[self.std == 0] = 1.0  

    def normalize(self, x):
        """Transform to zero mean, unit variance."""
        return (x - self.mean) / self.std

    def recover(self, x_norm):
        """Undo normalization (recover original scale)."""
        return x_norm * self.std + self.mean
