import pickle
import numpy as np
import mdtraj as md

class TICA_WRAPPER:
    """TICA wrapper for coordinate transformation."""
    def __init__(self, tica_model_path, pdb_path, tica_switch: bool = False):
        with open(tica_model_path, 'rb') as f:
            self.tica_model = pickle.load(f)
        self.pdb = md.load(pdb_path)
        self.tica_switch = tica_switch
        self.r_0 = 0.8
        self.nn = 6
        self.mm = 12
        
        # Pre-compute frequently used attributes to avoid repeated calculations
        self.ca_atoms = self.pdb.topology.select('name CA')
        self.n_ca = len(self.ca_atoms)
        self.ca_topology = self.pdb.topology.subset(self.ca_atoms)
        self.ca_pairs = np.array([[i, j] for i in range(self.n_ca) for j in range(i+1, self.n_ca)])

    def transform(self, cads: np.ndarray):
        if self.tica_switch:
            cads = (1 - np.power(cads / self.r_0, self.nn)) / (1 - np.power(cads / self.r_0, self.mm))
        tica_cvs = self.tica_model.transform(cads)
        return tica_cvs

    def pos2cad(self, ca_positions: np.ndarray):
        ca_traj = md.Trajectory(ca_positions, self.ca_topology)
        return md.compute_distances(ca_traj, self.ca_pairs)