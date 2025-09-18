import pickle
import numpy as np
import mdtraj as md
from itertools import combinations

class TICA_WRAPPER:
    """TICA wrapper for coordinate transformation."""
    def __init__(self, tica_model_path, pdb_path, tica_switch: bool = False):
        with open(tica_model_path, 'rb') as f:
            self.tica_model = pickle.load(f)
        self.pdb = md.load(pdb_path)
        self.ca_resid_pair = np.array(
            [(a.index, b.index) for a, b in combinations(list(self.pdb.topology.residues), 2)]
        )
        self.tica_switch = tica_switch
        self.r_0 = 0.8
        self.nn = 6
        self.mm = 12

    def transform(self, cad_data: np.ndarray):
        if self.tica_switch:
            cad_data = (1 - np.power(cad_data / self.r_0, self.nn)) / (1 - np.power(cad_data / self.r_0, self.mm))
        tica_cvs = self.tica_model.transform(cad_data)
        return tica_cvs

    def pos2cad(self, pos_data: np.ndarray):
        self.pdb.xyz = pos_data
        ca_pair_distances = md.compute_contacts(self.pdb, scheme="ca", contacts=self.ca_resid_pair, periodic=False)[0]
        return ca_pair_distances