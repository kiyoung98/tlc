import os
import random
import mdtraj
import pandas
import datetime

from rdkit import Chem
from  tqdm import tqdm
from rdkit.Chem import AllChem, rdMolTransforms
from rdkit.Chem import Draw

import numpy as np
import mdtraj as md


phi_idx1, phi_idx2, phi_idx3, phi_idx4 = 4, 6, 8, 14
psi_idx1, psi_idx2, psi_idx3, psi_idx4 = 6, 8, 14, 16

def set_dihedral(mol, idx_1, idx2, idx3, idx4, angle):
    conf = mol.GetConformer()
    rdMolTransforms.SetDihedralDeg(conf, idx_1, idx2, idx3, idx4, angle)
    
    
print("Genearting alanine dipeptide with various dihedral angles")
state = "c5"
molecule = Chem.MolFromPDBFile(f'../../data/alanine/{state}.pdb', removeHs=False)
for psi in tqdm(np.linspace(-178, 178, 90, endpoint=True), desc="psi"):
    for phi in np.linspace(-178, 178, 90, endpoint=True):
        set_dihedral(molecule, psi_idx1, psi_idx2, psi_idx3, psi_idx4, psi)
        set_dihedral(molecule, phi_idx1, phi_idx2, phi_idx3, phi_idx4, phi)

        new_file_name = f"../../data/projection/alanine/c5_{psi:.1f}_{phi:.1f}.pdb"
        if not os.path.exists(new_file_name):
            Chem.MolToPDBFile(molecule, new_file_name)
        else:
            print(f"File {new_file_name} already exists")
state = "c7ax"
molecule = Chem.MolFromPDBFile(f'../../data/alanine/{state}.pdb', removeHs=False)
for psi in tqdm(np.linspace(-176, 176, 89, endpoint=True), desc="psi"):
    for phi in np.linspace(-176, 176, 89, endpoint=True):
        set_dihedral(molecule, psi_idx1, psi_idx2, psi_idx3, psi_idx4, psi)
        set_dihedral(molecule, phi_idx1, phi_idx2, phi_idx3, phi_idx4, phi)

        new_file_name = f"../../data/projection/alanine/c7ax_{psi:.1f}_{phi:.1f}.pdb"
        if not os.path.exists(new_file_name):
            Chem.MolToPDBFile(molecule, new_file_name)
        else:
            print(f"File {new_file_name} already exists")
print("Done!!")