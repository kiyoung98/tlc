import os
import json
import torch
import pprint
import mdtraj
import random
import pandas

import numpy as np

from tqdm import tqdm

from openmm import *
from openmm.app import *
from openmm.unit import *

from torch.utils.data import Dataset

from util.ic import load_ic_transform
from util.dataset_config import init_dataset_args



ALDP_PHI_ANGLE = [4, 6, 8, 14]
ALDP_PSI_ANGLE = [6, 8, 14, 16]

ALANINE_HEAVY_ATOM_IDX = [
    1, 4, 5, 6, 8, 10, 14, 15, 16, 18
]
CHIGNOLIN_HEAVY_ATOM_IDX = [
    0, 1, 2, 3, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    30, 31, 32, 33, 34, 35, 36, 37, 42, 43, 44, 45, 46, 47, 48,
    56, 57, 58, 59, 60, 61, 62, 63, 64,
    71, 72, 73, 74, 75, 76, 77, 85, 86, 87, 88,
    92, 93, 94, 95, 96, 97, 98, 106, 107, 108, 109,
    110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
    130, 131, 132, 133, 134
]

class CL_dataset(Dataset):
    def __init__(
        self,
        data_list,
        data_positive_list,
        data_negative_list,
        temperature_list,
        energy_list
    ):
        super(CL_dataset, self).__init__()
        self.device = "cpu"
        
        self.x = data_list.to(self.device)
        self.x_augmented = data_positive_list.to(self.device)
        self.x_augmented_hard = data_negative_list.to(self.device)
        self.temperature = temperature_list.to(self.device)
        self.energy = energy_list.to(self.device)
        
    def __getitem__(self, index):
        return self.x[index], self.x_augmented[index], self.x_augmented_hard[index], self.temperature[index], self.energy[index]

    def __len__(self):
        return self.x.shape[0]
 


def coordinate2distance(
    position,
    molecule = "alanine"
):
    position = position.reshape(-1, 3)
    if molecule == "alanine":
        heavy_atom_position = position[ALANINE_HEAVY_ATOM_IDX]
    elif molecule == "chignolin":
        heavy_atom_position = position[CHIGNOLIN_HEAVY_ATOM_IDX]
    else:
        raise ValueError(f"Molecule {molecule} not found")
    
    num_heavy_atoms = len(heavy_atom_position)
    distance = []
    for i in range(num_heavy_atoms):
        for j in range(i+1, num_heavy_atoms):
            distance.append(torch.norm(heavy_atom_position[i] - heavy_atom_position[j]))
    distance = torch.stack(distance)
    
    return distance

def traj2dataset(
    traj_list,
    cfg_list,
    energy_list,
    preprocess = "coordinate"
):
    molecule = args.molecule
    dataset_size = args.dataset_size
    number_of_traj = len(traj_list)
    xyz_list = []
    current_energy_list = []
    xyz_positive_list = []
    xyz_negative_list = []
    distance_list = []
    distance_positive_list = []
    distance_negative_list = []
    temperature_list = []
    
    random_idx_list = []
    for i in range(number_of_traj):
        time_horizon = cfg_list[i]["time"]
        random_idx_list.append(np.random.choice(traj_list[0].n_frames - 2 - args.negative_sample_augmentation, dataset_size, replace=True))
    
    for i in tqdm(
        range(dataset_size),
        desc = "Sampling frames from trajectories"
    ):
        '''
            - positive sample: next frame
            - negative sample: future frame
        '''
        for j in range(number_of_traj):
            frame_idx = random_idx_list[j][i]
            current_frame = torch.tensor(traj_list[j][frame_idx].xyz.squeeze())
            xyz_list.append(current_frame)
            distance_list.append(coordinate2distance(current_frame, molecule))
            current_energy_list.append(energy_list[j][frame_idx])
            
            positive_frame = torch.tensor(traj_list[j][frame_idx + args.positive_sample_augmentation].xyz.squeeze())
            xyz_positive_list.append(positive_frame)
            distance_positive_list.append(coordinate2distance(positive_frame, molecule))
            
            # augment_idx = np.min([time_horizon - frame_idx - 1, np.random.randint(1, args.negative_sample_augmentation)])
            negative_frame = torch.tensor(traj_list[j][frame_idx + args.negative_sample_augmentation].xyz.squeeze())
            xyz_negative_list.append(negative_frame)
            distance_negative_list.append(coordinate2distance(negative_frame, molecule))
            
            temperature_list.append(torch.tensor(cfg_list[j]["temperature"]))
    
    xyz_list = torch.stack(xyz_list)
    xyz_positive_list = torch.stack(xyz_positive_list)
    xyz_negative_list = torch.stack(xyz_negative_list)
    distance_list = torch.stack(distance_list)
    distance_positive_list = torch.stack(distance_positive_list)
    distance_negative_list = torch.stack(distance_negative_list)
    temperature_list = torch.stack(temperature_list)
    current_energy_list = torch.stack(current_energy_list)
    
    return (xyz_list, xyz_positive_list, xyz_negative_list, temperature_list, current_energy_list), \
        (distance_list, distance_positive_list, distance_negative_list, temperature_list, current_energy_list)

def compute_dihedral(
    positions: torch.Tensor
):
	"""http://stackoverflow.com/q/20305272/1128289"""
	def dihedral(p):
		if not isinstance(p, np.ndarray):
			p = p.numpy()
		b = p[:-1] - p[1:]
		b[0] *= -1
		v = np.array([v - (v.dot(b[1]) / b[1].dot(b[1])) * b[1] for v in [b[0], b[2]]])
		
		# Normalize vectors
		v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1, 1)
		b1 = b[1] / np.linalg.norm(b[1])
		x = np.dot(v[0], v[1])
		m = np.cross(v[0], b1)
		y = np.dot(m, v[1])
		
		return np.arctan2(y, x)

	angles = np.array(list(map(dihedral, positions)))
	return angles


args = init_dataset_args()

if __name__ == "__main__":
    traj_list = []
    energy_list = []
    cfg_list = []
    simulation_dir = f"./log/{args.molecule}/{args.temperature}"
    
    # Load trajectories
    for traj_dir in tqdm(
        args.traj_dir,
        desc = "Loading trajecatory files"
    ):
        dir = f"{simulation_dir}/{traj_dir}"
        with open(f"{dir}/args.json", 'r') as f:
            config = json.load(f)
            cfg_list.append(config)
            state = config["state"]
        
        # Read trajectory file
        if args.molecule == "alanine":
            pdb_file = f"./data/{args.molecule}-stable/{state}.pdb"
        elif args.molecule == "chignolin":
            pdb_file = f"./data/{args.molecule}/{state}.pdb"
        else:
            raise ValueError(f"Molecule {args.molecule} not found")
        loaded_traj = mdtraj.load(
            f"{dir}/traj.dcd",
            top=pdb_file
        )
        traj_list.append(loaded_traj)
        
        # Read energy file
        # scalars = pandas.read_csv(f"{dir}/scalars.csv", usecols=[1])
        # energy_list.append(torch.tensor(scalars.values.squeeze()))
        energy_list.append(torch.zeros(loaded_traj.n_frames))
    
    # Check dataset directory
    save_dir = f"../data/dataset/{args.molecule}/{args.temperature}/{args.dataset_version}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if os.path.exists(f"{save_dir}/cl-xyz.pt"):
        print(f"Dataset-xyz already exists at {save_dir}")
    if os.path.exists(f"{save_dir}/cl-distance.pt"):
        print(f"Dataset-distance already exists at {save_dir}")
    
    # Create CL dataset
    print("\n>> Building CL Dataset...")
    xyz_data, distance_data = traj2dataset(
        traj_list,
        cfg_list,
        energy_list,
        preprocess = "both"
    )
    if not os.path.exists(f"{save_dir}/cl-xyz.pt"):
        torch.save(
            CL_dataset(
                xyz_data[0],
                xyz_data[1],
                xyz_data[2],
                xyz_data[3],
                xyz_data[4]
            ),
            f"{save_dir}/cl-xyz.pt"
        )
        print(f"CL-xyz dataset saved at {save_dir}")
    if not os.path.exists(f"{save_dir}/cl-distance.pt"):
        torch.save(
            CL_dataset(
                distance_data[0],
                distance_data[1],
                distance_data[2],
                distance_data[3],
                distance_data[4],
            ),
            f"{save_dir}/cl-distance.pt"
        )
        print(f"CL-distance dataset saved at {save_dir}")
    
    cfg_list.append(vars(args))
    with open(f"{save_dir}/config.json", 'w') as f:
        json.dump(cfg_list, f, indent=4)
    print(f"Dataset created at {save_dir}")
    
    # Compute dihedral angles and save it
    dataset = torch.load(f"{save_dir}/cl-xyz.pt")
    current_state_list = []
    positive_state_list = []
    negative_state_list = []
    for data in tqdm(dataset):
        current_state, positive, negative, _, _ = data
        current_state_list.append(current_state)
        positive_state_list.append(positive)
        negative_state_list.append(negative)
    current_state_list = np.stack(current_state_list)
    positive_state_list = np.stack(positive_state_list)
    negative_state_list = np.stack(negative_state_list)
    
    state_phi_list = compute_dihedral(current_state_list[:, ALDP_PHI_ANGLE])
    state_psi_list = compute_dihedral(current_state_list[:, ALDP_PSI_ANGLE])
    positive_phi_list = compute_dihedral(positive_state_list[:, ALDP_PHI_ANGLE])
    positive_psi_list = compute_dihedral(positive_state_list[:, ALDP_PSI_ANGLE])
    negative_phi_list = compute_dihedral(negative_state_list[:, ALDP_PHI_ANGLE])
    negative_psi_list = compute_dihedral(negative_state_list[:, ALDP_PSI_ANGLE])
    np.save(f"{save_dir}/phi.npy", state_phi_list)
    np.save(f"{save_dir}/psi.npy", state_psi_list)
    np.save(f"{save_dir}/positive_phi.npy", positive_phi_list)
    np.save(f"{save_dir}/positive_psi.npy", positive_psi_list)
    np.save(f"{save_dir}/negative_phi.npy", negative_phi_list)
    np.save(f"{save_dir}/negative_psi.npy", negative_psi_list)
    
    print(f"Dihedral angles saved at {save_dir}")