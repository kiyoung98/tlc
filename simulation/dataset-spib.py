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

from util.dataset_config import init_dataset_args


ALDP_PHI_ANGLE = [4, 6, 8, 14]
ALDP_PSI_ANGLE = [6, 8, 14, 16]
ALDP_THETA_ANGLE = [1, 4, 6, 8]
ALDP_OMEGA_ANGLE = [8, 14, 16, 18]


def check_and_save(
    dir,
    name,
    data
):    
    if not os.path.exists(dir):
        os.makedirs(dir)
    if os.path.exists(f"{dir}/{name}"):
        print(f"{name} already exists at {dir}/{name}")
        pass
    else:
        if isinstance(data, torch.Tensor):
            torch.save(data, f"{dir}/{name}")
        elif isinstance(data, np.ndarray):
            np.save(f"{dir}/{name}", data)
        else:
            raise ValueError(f"Data type {type(data)} not supported")
        print(f"{name} dataset saved at {dir}")

def compute_dihedral(
    positions: torch.Tensor
) -> np.ndarray:
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

    # angles = np.array(list(map(dihedral, positions)))
    angle_list = []
    for position in positions:
        angle_list.append(dihedral(position))
    angle_list = np.stack(angle_list)

    return angle_list



def traj2dataset(
    traj_list,
    cfg_list,
):
    dataset_size = args.dataset_size
    number_of_traj = len(traj_list)
    dihedral_list = []
    label_list = []
    
    phi = np.pi
    phi_values = np.linspace(-phi, phi, 11)
    
    print(f"Sampling {dataset_size} frames from {number_of_traj} trajectories with {traj_list[0].n_frames} frames")
    random_idx = np.random.choice(traj_list[0].n_frames - 2 - args.negative_sample_augmentation, dataset_size, replace=True)
    for traj_idx in range(number_of_traj):
        for i in tqdm(
            range(dataset_size),
            desc = f"Sampling frames from trajectory {traj_idx}"
        ):
            frame_idx = random_idx[i]
            frame = traj_list[traj_idx][frame_idx]
            phi = compute_dihedral(frame.xyz.reshape(1, -1, 3).astype(np.float64)[:, ALDP_PHI_ANGLE])
            psi = compute_dihedral(frame.xyz.reshape(1, -1, 3).astype(np.float64)[:, ALDP_PSI_ANGLE])
            theta = compute_dihedral(frame.xyz.reshape(1, -1, 3).astype(np.float64)[:, ALDP_THETA_ANGLE])
            omega = compute_dihedral(frame.xyz.reshape(1, -1, 3).astype(np.float64)[:, ALDP_OMEGA_ANGLE])
            dihedral_list.append([phi, psi, theta, omega])
            label = np.searchsorted(phi_values, phi, side='right') - 1
            label_one_hot = np.zeros(10, dtype=np.float64)
            label_one_hot[label] = 1
            label_list.append(label_one_hot)
    
    dihedral_list = np.stack(dihedral_list).squeeze()
    label_list = np.stack(label_list)
    print(dihedral_list.shape, label_list.shape)

    return dihedral_list, label_list


args = init_dataset_args()

if __name__ == "__main__":
    traj_list = []
    energy_list = []
    cfg_list = []
    simulation_dir = f"./log/{args.molecule}/{args.temperature}"
    print(f">> Building timelag dataset {args.dataset_version}")
    
    # Load trajectories
    for traj_dir in tqdm(
        args.traj_dir,
        desc = "Loading trajecatory files"
    ):
        # Load configuration file
        dir = f"{simulation_dir}/{traj_dir}"
        with open(f"{dir}/args.json", 'r') as f:
            config = json.load(f)
            cfg_list.append(config)
            state = config["state"]
        
        # Load topology file from pdb
        if args.molecule == "alanine":
            pdb_file = f"./data/{args.molecule}-stable/{state}.pdb"
        elif args.molecule == "chignolin":
            pdb_file = f"./data/{args.molecule}/{state}.pdb"
        else:
            raise ValueError(f"Molecule {args.molecule} not found")
        
        # Load trajectory file
        loaded_traj = mdtraj.load(
            f"{dir}/traj.dcd",
            top=pdb_file
        )
        traj_list.append(loaded_traj)
    
    # Check dataset directory
    save_dir = f"../data/dataset/{args.molecule}/{args.temperature}/{args.dataset_version}"
    for name in ["dihedral.npy", "label.npy"]:
        if os.path.exists(f"{save_dir}/{name}"):
            print(f"{name} already exists at {save_dir}")
            exit()
    
    # Create timelag dataset
    print("\n>> Building SPIB Dataset...")
    dihedral_list, label_list = traj2dataset(
        traj_list,
        cfg_list,
    )
    
    check_and_save(dir = save_dir, name = "dihedral.npy", data = dihedral_list)
    check_and_save(dir = save_dir, name = "label.npy", data = label_list)

