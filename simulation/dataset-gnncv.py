import os
import json
import torch
import mdtraj as md
import numpy as np

from tqdm import tqdm
from util.dataset_config import init_dataset_args

import mlcolvar
import mlcolvar.graph as mg


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
        if isinstance(data, torch.Tensor) or isinstance(data, mlcolvar.graph.data.GraphDataSet):
            torch.save(data, f"{dir}/{name}")
        elif isinstance(data, np.ndarray):
            np.save(f"{dir}/{name}", data)
        else:
            raise ValueError(f"Data type {type(data)} not supported")
        print(f"{name} dataset saved at {dir}")


args = init_dataset_args()

if __name__ == "__main__":
    traj_list = []
    energy_list = []
    cfg_list = []
    simulation_dir = f"./log/{args.molecule}/{args.temperature}"
    print(f">> Building graph dataset {args.dataset_version}")
    
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
        traj_temp = md.load(f"{dir}/traj.gro", top=pdb_file)[::400]
        # random_idx = np.random.choice(traj_temp.n_frames - 2, args.dataset_size, replace=True)
        # traj_temp = traj_temp[random_idx]
        traj_temp.save_gro(f"{dir}/traj_sampled.gro")
        traj_list.append(f"{dir}/traj_sampled.gro")
    
    save_dir = f"../data/dataset/{args.molecule}/{args.temperature}/{args.dataset_version}"
    dataset = mg.utils.io.create_dataset_from_trajectories(
        trajectories=traj_list,
        # trajectories=["../base/gnncv/alad/data/A.gro", "../base/gnncv/alad/data/B.gro"],
        top=["./data/alanine-stable/c5.pdb", "./data/alanine-stable/c5.pdb"],
        cutoff=10,
        create_labels=True,
        system_selection='not type H',
    )
    # mg.data.save_dataset(dataset, f"{save_dir}/graph-dataset.pt")
    check_and_save(save_dir, "graph-dataset.pt", dataset)
    print(f"Dataset saved at {save_dir}/graph-dataset.pt")
