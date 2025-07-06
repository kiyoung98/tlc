import torch
import wandb

import numpy as np

import openmm as mm
import openmm.unit as unit

from openmm import *
from openmm.app import *

from tqdm import tqdm

from .util.constant import *
from .util.angle import compute_dihedral
from .util.rotate import kabsch_rmsd

from .simulation.dynamics import load_forcefield, load_system

pairwise_distance = torch.cdist


def potential_energy(
    cfg,
    trajectory
):
    energy_list = np.zeros(shape=(trajectory.shape[0], 1))
    molecule = cfg.data.molecule
    pbb_file_path = f"../simulation/data/{molecule}/{cfg.steeredmd.start_state}.pdb"
    simulation = init_simulation(cfg, pbb_file_path)
    
    for idx, frame in enumerate(trajectory):
        try:
            simulation = set_simulation(simulation, frame)
            energy = simulation.context.getState(getEnergy=True).getPotentialEnergy()
            # energy_list.append(energy._value)
            energy_list[idx] = energy._value
        
        except Exception as e:
            print(f"Error in computing energy: {e}")
            energy_list.append(np.inf)
    
    return energy_list


def compute_thp(
    cfg,
    trajectory_list,
):
    device = trajectory_list.device
    molecule = cfg.steeredmd.molecule
    sample_num = cfg.steeredmd.sample_num
    hit_cnt = 0
    hit_mask = []
    hit_index = []
    
    if molecule == "alanine":
        goal_state_info = torch.load(f"../simulation/data/{molecule}/{cfg.steeredmd.goal_state}.pt")
        phi_goal, psi_goal = goal_state_info["phi"].to(device), goal_state_info["psi"].to(device)
        for i in tqdm(
            range(sample_num),
            desc = f"Computing THP for {trajectory_list.shape[0]} trajectories"
        ):
            psi = compute_dihedral(trajectory_list[i, :, ALDP_PSI_ANGLE])
            phi = compute_dihedral(trajectory_list[i, :, ALDP_PHI_ANGLE])
            distance_to_goal = torch.pow(psi - psi_goal, 2) + torch.pow(phi - phi_goal, 2)
            hit_in_path = distance_to_goal < (ALANINE_CV_BOUND ** 2)
            
            if torch.any(hit_in_path):
                hit_cnt += 1
                hit_mask.append(True)
                # hit_frames = torch.nonzero(hit_in_path, as_tuple=False).squeeze(1)
                # hit_distances = hit_in_path[hit_frames]
                # min_in_region = torch.argmin(hit_in_path)
                hit_index.append(torch.argmin(distance_to_goal))
            else:
                hit_mask.append(False)
                hit_index.append(-1)
                
        hit_rate = hit_cnt / sample_num
        hit_mask = torch.tensor(hit_mask)
        hit_index = torch.tensor(hit_index, dtype=torch.int32)

    elif molecule == "chignolin":
        raise NotImplementedError(f"THP for molecule {molecule} to be implemented")
    
    else:
        raise ValueError(f"THP for molecule {molecule} TBA")
    
    return hit_rate, hit_mask, hit_index


def compute_epd(
    cfg,
    trajectory_list,
    logger,
    hit_mask,
    hit_index
):
    atom_num = cfg.data.atom
    unit_scale_factor = 1000
    hit_trajectory = trajectory_list[hit_mask]
    hit_path_num = hit_mask.sum().item()
    # goal_state = goal_state[hit_mask]
    goal_state_info = torch.load(f"../simulation/data/{cfg.steeredmd.molecule}/{cfg.steeredmd.goal_state}.pt")
    goal_state_xyz = goal_state_info["xyz"].repeat(hit_path_num, 1, 1).to(trajectory_list.device)
    
    if hit_path_num != 0:
        hit_state_list = []
        rmsd = []
        for i in tqdm(
            range(hit_path_num),
            desc = f"Computing EPD, RMSD for {hit_path_num} hitting trajectories"
        ):
            hit_state_list.append(hit_trajectory[i, hit_index[i]])
            rmsd.append(kabsch_rmsd(hit_trajectory[i, hit_index[i]], goal_state_xyz[i]))
        
        hit_state_list = torch.stack(hit_state_list)
        matrix_f_norm = torch.sqrt(torch.square(
            pairwise_distance(hit_state_list, hit_state_list) - pairwise_distance(goal_state_xyz, goal_state_xyz)
        ).sum((1, 2)))
        
        epd_list = matrix_f_norm / (atom_num ** 2) * unit_scale_factor
        epd, epd_std = epd_list.mean(), epd_list.std()
        rmsd_list = torch.tensor(rmsd)
        rmsd, rmsd_std = rmsd_list.mean(), rmsd_list.std()
        
        if torch.isnan(epd):
            logger.info("EPD is NaN")
            epd = None
            epd_std = None
        if torch.isnan(rmsd):
            logger.info("RMSD is NaN")
            rmsd = None
            rmsd_std = None
        
        return epd, epd_std, rmsd, rmsd_std
    
    else:
        logger.info("No hitting trajectories found")
        return None, None, None, None


def compute_energy(
    cfg,
    trajectory_list,
    hit_mask
):  
    all_path_max_energy = None
    all_path_max_energy_std = None
    all_path_final_energy = None
    all_path_final_energy_std = None
    hitting_path_max_energy = None
    hitting_path_max_energy_std = None
    hitting_path_final_energy = None
    hitting_path_final_energy_std = None
    all_path_max_energy_idx = None
    hitting_path_max_energy_idx = None
    
    try:
        path_energy_list = np.zeros(shape=(trajectory_list.shape[0], trajectory_list.shape[1], 1))    
        for idx,trajectory in enumerate(tqdm(
            trajectory_list,
            desc=f"Computing energy for {trajectory_list.shape[0]} trajectories"
        )):
            path_energy_list[idx] = potential_energy(cfg, trajectory)
        
        all_path_max_energy = np.max(path_energy_list, axis=1).mean()
        all_path_max_energy_std = np.max(path_energy_list, axis=1).std()
        all_path_final_energy = path_energy_list[:, -1].mean()
        all_path_final_energy_std = path_energy_list[:, -1].std()
        all_path_max_energy_idx = np.argmax(path_energy_list, axis=1)
        
        if hit_mask.sum() != 0:
            hitting_path_energy_list = path_energy_list[hit_mask]
            hitting_path_max_energy = np.max(hitting_path_energy_list, axis=1).mean()
            hitting_path_max_energy_std = np.max(hitting_path_energy_list, axis=1).std()
            hitting_path_final_energy = hitting_path_energy_list[:, -1].mean()
            hitting_path_final_energy_std = hitting_path_energy_list[:, -1].std()
            hitting_path_max_energy_idx = np.argmax(hitting_path_energy_list, axis=1)

    except Exception as e:
        print(f"Error in computing energy: {e}")
    
    return all_path_max_energy, all_path_max_energy_std, all_path_final_energy, all_path_final_energy_std, \
        hitting_path_max_energy, hitting_path_max_energy_std, hitting_path_final_energy, hitting_path_final_energy_std, \
        all_path_max_energy_idx, hitting_path_max_energy_idx, path_energy_list


def init_simulation(
    cfg,
    pdb_file_path
):
    pdb = PDBFile(pdb_file_path)
    force_field = load_forcefield(cfg, cfg.steeredmd.molecule)
    system = load_system(cfg.steeredmd.molecule, pdb, force_field)
    
    cfg_simulation = cfg.steeredmd.simulation
    integrator = LangevinIntegrator(
        cfg_simulation.temperature * unit.kelvin,
        cfg_simulation.friction / unit.femtoseconds,
        cfg_simulation.timestep * unit.femtoseconds
    )
    platform = mm.Platform.getPlatformByName(cfg_simulation.platform)
    properties = {'DeviceIndex': '0', 'Precision': cfg_simulation.precision}

    simulation = Simulation(
        pdb.topology,
        system,
        integrator,
        platform,
        properties
    )        
    
    simulation.context.setPositions(pdb.positions)   
    simulation.minimizeEnergy()
    
    return simulation

def set_simulation(
    simulation,
    frame
):
    if frame == None:
        raise ValueError("Frame is None")
    
    atom_xyz = frame.detach().cpu().numpy()
    atom_list = [Vec3(atom[0], atom[1], atom[2]) for atom in atom_xyz]
    current_state_openmm = unit.Quantity(value=atom_list, unit=unit.nanometer)
    simulation.context.setPositions(current_state_openmm)
    simulation.context.setVelocities(unit.Quantity(value=np.zeros(frame.shape), unit=unit.nanometer/unit.picosecond))
    
    return simulation
