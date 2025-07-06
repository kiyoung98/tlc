import torch

from openmm import app
from openmm.app import *
from openmm.unit import *

from tqdm import tqdm
from torch.distributions import Normal

from .dynamics import Alanine, SteeredAlanine


STEERED_DYNAMICS_DICT = {
    "alanine": SteeredAlanine,
}
COMMITTOR_DYNAMICS_DICT = {
    "alanine": Alanine,
}

def simulate_steered_md(cfg, model, logger, repeat_idx, checkpoint_path):
    steered_md_simulation = SteeredMDSimulation(
        cfg = cfg,
        model = model,
    )
    time_horizon = cfg.steeredmd.simulation.time_horizon
    sample_num = cfg.steeredmd.sample_num
    goal_mlcv = steered_md_simulation.md_simulation_list[0].goal_mlcv
        
    # Shape: (sample_num, time_horizon, atom_num, 3)
    trajectory_list = torch.empty(size=(sample_num, time_horizon, cfg.data.atom, 3)).to(model.device)
    mlcv_list = torch.empty(size=(sample_num, time_horizon, goal_mlcv.shape[1])).to(model.device)
    current_position, current_mlcv = steered_md_simulation.report()
    trajectory_list[:, 0, : ] = current_position
    mlcv_list[:, 0, : ] = current_mlcv
    
    # simulate
    try:    
        for step in tqdm(
            range(1, time_horizon+1),
            desc = f"Genearting {sample_num} trajectories for {time_horizon} steps",
        ):
            steered_md_simulation.step(step)
            current_position, current_mlcv = steered_md_simulation.report()
            trajectory_list[:, step - 1, : ] = current_position
            mlcv_list[:, step - 1, : ] = current_mlcv
        
    except Exception as e:
        logger.error(f"Error in simulating steered MD: {e}")
        raise e

    torch.save(trajectory_list, f"{checkpoint_path}/SteeredMD/{repeat_idx}-traj.pt", )
    torch.save(mlcv_list, f"{checkpoint_path}/SteeredMD/{repeat_idx}-mlcv.pt")
    logger.info(f">> {trajectory_list.shape[0]} trajectories saved at: {checkpoint_path}/SteeredMD/{repeat_idx}-traj.pt")
    logger.info(f">> {mlcv_list.shape[0]} trajectories MLCV saved at: {checkpoint_path}/SteeredMD/{repeat_idx}-mlcv.pt")
    
    return trajectory_list, mlcv_list, goal_mlcv


class MDSimulation:
    def __init__(self, cfg, sample_num, device):
        self.device = device
        self.molecule = cfg.data.molecule
        self.start_state = cfg.md.start_state
        self.goal_state = cfg.md.goal_state
        self.sample_num = sample_num

        self._set_md(cfg)
        self.md_simulation_list = self._init_md_simulation_list(cfg)
        self.log_prob = Normal(0, self.std).log_prob
        
    def _load_dynamics(self, cfg):
        molecule = cfg.data.molecule
        dynamics = None
        
        if molecule == "alanine":
            dynamics = Alanine(cfg, self.start_state)
        else:
            raise ValueError(f"Molecule {molecule} not found")
        
        assert dynamics is not None, f"Failed to load dynamics for {molecule}"
        
        return dynamics
    
    def _set_md(self, cfg):
        # goal_state_md = getattr(dynamics, self.molecule)(cfg, self.end_state)
        goal_state_md = self._load_dynamics(cfg)
        self.num_particles = cfg.data.atom
        self.heavy_atoms = goal_state_md.heavy_atoms
        self.energy_function = goal_state_md.energy_function
        self.goal_position = torch.tensor(
            goal_state_md.position, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        self.m = torch.tensor(
            goal_state_md.m,
            dtype=torch.float,
            device=self.device,
        ).unsqueeze(-1)
        self.std = torch.tensor(
            goal_state_md.std,
            dtype=torch.float,
            device=self.device,
        )

    def _init_md_simulation_list(self, cfg):
        md_simulation_list = []
        for _ in tqdm(
            range(self.sample_num),
            desc="Initializing MD Simulation",
        ):
            # md = getattr(dynamics, self.molecule.title())(args, self.start_state)
            md_simulation_list.append(self._load_dynamics(cfg))

        self.start_position = torch.tensor(
            md_simulation_list[0].position, dtype=torch.float, device=self.device
        ).unsqueeze(0)
        
        return md_simulation_list

    def step(self, force):
        force = force.detach().cpu().numpy()
        for i in range(self.sample_num):
            self.md_simulation_list[i].step(force[i])

    def report(self):
        position_list, force_list = [], []
        for i in range(self.sample_num):
            position, force = self.md_simulation_list[i].report()
            position_list.append(position)
            force_list.append(force)

        position_list = torch.tensor(position_list, dtype=torch.float, device=self.device)
        force_list = torch.tensor(force_list, dtype=torch.float, device=self.device)
        return position_list, force_list

    def reset(self):
        for i in range(self.sample_num):
            self.md_simulation_list[i].reset()

    def set_position(self, positions):
        for i in range(self.sample_num):
            self.md_simulation_list[i].set_position(positions[i])

    def set_temperature(self, temperature):
        for i in range(self.sample_num):
            self.md_simulation_list[i].set_temperature(temperature)
            

class CommittorSimulation:
    def __init__(
        self,
        cfg,
    ):
        self.cfg = cfg
        self.device = cfg.device
        self.molecule = cfg.data.molecule
        
        self._init_md_simulation_list()
        
    def _init_md_simulation_list(self):
        self.md_simulation_list = []
        for _ in tqdm(
            range(self.sample_num),
            desc="Initializing MD Simulation",
        ):
            steered_dynamics = COMMITTOR_DYNAMICS_DICT[self.molecule](
                cfg = self.cfg,
            )
            self.md_simulation_list.append(steered_dynamics)
        
    def step(self, time):
        pass

    def report(self):
        pass
    

class SteeredMDSimulation:
    def __init__(
        self,
        cfg,
        model,
    ):
        self.cfg = cfg
        self.model = model
        self.device = model.device

        self.molecule = cfg.data.molecule
        self.start_state = cfg.steeredmd.start_state
        self.goal_state = cfg.steeredmd.goal_state
        self.sample_num = cfg.steeredmd.sample_num

        self.start_pdb = app.PDBFile(f"../simulation/data/{self.molecule}/{self.start_state}.pdb")
        self.goal_pdb = app.PDBFile(f"../simulation/data/{self.molecule}/{self.goal_state}.pdb")
        self.reference_frame = torch.load(f"../simulation/data/{self.molecule}/{self.start_state}.pt")['xyz'][0].to(self.device)

        self._init_md_simulation_list()
        self.position_list = torch.empty(size=(self.sample_num, cfg.data.atom, 3)).to(self.device)
        self.mlcv_list = torch.empty(size=(self.sample_num, self.md_simulation_list[0].goal_mlcv.shape[1])).to(self.device)

    def step(self, time):
        for i in range(self.sample_num):
            self.md_simulation_list[i].simulation.context.setParameter("time", time)
            self.md_simulation_list[i].step(time * self.md_simulation_list[i].timestep)

    def report(self):
        for i in range(self.sample_num):
            position, mlcv = self.md_simulation_list[i].report()
            self.position_list[i] = position
            self.mlcv_list[i] = mlcv
        
        return self.position_list, self.mlcv_list
    
    def _init_md_simulation_list(self):
        self.md_simulation_list = []
        for _ in tqdm(
            range(self.sample_num),
            desc="Initializing MD Simulation",
        ):
            steered_dynamics = STEERED_DYNAMICS_DICT[self.molecule](
                cfg = self.cfg,
                model = self.model,
                start_pdb = self.start_pdb,
                goal_pdb = self.goal_pdb,
                reference_frame = self.reference_frame,
            )
            self.md_simulation_list.append(steered_dynamics)

