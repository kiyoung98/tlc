import torch
import random 

import numpy as np

from openmm import *
from openmm.app import *
from openmm.unit import *

from tqdm import tqdm
from torch.utils.data import Dataset

class MD_Dataset(Dataset):
    def __init__(
        self,
        loaded_traj,
        config,
        args,
        sanity_check=False
    ):
        super(MD_Dataset, self).__init__()
        
        self.molecule = config['molecule']
        self.state = config['state']
        self.temperature = config['temperature']
        self.time = config['time']
        self.force_field = config['force_field']
        self.solvent = config['solvent']
        self.platform = config['platform']
        self.precision = config['precision']
        self.args = args
        
        data_x_list = []
        data_y_list = []
        data_interval_list = []
        data_goal_list = []
        
        random_indices = np.random.choice(self.time - 2, args.sim_length, replace=True)
        
        if args.dataset_type == "random":
            for t in tqdm(
                random_indices,
                desc="Loading data by random idx"
            ):
                current_state = torch.tensor(loaded_traj[t].xyz.squeeze())
                next_state = torch.tensor(loaded_traj[t+1].xyz.squeeze())
                for i in range(args.sim_repeat_num):
                    random_interval = random.sample(range(1, np.min([self.time - t, args.max_path_length])), 1)[0]
                    goal_state = torch.tensor(loaded_traj[t+random_interval].xyz.squeeze())
                    
                    data_x_list.append(current_state)
                    data_y_list.append(next_state)
                    data_goal_list.append(goal_state)
                    data_interval_list.append(torch.tensor(random_interval).unsqueeze(0))
        elif args.dataset_type == "goal":
            for t in tqdm(
                random_indices,
                desc="Loading data by random idx"
            ):
                random_sim = random.sample(range(sim_num), 1)[0]
                current_state = torch.tensor(loaded_traj[t+1].xyz.squeeze())
                next_state = torch.tensor(loaded_traj[t].xyz.squeeze())
                goal_state = torch.tensor(loaded_traj[0].xyz.squeeze())
                
                data_x_list.append(current_state)
                data_y_list.append(next_state)
                data_goal_list.append(goal_state)
                data_interval_list.append(torch.tensor(t+1).unsqueeze(0))
        elif args.dataset_type == "multi-next":
            self.set_simulation()
            for t in tqdm(
                random_indices,
                desc="Multi goal dataset construction"
            ):
                current_state = torch.tensor(loaded_traj[t].xyz.squeeze())
                
                # Short simulation, get next_state and goal_state
                for i in range(args.sim_repeat_num):
                    random_interval = random.sample(range(1, np.min([self.time - t, args.max_path_length])), 1)[0]
                    next_state, goal_state = self.short_simulation(current_state, random_interval)
                    data_x_list.append(current_state)
                    data_y_list.append(next_state)
                    data_goal_list.append(goal_state)
                    data_interval_list.append(torch.tensor(random_interval).unsqueeze(0))
            self.simulation = None
        elif args.dataset_type == "multi-next-ic":
            data_x_ic_list = []
            data_y_ic_list = []
            data_goal_ic_list = []
            self.set_simulation()
            for t in tqdm(
                random_indices,
                desc="Multi goal dataset construction with ic"
            ):
                current_state = torch.tensor(loaded_traj[t].xyz.squeeze())
                xyz2ic = load_internal_coordinate_transform()
                current_state_ic = xyz2ic(current_state.unsqueeze(0))
                
                # Short simulation, get next_state and goal_state
                for i in range(args.sim_repeat_num):
                    random_interval = random.randint(2, 100)
                    next_state, goal_state = self.short_simulation(current_state, random_interval)
                    next_state_ic = xyz2ic(next_state.unsqueeze(0))
                    goal_state_ic = xyz2ic(goal_state.unsqueeze(0))
                    data_x_list.append(current_state)
                    data_y_list.append(next_state)
                    data_goal_list.append(goal_state)
                    data_interval_list.append(torch.tensor(random_interval).unsqueeze(0))
                    data_x_ic_list.append(current_state_ic)
                    data_y_ic_list.append(next_state_ic)
                    data_goal_ic_list.append(goal_state_ic)
            self.simulation = None
        elif args.dataset_type == "two-step":
            for t in tqdm(
                random_indices,
                desc="Two step dataset construction"
            ):
                current_state = torch.tensor(loaded_traj[t].xyz.squeeze())
                next_state = torch.tensor(loaded_traj[t+1].xyz.squeeze())
                goal_state = torch.tensor(loaded_traj[t+2].xyz.squeeze())

                data_x_list.append(current_state)
                data_y_list.append(next_state)
                data_goal_list.append(goal_state)
                data_interval_list.append(torch.tensor(2).unsqueeze(0))
        else:
            raise ValueError(f"Index {args.dataset_type} not found")
                
        self.x = torch.stack(data_x_list)
        self.y = torch.stack(data_y_list)
        self.goal = torch.stack(data_goal_list)
        self.delta_time = torch.stack(data_interval_list)
        if args.dataset_type == "multi-next-ic":
            self.x_ic = torch.stack(data_x_ic_list)
            self.y_ic = torch.stack(data_y_ic_list)
            self.goal_ic = torch.stack(data_goal_ic_list)
        
        
    def sanity_check(self, loaded_traj):
        assert torch.equal(self.x.shape, self.y.shape), f"Shape of x and y not equal"
        
        for t in tqdm(
            range(self.time -1),
            desc="Sanity check"
        ):
            x = self.x[t]
            y = self.y[t]
            x_frame = torch.tensor(loaded_traj[t].xyz.squeeze())
            y_frame = torch.tensor(loaded_traj[t+1].xyz.squeeze())
            
            assert torch.equal(x, x_frame), f"Frame {t}, x not equal"
            assert torch.equal(y, y_frame), f"Frame {t+1}, y not equal"
        
    def __getitem__(self, index):
	    return self.x[index], self.y[index], self.goal[index], self.delta_time[index], \
            self.x_ic[index], self.y_ic[index], self.goal_ic[index]
 
    def __len__(self):
	    return self.x.shape[0]
    
    def set_simulation(self):
        pdb_file = PDBFile(f"./data/{self.molecule}-stable/{self.state}.pdb")
        force_field = ForceField("amber99sbildn.xml", "tip3p.xml")
        system = force_field.createSystem(
            pdb_file.topology,
            nonbondedCutoff=3 * nanometer,
            constraints=HBonds
        )
        integrator = LangevinIntegrator(
            self.temperature * kelvin,
            1 / picosecond,
            1 * femtoseconds
        )
        platform = Platform.getPlatformByName("OpenCL")
        properties = {'Precision': "mixed"}
        simulation = Simulation(
            pdb_file.topology,
            system,
            integrator,
            platform,
            properties
        )
        self.simulation = simulation
        
    def short_simulation(self, current_state, step=100):
        atom_xyz = current_state.detach().cpu().numpy()
        atom_list = [Vec3(atom[0], atom[1], atom[2]) for atom in atom_xyz]
        current_state_openmm = Quantity(value=atom_list, unit=nanometer)
        self.simulation.context.setPositions(current_state_openmm)
        self.simulation.context.setVelocities(Quantity(value=np.zeros((len(atom_list), 3)), unit=nanometer/picosecond))
        self.simulation.minimizeEnergy()
        
        self.simulation.step(1)
        next_state = torch.tensor(self.simulation.context.getState(getPositions=True).getPositions().value_in_unit(nanometer))
        self.simulation.step(step-1)
        goal_state = torch.tensor(self.simulation.context.getState(getPositions=True).getPositions().value_in_unit(nanometer))
        
        return next_state, goal_state