import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import openmm as mm
import openmm.unit as unit
from openmm import app
from openmmtools.integrators import VVVRIntegrator

from mlcolvar.core.transform.descriptors import PairwiseDistances
from plot import TICA_WRAPPER
from utils import kabsch_rmsd
from constant import METHODS, FONTSIZE_SMALL


def extract_cell_from_pdb(pdb_file):
    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('CRYST1'):
                # CRYST1 레코드 형식: CRYST1 a b c alpha beta gamma spacegroup z
                parts = line.split()
                if len(parts) >= 4:
                    a = float(parts[1])  # a 축 길이
                    b = float(parts[2])  # b 축 길이  
                    c = float(parts[3])  # c 축 길이
                    return torch.tensor([a, b, c])
                break
    
    # CRYST1 레코드가 없으면 기본값 반환
    return torch.ones(3)

class SteeredMolecularDynamics:
    def __init__(self, cfg, state):
        self.pdb_file = f"./data/{cfg.molecule}/{state}.pdb"
        self.simulation, self.external_force = self._set_simulation(cfg, state)
        self.position = self.report()[0]
        self.num_atoms = self.position.shape[0]
        self.alpha_carbon_indices = self.get_alpha_carbon_indices()
        self.heavy_atom_indices = self.get_heavy_atom_indices()
        
        print(self.alpha_carbon_indices)
        print(self.heavy_atom_indices)
        
    def _set_simulation(self, cfg, state):
        forcefield = app.ForceField(*cfg.simulation.force_field)
        pdb = app.PDBFile(f"./data/{cfg.molecule}/{state}.pdb")

        system = forcefield.createSystem(
            pdb.topology,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005,
        )

        # integrator
        integrator = VVVRIntegrator(
            cfg.simulation.temperature * unit.kelvin,
            cfg.simulation.friction / unit.femtoseconds,
            cfg.simulation.timestep * unit.femtoseconds
        )
        integrator.setConstraintTolerance(0.00001)

        # Add external force to system before creating simulation
        external_force = mm.CustomExternalForce("-(fx*x + fy*y + fz*z)")
        external_force.addPerParticleParameter("fx")
        external_force.addPerParticleParameter("fy")
        external_force.addPerParticleParameter("fz")
        for i in range(system.getNumParticles()):
            external_force.addParticle(i, [0, 0, 0])
        # Set a unique force group for this external force
        external_force.setForceGroup(1)
        system.addForce(external_force)

        platform = mm.Platform.getPlatformByName(cfg.simulation.platform)
        if cfg.simulation.platform in ["CUDA", "OpenCL"]:
            properties = {'DeviceIndex': '0', 'Precision': cfg.simulation.precision}
        elif cfg.simulation.platform == "CPU":
            properties = {}
        else:
            raise ValueError(f"Platform {cfg.simulation.platform} not found")
        
        simulation = app.Simulation(
            pdb.topology,
            system,
            integrator,
            platform,
            properties
        )
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()

        return simulation, external_force

    def step(self, external_forces):
        for i in range(external_forces.shape[0]):
            self.external_force.setParticleParameters(i, i, external_forces[i])
        self.external_force.updateParametersInContext(self.simulation.context)
        self.simulation.step(1)

    def report(self):
        state = self.simulation.context.getState(getPositions=True, getEnergy=True)
        position = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        total_potential = state.getPotentialEnergy().value_in_unit(unit.kilojoules / unit.mole)
        
        # Extract biased potential from CustomExternalForce using force group
        biased_state = self.simulation.context.getState(getEnergy=True, groups={1})
        biased_potential = biased_state.getPotentialEnergy().value_in_unit(unit.kilojoules / unit.mole)
        
        potential = total_potential - biased_potential
        return position, potential

    def get_alpha_carbon_indices(self): # TODO: mdtraj library
        atoms = list(self.simulation.topology.atoms())
        alpha_carbon_indices = np.array([
            i for i in range(len(atoms))
            if atoms[i].name.strip() == 'CA'
        ])
        return alpha_carbon_indices

    def get_heavy_atom_indices(self): # TODO: mdtraj library
        atoms = list(self.simulation.topology.atoms())
        heavy_atom_indices = np.array([
            i for i in range(len(atoms))
            if atoms[i].element.atomic_number != 1  # Exclude hydrogen atoms
        ])
        return heavy_atom_indices


class SMDs:
    def __init__(self, cfg, model):
        self.num_samples = cfg.num_samples
        self.molecule = cfg.molecule

        self.name = cfg.model.name
        if self.name == "rmsd":
            self.device = torch.device("cuda")
        else:
            self.model = model
            self.device = model.device
        self.descriptor = cfg.model.descriptor

        self.k = cfg.simulation.k
        self.num_steps = cfg.simulation.num_steps

        self._init_smds(cfg)

        self.pdb_file = self.smds[0].pdb_file
        self.num_atoms = self.smds[0].num_atoms
        self.alpha_carbon_indices = self.smds[0].alpha_carbon_indices
        self.heavy_atom_indices = self.smds[0].heavy_atom_indices

        self.start_position = torch.tensor(SteeredMolecularDynamics(cfg, cfg.start_state).position, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.goal_position = torch.tensor(SteeredMolecularDynamics(cfg, cfg.goal_state).position, dtype=torch.float32, device=self.device).unsqueeze(0)
    
        if self.name == "rmsd":
            self.start_cv = kabsch_rmsd(self.compute_descriptor(self.start_position), self.compute_descriptor(self.goal_position))
            self.goal_cv = 0
        else:
            self.start_cv = self.model(self.compute_descriptor(self.start_position))
            self.goal_cv = self.model(self.compute_descriptor(self.goal_position))

    def step(self, step, positions):
        bias_forces = self.bias_force(step, positions).cpu().numpy()
        for i in range(self.num_samples):
            self.smds[i].step(bias_forces[i])

    def report(self):
        positions = []
        potentials = []
        for i in range(self.num_samples):
            position, potential = self.smds[i].report()  
            positions.append(position)
            potentials.append(potential)
        positions = torch.tensor(np.array(positions), dtype=torch.float32, device=self.device)
        potentials = torch.tensor(np.array(potentials), dtype=torch.float32, device=self.device)
        return positions, potentials

    def bias_force(self, step, positions):
        positions.requires_grad = True
        
        if self.name == "rmsd":
            cv = kabsch_rmsd(self.compute_descriptor(positions), self.compute_descriptor(self.goal_position))
        elif self.name in METHODS:
            cv = self.model(self.compute_descriptor(positions))
        else:
            raise ValueError(f"{self.name} not found")

        target_cv = self.start_cv + (self.goal_cv - self.start_cv) * (step / self.num_steps)
        bias_potential = 0.5 * self.k * torch.linalg.norm(target_cv - cv, ord=2)
        bias_force = -torch.autograd.grad(bias_potential, positions, retain_graph=False)[0]
        return bias_force
    
    def compute_descriptor(self, positions):
        if self.descriptor == "alpha_carbon_distance":
            cell = extract_cell_from_pdb(self.pdb_file).to(device=positions.device, dtype=positions.dtype)
            alpha_carbon_positions = positions[:, self.alpha_carbon_indices]
            ComputeDistances = PairwiseDistances(
                n_atoms=alpha_carbon_positions.shape[1],
                PBC=True,
                cell=cell,
                scaled_coords=False
            )
            descriptors = ComputeDistances(alpha_carbon_positions)
        elif self.descriptor == "alpha_carbon":
            descriptors = positions[:, self.alpha_carbon_indices]
        elif self.descriptor == "heavy_atom":
            descriptors = positions[:, self.heavy_atom_indices]
        else:
            raise ValueError(f"Representation {self.descriptor} not found")    
        
        return descriptors

    def _init_smds(self, cfg):
        self.smds = []
        for _ in tqdm(range(self.num_samples), desc="Initializing SMDs"):
            smd = SteeredMolecularDynamics(cfg, cfg.start_state)
            self.smds.append(smd)

    def metrics(self, positions, potentials, threshold):
        metrics = {}
        rmsd = 10 * kabsch_rmsd(positions[:, self.alpha_carbon_indices], self.goal_position[:, self.alpha_carbon_indices]) # nm to angstrom
        metrics["rmsd"] = rmsd.mean().item()
        metrics["rmsd_std"] = rmsd.std().item()

        hit = rmsd < threshold
        thp = 100 * hit.sum().item() / len(hit)
        metrics["thp"] = thp

        etss = []
        for i, hit_idx in enumerate(hit):
            if hit_idx:
                ets = potentials[i].max(0)[0]
                etss.append(ets)

        if thp > 0:
            etss = torch.tensor(etss)
            metrics["ets"] = etss.mean().item()
            metrics["ets_std"] = etss.std().item()

        return metrics

    def plot(self, positions, potentials):
        plots = {}
        tica_plot = self.plot_paths(positions)
        potentials_plot = self.plot_potentials(potentials)
        plots["paths"] = tica_plot
        plots["potentials"] = potentials_plot
        return plots

    # Plots
    def plot_paths(self, positions):
        tica_cvs = np.load(f"./data/{self.molecule}/tica_cvs.npy")

        tica_wrapper = TICA_WRAPPER(
            tica_model_path=f"./data/{self.molecule}/tica_model.pkl",
            pdb_path=f"./data/{self.molecule}/folded.pdb",
            tica_switch=True if self.molecule == "chignolin" else False
        )

        tic1 = tica_cvs[:, 0]
        tic2 = tica_cvs[:, 1]

        positions_reshaped = positions.cpu().numpy().reshape(-1, *positions.shape[2:])
        path_tica_cvs = tica_wrapper.transform(tica_wrapper.pos2cad(positions_reshaped)).reshape(*positions.shape[:2], 2)
        path_tic1 = path_tica_cvs[:, :, 0]
        path_tic2 = path_tica_cvs[:, :, 1]

        colors = plt.cm.plasma(np.linspace(0, 1, path_tica_cvs.shape[0]))

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        ax.hexbin(
            tic1, tic2, 
            mincnt=1,
            gridsize=200,
            cmap='viridis',
            zorder=2,
            rasterized=True,
        )

        for i in range(path_tica_cvs.shape[0]):
            ax.scatter(path_tic1[i], path_tic2[i], color=colors[i], s=5, zorder=3, rasterized=True)

        # Transform start and goal positions through TICA and add as stars
        start_pos_reshaped = self.start_position.cpu().numpy().reshape(-1, *self.start_position.shape[1:])
        goal_pos_reshaped = self.goal_position.cpu().numpy().reshape(-1, *self.goal_position.shape[1:])
        
        start_tica_cv = tica_wrapper.transform(tica_wrapper.pos2cad(start_pos_reshaped)).reshape(*self.start_position.shape[:1], 2)
        goal_tica_cv = tica_wrapper.transform(tica_wrapper.pos2cad(goal_pos_reshaped)).reshape(*self.goal_position.shape[:1], 2)
        
        start_tic1, start_tic2 = start_tica_cv[:, 0], start_tica_cv[:, 1]
        goal_tic1, goal_tic2 = goal_tica_cv[:, 0], goal_tica_cv[:, 1]
        
        # Add start position as green star
        ax.scatter(start_tic1, start_tic2, marker='*', color='green', s=200, zorder=4, 
                  edgecolors='black', linewidth=1, label='Start')
        # Add goal position as red star  
        ax.scatter(goal_tic1, goal_tic2, marker='*', color='red', s=200, zorder=4,
                  edgecolors='black', linewidth=1, label='Goal')
        
        ax.legend(fontsize=FONTSIZE_SMALL)
        ax.set_xlabel("TIC 1", fontsize=FONTSIZE_SMALL)
        ax.set_ylabel("TIC 2", fontsize=FONTSIZE_SMALL)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        return fig


    def plot_potentials(self, potentials):
        potentials = potentials.cpu().numpy()
        fig = plt.figure(figsize=(7, 7))

        colors = plt.cm.plasma(np.linspace(0, 1, potentials.shape[0]))
        for i in range(potentials.shape[0]):
            plt.plot(potentials[i], color=colors[i])
        return fig