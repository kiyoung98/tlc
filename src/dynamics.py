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
                parts = line.split()
                if len(parts) >= 4:
                    a = float(parts[1])
                    b = float(parts[2])  
                    c = float(parts[3]) 
                    return torch.tensor([a, b, c])
                break
    return torch.ones(3)

class SteeredMolecularDynamics:
    def __init__(self, cfg, state):
        self.pdb_file = f"./data/{cfg.molecule}/{state}.pdb"
        self.simulation, self.external_force, self.ca_indices = self._set_simulation(cfg, state)
        self.ca_position = self.report()[0]
        
    def _set_simulation(self, cfg, state):
        pdb = app.PDBFile(f"./data/{cfg.molecule}/{state}.pdb")
        forcefield = app.ForceField(*cfg.simulation.force_field)

        if cfg.molecule in ["chignolin", "trpcage"]:
            system = forcefield.createSystem(
                pdb.topology,
                nonbondedMethod=app.PME, 
                nonbondedCutoff=0.95 * unit.nanometers, 
                constraints=app.HBonds,
                ewaldErrorTolerance=0.0005,
            )
        elif cfg.molecule in ["chignolin_implicit", "trpcage_implicit"]:
            system = forcefield.createSystem(
                pdb.topology,
                constraints=app.HBonds,
                ewaldErrorTolerance=0.0005,
            )
        else:
            raise ValueError(f"State {cfg.molecule} not found")

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
        
        # Get alpha carbon indices for this topology
        atoms = list(pdb.topology.atoms())
        ca_indices = np.array([
            i for i in range(len(atoms))
            if atoms[i].name.strip() == 'CA'
        ])
        
        # Add external force only to alpha carbons
        for i in ca_indices:
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

        return simulation, external_force, ca_indices

    def step(self, external_forces):
        for force_idx, ca_idx in enumerate(self.ca_indices):
            self.external_force.setParticleParameters(force_idx, ca_idx, external_forces[force_idx])
        self.external_force.updateParametersInContext(self.simulation.context)
        self.simulation.step(1)

    def report(self):
        state = self.simulation.context.getState(getPositions=True, getEnergy=True)
        position = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        ca_position = position[self.ca_indices]  # Extract only alpha carbon positions
        total_potential = state.getPotentialEnergy().value_in_unit(unit.kilojoules / unit.mole)
        
        # Extract biased potential from CustomExternalForce using force group
        biased_state = self.simulation.context.getState(getEnergy=True, groups={1})
        biased_potential = biased_state.getPotentialEnergy().value_in_unit(unit.kilojoules / unit.mole)
        
        potential = total_potential - biased_potential
        return ca_position, potential, biased_potential, total_potential


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
        self.ca_indices = self.smds[0].ca_indices

        self.start_ca_position = torch.tensor(SteeredMolecularDynamics(cfg, cfg.start_state).ca_position, dtype=torch.float32, device=self.device).unsqueeze(0)
        self.goal_ca_position = torch.tensor(SteeredMolecularDynamics(cfg, cfg.goal_state).ca_position, dtype=torch.float32, device=self.device).unsqueeze(0)
    
        if self.name == "rmsd":
            self.start_cv = kabsch_rmsd(self.start_ca_position, self.goal_ca_position)
            self.goal_cv = 0
        else:
            self.start_cv = self.model(self.compute_descriptor(self.start_ca_position))
            self.goal_cv = self.model(self.compute_descriptor(self.goal_ca_position))

    def step(self, bias_ca_forces):
        bias_ca_forces = bias_ca_forces.cpu().numpy()
        for i in range(self.num_samples):
            self.smds[i].step(bias_ca_forces[i])

    def report(self):
        ca_positions = []
        potentials = []
        biased_potentials = []
        total_potentials = []
        for i in range(self.num_samples):
            ca_position, potential, biased_potential, total_potential = self.smds[i].report()  
            ca_positions.append(ca_position)
            potentials.append(potential)
            biased_potentials.append(biased_potential)
            total_potentials.append(total_potential)
        ca_positions = torch.tensor(np.array(ca_positions), dtype=torch.float32, device=self.device)
        potentials = torch.tensor(np.array(potentials), dtype=torch.float32, device=self.device)
        biased_potentials = torch.tensor(np.array(biased_potentials), dtype=torch.float32, device=self.device)
        total_potentials = torch.tensor(np.array(total_potentials), dtype=torch.float32, device=self.device)
        return ca_positions, potentials, biased_potentials, total_potentials

    def bias_ca_force(self, step, ca_positions):
        ca_positions.requires_grad = True
        
        if self.name == "rmsd":
            cv = kabsch_rmsd(ca_positions, self.goal_ca_position)
        elif self.name in METHODS:
            cv = self.model(self.compute_descriptor(ca_positions))
        else:
            raise ValueError(f"{self.name} not found")

        target_cv = self.start_cv + (self.goal_cv - self.start_cv) * (step / self.num_steps)
        bias_potential = 0.5 * self.k * torch.linalg.norm(target_cv - cv, ord=2)
        bias_ca_force = -torch.autograd.grad(bias_potential, ca_positions, retain_graph=False)[0]
        return bias_ca_force.detach(), cv.squeeze(-1).detach(), target_cv.squeeze(-1).detach()
    
    def compute_descriptor(self, ca_positions):
        if self.descriptor == "pairwise_distance":
            cell = extract_cell_from_pdb(self.pdb_file).to(device=ca_positions.device, dtype=ca_positions.dtype)
            ComputeDistances = PairwiseDistances(
                n_atoms=ca_positions.shape[1],
                PBC=True,
                cell=cell,
                scaled_coords=False
            )
            descriptors = ComputeDistances(ca_positions)
        else:
            raise ValueError(f"Representation {self.descriptor} not found")    
        
        return descriptors

    def _init_smds(self, cfg):
        self.smds = []
        for _ in tqdm(range(self.num_samples), desc="Initializing SMDs"):
            smd = SteeredMolecularDynamics(cfg, cfg.start_state)
            self.smds.append(smd)

    def metrics(self, ca_positions, potentials, threshold):
        metrics = {}
        rmsds = 10 * kabsch_rmsd(ca_positions, self.goal_ca_position) # nm to angstrom

        hit = rmsds < threshold
        thp = 100 * hit.sum().item() / len(hit)
        
        max_potentials = potentials.max(dim=1)[0]
        etss = max_potentials[hit]

        metrics["rmsd"] = rmsds.mean().item()
        metrics["rmsd_std"] = rmsds.std().item()
        metrics["thp"] = thp
        metrics["ets"] = etss.mean().item()
        metrics["ets_std"] = etss.std().item()

        return metrics, rmsds, etss

    def plot(self, ca_positions, potentials, biased_potentials, total_potentials, cvs, target_cvs):
        plots = {}
        tica_plot = self.plot_paths(ca_positions)
        potentials_plot = self.plot_potentials(potentials)
        biased_potentials_plot = self.plot_potentials(biased_potentials)
        total_potentials_plot = self.plot_potentials(total_potentials)
        cvs_plot = self.plot_cvs(cvs, target_cvs)
        plots["paths"] = tica_plot
        plots["potentials"] = potentials_plot
        plots["biased_potentials"] = biased_potentials_plot
        plots["total_potentials"] = total_potentials_plot
        plots["cvs"] = cvs_plot
        return plots

    # Plots
    def plot_paths(self, ca_positions):
        tica_cvs = np.load(f"./data/{self.molecule}/tica_cvs.npy")

        tica_wrapper = TICA_WRAPPER(
            tica_model_path=f"./data/{self.molecule}/tica_model.pkl",
            pdb_path=f"./data/{self.molecule}/folded.pdb",
            tica_switch=True if self.molecule == "chignolin" else False
        )

        tic1 = tica_cvs[:, 0]
        tic2 = tica_cvs[:, 1]

        ca_positions_reshaped = ca_positions.detach().cpu().numpy().reshape(-1, *ca_positions.shape[2:])
        path_tica_cvs = tica_wrapper.transform(tica_wrapper.pos2cad(ca_positions_reshaped)).reshape(*ca_positions.shape[:2], 2)
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
        start_ca_pos_reshaped = self.start_ca_position.detach().cpu().numpy().reshape(-1, *self.start_ca_position.shape[1:])
        goal_ca_pos_reshaped = self.goal_ca_position.detach().cpu().numpy().reshape(-1, *self.goal_ca_position.shape[1:])
        
        start_tica_cv = tica_wrapper.transform(tica_wrapper.pos2cad(start_ca_pos_reshaped)).reshape(*self.start_ca_position.shape[:1], 2)
        goal_tica_cv = tica_wrapper.transform(tica_wrapper.pos2cad(goal_ca_pos_reshaped)).reshape(*self.goal_ca_position.shape[:1], 2)
        
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
        potentials = potentials.detach().cpu().numpy()
        fig = plt.figure(figsize=(7, 7))

        colors = plt.cm.plasma(np.linspace(0, 1, potentials.shape[0]))
        for i in range(potentials.shape[0]):
            plt.plot(potentials[i], color=colors[i], label=f'Path {i+1}')
        plt.legend(fontsize=FONTSIZE_SMALL)
        plt.xlabel("Step", fontsize=FONTSIZE_SMALL)
        plt.ylabel("Potential", fontsize=FONTSIZE_SMALL)
        return fig

    def plot_cvs(self, cvs, target_cvs):
        cvs = cvs.detach().cpu().numpy()
        target_cvs = target_cvs.detach().cpu().numpy()
        fig = plt.figure(figsize=(7, 7))
        colors = plt.cm.plasma(np.linspace(0, 1, cvs.shape[0]))
        for i in range(cvs.shape[0]):
            plt.plot(cvs[i], color=colors[i], label=f'CV Path {i+1}')
        plt.plot(target_cvs[0], color='black', linestyle='--', label='Target CV')
        plt.legend(fontsize=FONTSIZE_SMALL)
        plt.xlabel("Step", fontsize=FONTSIZE_SMALL)
        plt.ylabel("CV", fontsize=FONTSIZE_SMALL)
        return fig