import torch

import openmm as mm
import openmm.unit as unit
from openmm import app
from openmmtools.integrators import VVVRIntegrator


from ...util.constant import *
from ...util.rotate import kabsch_rmsd, kabsch
from ...util.angle import compute_dihedral
from ...util.convert import input2representation
from .dynamics import BaseDynamics, load_forcefield, load_system


class Alanine(BaseDynamics):
    def __init__(self, cfg, state):
        super().__init__(cfg, state)

    def setup(self):
        forcefield = app.ForceField("amber99sbildn.xml")
        pdb = app.PDBFile(self.start_file)
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            ewaldErrorTolerance=0.0005,
        )
        # Set integrator
        integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
        )
        integrator.setConstraintTolerance(0.00001)
        platform = mm.Platform.getPlatformByName("OpenCL")
        properties = {'DeviceIndex': '0', 'Precision': self.cfg.steeredmd.simulation.precision}
        self.simulation = app.Simulation(
            pdb.topology,
            system,
            integrator,
            platform,
            properties
        )
        self.simulation.context.setPositions(pdb.positions)

        return pdb, integrator, self.simulation



class SteeredAlanine:
    def __init__(
        self,
        cfg,
        model,
        start_pdb,
        goal_pdb,
        reference_frame
    ):
        self.cfg = cfg
        self.model = model
        self.device = model.device
        self.reference_frame = reference_frame

        # Simulation parameters
        self.force_type = cfg.model.name
        self.atom_num = cfg.data.atom  
        self.k = cfg.steeredmd.simulation.k
        self.molecule = cfg.steeredmd.molecule
        self.time_horizon = cfg.steeredmd.simulation.time_horizon
        self.temperature = cfg.steeredmd.simulation.temperature * unit.kelvin
        self.friction = cfg.steeredmd.simulation.friction / unit.femtoseconds
        self.timestep = cfg.steeredmd.simulation.timestep * unit.femtoseconds
        self.platform = self.cfg.steeredmd.simulation.platform

        # Load simulation components
        self.forcefield = load_forcefield(cfg, self.molecule)
        self.system = load_system(self.molecule, start_pdb, self.forcefield)
        self._set_start_position(start_pdb, self.system)
        self._set_goal_position(goal_pdb, self.system)
        self._set_custom_force()

        # Set simulation
        integrator = self._new_integrator()
        platform = mm.Platform.getPlatformByName(self.platform)
        if self.platform in ["CUDA", "OpenCL"]:
            properties = {'DeviceIndex': '0', 'Precision': self.cfg.steeredmd.simulation.precision}
        elif self.platform == "CPU":
            properties = {}
        else:
            raise ValueError(f"Platform {self.platform} not found")
        self.simulation = app.Simulation(
            start_pdb.topology,
            self.system,
            integrator,
            platform,
            properties
        )
        self.simulation.context.setPositions(start_pdb.positions)
        self.simulation.minimizeEnergy()

    def step(self, time):
        self.simulation.context.setParameter("time", time)
        current_position = self.current_position.reshape(-1, self.atom_num, 3)
        
        # Set external force
        # torch.autograd.set_detect_anomaly(True)
        if self.force_type in MLCOLVAR_METHODS:
            current_position.requires_grad = True
            self.current_mlcv = self.model(input2representation(self.cfg, current_position, self.reference_frame)).reshape(-1)
            current_target_mlcv = self.start_mlcv + (self.goal_mlcv - self.start_mlcv) * (time / self.time_horizon).value_in_unit(unit.femtosecond)
            mlcv_difference = 0.5 * self.k * torch.linalg.norm(current_target_mlcv - self.current_mlcv, ord=2)
            
            bias_force = torch.autograd.grad(mlcv_difference, current_position)[0].reshape(self.atom_num, 3)
            for i in range(self.atom_num):
                self.external_force.setParticleParameters(i, i, bias_force[i])
            self.external_force.updateParametersInContext(self.simulation.context)

        elif self.force_type == "dihedral":
            current_phi = compute_dihedral(current_position[:, ALDP_PHI_ANGLE])
            current_psi = compute_dihedral(current_position[:, ALDP_PSI_ANGLE])
            self.current_mlcv = torch.cat([current_psi, current_phi], dim=-1).unsqueeze(0)
            self.simulation.context.setParameter("time", time)

        elif self.force_type == "rmsd":
            pass

        else:
            raise ValueError(f"Force type {self.force_type} not found")

        self.simulation.step(1)

    def report(self):
        self.current_position = torch.tensor(
            [list(p) for p in self.simulation.context.getState(getPositions=True).getPositions().value_in_unit(unit.nanometer)],
            dtype=torch.float32, device = self.device
        ).reshape(self.atom_num, 3)
        
        return self.current_position, self.current_mlcv

    def reset(self):
        self.simulation.context.setPositions(self.position)
        self.simulation.context.setVelocitiesToTemperature(self.temperature)

    def _new_integrator(self):
        integrator = VVVRIntegrator(
            self.temperature,
            self.friction,
            self.timestep,
        )
        integrator.setConstraintTolerance(0.00001)
        integrator.setRandomNumberSeed(0)

        return integrator

    def _set_start_position(self, pdb, system):
        # Set start position
        integrator = self._new_integrator()
        platform = mm.Platform.getPlatformByName(self.platform)
        if self.platform in ["CUDA", "OpenCL"]:
            properties = {'DeviceIndex': '0', 'Precision': self.cfg.steeredmd.simulation.precision}
        elif self.platform == "CPU":
            properties = {}
        else:
            raise ValueError(f"Platform {self.platform} not found")
        simulation = app.Simulation(
            pdb.topology,
            system,
            integrator,
            platform,
            properties
        )
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        self.start_position = torch.tensor(
            [list(p) for p in simulation.context.getState(getPositions=True).getPositions().value_in_unit(unit.nanometer)],
            dtype=torch.float32, device = self.device
        ).reshape(self.atom_num, 3)
        
        # Set start mlcv
        if self.force_type == "dihedral":
            self.start_psi = compute_dihedral(self.start_position.unsqueeze(0)[:, ALDP_PSI_ANGLE])
            self.start_phi = compute_dihedral(self.start_position.unsqueeze(0)[:, ALDP_PHI_ANGLE])
            self.start_mlcv = torch.cat([self.start_psi, self.start_phi], dim=-1).unsqueeze(0)
        
        elif self.force_type == "rmsd":
            pass
        
        elif self.force_type in MLCOLVAR_METHODS:
            start_position = self.start_position
            start_position.requires_grad = True
            self.start_mlcv = self.model(input2representation(
                self.cfg,
                start_position.unsqueeze(0),
                self.reference_frame
            ))
            
        else:
            raise ValueError(f"Force type {self.force_type} not supported")

        self.current_position = self.start_position
        self.current_mlcv = self.start_mlcv

    def _set_goal_position(self, pdb, system):
        # Set goal position
        integrator = self._new_integrator()
        platform = mm.Platform.getPlatformByName(self.platform)
        if self.platform in ["CUDA", "OpenCL"]:
            properties = {'DeviceIndex': '0', 'Precision': self.cfg.steeredmd.simulation.precision}
        elif self.platform == "CPU":
            properties = {}
        else:
            raise ValueError(f"Platform {self.platform} not found")
        simulation = app.Simulation(
            pdb.topology,
            system,
            integrator,
            platform,
            properties
        )
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        self.goal_position = torch.tensor(
            [list(p) for p in simulation.context.getState(getPositions=True).getPositions().value_in_unit(unit.nanometer)],
            dtype=torch.float32, device = self.device
        ).reshape(self.atom_num, 3)

        # Set goal mlcv
        if self.force_type == "dihedral":
            self.goal_psi = compute_dihedral(self.goal_position.unsqueeze(0)[:, ALDP_PSI_ANGLE])
            self.goal_phi = compute_dihedral(self.goal_position.unsqueeze(0)[:, ALDP_PHI_ANGLE])
            self.goal_mlcv = torch.cat([self.goal_psi, self.goal_phi], dim=-1).unsqueeze(0)
        
        elif self.force_type == "rmsd":
            pass
        
        elif self.force_type in MLCOLVAR_METHODS:
            self.goal_mlcv = self.model(input2representation(
                self.cfg,
                self.goal_position.unsqueeze(0),
                self.reference_frame
            ))
            
        else:
            raise ValueError(f"Force type {self.force_type} not supported")

    def _set_custom_force(self):        
        if self.force_type == "dihedral":
            custom_cv_force = mm.CustomTorsionForce(
                "0.5 * k * (theta - (theta_start + (theta_goal - theta_start) * (time / total_time)))^2"
            )
            custom_cv_force.addTorsion(*ALDP_PSI_ANGLE, [self.start_psi, self.goal_psi])
            custom_cv_force.addTorsion(*ALDP_PHI_ANGLE, [self.start_phi, self.goal_phi])
            custom_cv_force.addPerTorsionParameter("theta_start")
            custom_cv_force.addPerTorsionParameter("theta_goal")
            custom_cv_force.addGlobalParameter("k", self.k)
            custom_cv_force.addGlobalParameter("time", 0)
            custom_cv_force.addGlobalParameter("total_time", self.time_horizon * self.timestep)
            self.system.addForce(custom_cv_force)

        elif self.force_type == "rmsd":
            custom_cv_force = mm.CustomCVForce(
                "0.5 * k * ( rmsd - start_rmsd * (1 - time / total_time) )^2"
            )
            custom_cv_force.addCollectiveVariable("rmsd", mm.RMSDForce(self.goal_position.cpu().numpy()))
            custom_cv_force.addGlobalParameter("start_rmsd", kabsch_rmsd(self.start_position, self.goal_position).squeeze().item())
            custom_cv_force.addGlobalParameter("k", self.k)
            custom_cv_force.addGlobalParameter("time", 0)
            custom_cv_force.addGlobalParameter("total_time", self.time_horizon * self.timestep)
            self.system.addForce(custom_cv_force)

        elif self.force_type in MLCOLVAR_METHODS:
            external_force = mm.CustomExternalForce("(fx*x + fy*y + fz*z)")
            # custom_cv_force.addGlobalParameter("k", self.k)
            external_force.addGlobalParameter("time", 0)
            external_force.addGlobalParameter("total_time", self.time_horizon * self.timestep)
            external_force.addPerParticleParameter("fx")
            external_force.addPerParticleParameter("fy")
            external_force.addPerParticleParameter("fz")
            for i in range(self.atom_num):
                external_force.addParticle(i, [0, 0, 0])
            self.system.addForce(external_force)
            self.external_force = external_force           

        else:
            raise ValueError(f"Force type {self.force_type} not found")

        return

