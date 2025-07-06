import os
import csv
import pytz
import json
import argparse

from sys import stdout
from datetime import datetime

from openmm import *
from openmm.app import *
from openmm.unit import *


class ForceReporter(object):
    def __init__(self, file_name, reportInterval, append=False):
        self._reportInterval = reportInterval
        self._openedFile = isinstance(file_name, str)
        if self._openedFile:
            self._out = open(file_name, 'a' if append else 'w')
            self.writer  = csv.writer(self._out)
            header = []
            for atom in range(1, 23):
                header.extend([f'atom_{atom}_force_x', f'atom_{atom}_force_y', f'atom_{atom}_force_z'])
            self.writer.writerow(header)

    # def __del__(self):
        # self.file.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, True, False, None)

    def report(self, simulation, state):
        forces = state.getForces().value_in_unit(kilojoules/mole/nanometer)
        row = []
        for f in forces:
            # self.file.write('%g %g %g\n' % (f[0], f[1], f[2]))
            row.extend([f[0], f[1], f[2]])
        self.writer.writerow(row)



def init_args():
    parser = argparse.ArgumentParser(description="Simulation script")

    # Config file
    parser.add_argument("--config", type=str, help="Path to the config file", default="config/alanine/debug.json")

    # Simluation arguments
    parser.add_argument("--molecule", type=str, help="Path to the PDB file", default="alanine")
    parser.add_argument("--state", type=str, help="Molecule state to start the simulation", default="c5")
    parser.add_argument("--force_field", type=str, help="Force field to use", default="amber14")
    parser.add_argument("--solvent", type=str, help="Solvent to use", default="tip3p")
    parser.add_argument("--temperature", type=float, help="Temperature to use", default=300)
    parser.add_argument("--time", type=int, help="Total simulation steps", default=1e+8)
    parser.add_argument("--platform", type=str, help="Platform to use", default="OpenCL")
    parser.add_argument("--precision", type=str, help="Precision to use", default="single")
    parser.add_argument("--seed", type=int, help="Seed to use for integrator", default=0)

    # Logging
    parser.add_argument("--log_stdout", type=bool, help="Logging for stdout", default=False)
    parser.add_argument("--log_dcd", type=bool, help="Logging for dcd", default=True)
    parser.add_argument("--log_csv", type=bool, help="Logging for csv", default=True)
    parser.add_argument("--log_force", type=bool, help="Logging for force", default=False)
    parser.add_argument("--freq_stdout", type=int, help="Logging interval for stdout", default="1_000")
    parser.add_argument("--freq_dcd", type=int, help="Logging interval for dcd", default="1")
    parser.add_argument("--freq_csv", type=int, help="Logging interval for csv", default="1")

    args = parser.parse_args()
    for key, value in vars(args).items():
        print(f">> {key}: {value}")
    print("\n")
        
    return args


def set_molecule(molecule, state):
    if molecule == "alanine":
        if state in [
            "c5", "c7", "c7ax", "pII", "alpha_L", "alpha_R", "alpha_P", "debug"
        ]:
            pdb_file_name = f"alanine/{state}.pdb"
        else:
            raise ValueError(f"State {state} not recognized")
    elif molecule == "chignolin":
        if state in ["unfolded", "folded"]:
            pdb_file_name = f"chignolin/{state}.pdb"
    else:
        raise ValueError(f"Molecule {molecule} not recognized")
    
    pdb = PDBFile("./data/" + pdb_file_name) 
    return pdb


def set_force_field(force_field, solvent):
    files = []
    
    if force_field == "amber14":
        files.append("amber14-all.xml")
    elif force_field == "amber99":
        files.append("amber99sbildn.xml")
    elif force_field == "protein.ff14SBonlysc":
        files.append("./data/forcefield/protein.ff14SBonlysc.xml")
    else:
        raise ValueError(f"Force field {force_field} not recognized")
    
    if solvent == "tip3p":
        files.append("tip3pfb.xml")
    elif solvent == "implicit/gbn2":
        files.append("implicit/gbn2.xml")
    elif solvent == "vacuum":
        pass
    else:
        raise ValueError(f"Solvent {solvent} not recognized")

    return files


def set_platform(platform, precision):
    if platform == "CPU":
        platform = Platform.getPlatformByName('CPU')
        properties = {}
    elif platform == "CUDA":
        platform = Platform.getPlatformByName('CUDA')
        assert precision in ['single', 'mixed', 'double'], f"Precision {precision} not recognized"
        properties = {'DeviceIndex': 0, 'Precision': f"{precision}"}
    elif platform == "OpenCL":
        platform = Platform.getPlatformByName('OpenCL')
        assert precision in ['single', 'mixed', 'double'], f"Precision {precision} not recognized"
        properties = {'Precision': f"{precision}"}
    else:
        raise ValueError(f"Platform {platform} not recognized")
    
    return platform, properties


def set_simulation(args, forcefield_files, start_pdb, platform, properties):
    # Set force field, system, integrator for simulation
    forcefield = ForceField(*forcefield_files)
    system = forcefield.createSystem(
        start_pdb.topology,
        nonbondedCutoff=3 * nanometer,
        constraints=HBonds
    )
    integrator = LangevinIntegrator(
        args.temperature * kelvin,
        1 / picosecond,
        1 * femtoseconds
    )
    integrator.setRandomNumberSeed(args.seed)
    
    # Create simulation
    simulation = Simulation(
        start_pdb.topology,
        system,
        integrator,
        platform,
        properties
    )
    simulation.context.setPositions(start_pdb.positions)
    simulation.minimizeEnergy()
    
    return simulation

    
def set_logging(args):
    # Set logging directory
    kst = pytz.timezone('Asia/Seoul')
    current_date = datetime.now(kst).strftime("%y-%m-%d_%H:%M")
    date, time = current_date.split("_")[0], current_date.split("_")[1]
    log_dir = f"./log/{args.molecule}/{args.temperature}/{date}/{time}"
    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    else:
        raise ValueError(f"Directory {log_dir} already exists")
    with open(f"{log_dir}/args.json", 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)
    
    
    # Set reporters
    reporters = []
    print(f">> Logging to {log_dir}")
    traj_file_name = f"{log_dir}/traj.dcd"
    csv_file_name = f"{log_dir}/scalars.csv"
    force_file_name = f"{log_dir}/forces.csv"
    if args.log_stdout:
        reporters.append(
            StateDataReporter(
                stdout,
                reportInterval=args.freq_stdout,
                step=True,
                time=True,
                potentialEnergy=True,
                temperature=True,
                progress=True,
                elapsedTime=True,
                totalSteps=args.time
        ))
    if args.log_dcd:
        reporters.append(
            DCDReporter(
                file=traj_file_name,
                reportInterval=args.freq_dcd
        ))
    if args.log_csv:
        reporters.append(
            StateDataReporter(
                csv_file_name,
                reportInterval=args.freq_csv,
                time=True,
                potentialEnergy=True,
                totalEnergy=True,
                temperature=True,
        ))
    # if args.log_force:
    #     reporters.append(
    #         ForceReporter(
    #             file_name=force_file_name,
    #             reportInterval=1
    #     ))

    return reporters