import os 
import json
import wandb
import argparse


import numpy as np
import pandas as pd
import mdtraj as md
import nglview as nv
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
# from nglview.contrib.movie import MovieMaker


from util.plot import *

def init_args():
    parser = argparse.ArgumentParser(description="Simulation result")
    parser.add_argument("--path", type=str, help="Path to the simulation result file", default="log/alanine/300.0/test")
    args = parser.parse_args()
    
    return args


if __name__ == "__main__":    
    # Load configs and save them
    print(f">> Loading configs...")
    args = init_args()
    result_path = args.path
    if not os.path.exists(result_path):
        raise FileNotFoundError(f"Result path {result_path} does not exist")
    with open(f"{result_path}/args.json", "r") as f:
        config = json.load(f)    
    
    # Load the simulation result
    print(f">> Loading simulation result...")
    pdb_file = f"./data/{config['molecule']}/{config['state']}.pdb"
    df = pd.read_csv(f"{result_path}/scalars.csv")
    traj = md.load(f"{result_path}/traj.dcd", top=pdb_file)
    
    # Compute plots and values
    print(f">> Computing plots and values...")
    plot_log_dict = {}
    plot_log_dict["ramachandran"], plot_log_dict["angle"] = wandb.Image(plot_ramachandran(traj, pdb_file, result_path)[0]), wandb.Image(plot_ramachandran(traj, pdb_file, result_path)[1])
    plot_log_dict["potential_energy"] = wandb.Image(plot_potential_energy(df, result_path))
    plot_log_dict["total_energy"] = wandb.Image(plot_total_energy(df, result_path))
    plot_log_dict["temperature"] = wandb.Image(plot_temperature(df, result_path))
    
    # Log to wandb
    print(">> Logging...")
    wandb.init(
        project = "simulation",
        entity = "eddy26",
        name = f"{config['molecule']}_{config['state']}_{config['seed']}_{config['time'] / 1000000}ns",
        config = config
    )
    wandb.log(plot_log_dict)
    wandb.finish()
    
    print("Done!")