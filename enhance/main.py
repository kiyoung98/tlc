import os
import wandb
import argparse

from src import *

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze enhanced sampling simulations')
    parser.add_argument('--plumed', type = str, default = "./simulations/aldp/DeepLDA/plumed.dat", help='File to plumed')
    parser.add_argument('--date', type = str, default = "debug", help='Date for the experiment')
    parser.add_argument('--ckpt', type = str, default = "debug", help='Checkpoint name')
    parser.add_argument('--method', type = str, default = "phi", help='Method name')
    parser.add_argument('--ns', type = int, default = 10, help = "Length of simulation used for dataset")
    parser.add_argument('--step', type = int, default = 10000000, help = "Number of steps")
    parser.add_argument('--seed', type = int, default = 0, help = "Number of seeds")
    parser.add_argument('--tags', nargs = '*', help='Tags for Wandb')
    
    return parser.parse_args()

def parse_plumed_metad(filename):
    params = {}
    inside_enhance_sampling = False

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            if line.startswith("METAD"):
                inside_enhance_sampling = True
                prefix = "metad"
                continue
            elif line.startswith("opes"):
                inside_enhance_sampling = True
                prefix = "opes"
                continue

            if inside_enhance_sampling and line.startswith("... METAD"):
                break
            elif inside_enhance_sampling and line.startswith("opes"):
                break

            if inside_enhance_sampling and line and not line.startswith("#"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    # if key.strip() in ['ARG', 'FILE', 'STATE_WFILE', 'STATE_WSTRIDE']:
                    params[prefix + "/" + key.strip()] = value.strip()
                    # else:
                        # params[prefix + "/" + key.strip()] = float(value.strip())
        
    return params


args = parse_args()
base_dir = f'{os.getcwd()}/simulations/aldp/{args.method}/{args.ns}'
plumed_params = parse_plumed_metad(args.plumed)

wandb.init(
    project="enhanced-sampling",
    entity="eddy26",
    tags = args.tags,
    config = vars(args),
)
wandb.log(plumed_params)

print(">>Plotting phi distribution")
plot_phi_distribution(args, base_dir)
print(">>Plotting free energy difference")
plot_free_energy_difference(args, base_dir)
if args.method != "ref":
    print(">>Plotting FES over CV")
    plot_fes_over_cv(args, base_dir)

wandb.finish()