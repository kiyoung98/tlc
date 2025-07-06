import os
import copy
import torch
import wandb
import numpy as np

import matplotlib.pyplot as plt
from itertools import combinations
from collections import defaultdict

from ..util.constant import *
from ..util.angle import compute_dihedral

from mlcolvar.data import DictDataset
from mlcolvar.explain import sensitivity_analysis



class AlaninePotential():
    def __init__(self, landscape_path):
        super().__init__()
        self.open_file(landscape_path)

    def open_file(self, landscape_path):
        with open(landscape_path) as f:
            lines = f.readlines()

        dims = [90, 90]

        self.locations = torch.zeros((int(dims[0]), int(dims[1]), 2))
        self.data = torch.zeros((int(dims[0]), int(dims[1])))

        i = 0
        for line in lines[1:]:
            splits = line[0:-1].split(" ")
            vals = [y for y in splits if y != '']

            x = float(vals[0])
            y = float(vals[1])
            val = float(vals[-1])

            self.locations[i // 90, i % 90, :] = torch.tensor(np.array([x, y]))
            self.data[i // 90, i % 90] = (val)  # / 503.)
            i = i + 1

    def potential(self, inp):
        loc = self.locations.view(-1, 2)
        distances = torch.cdist(inp, loc.double(), p=2)
        index = distances.argmin(dim=1)

        x = torch.div(index, self.locations.shape[0], rounding_mode='trunc')  # index // self.locations.shape[0]
        y = index % self.locations.shape[0]

        z = self.data[x, y]
        return z

    def drift(self, inp):
        loc = self.locations.view(-1, 2)
        distances = torch.cdist(inp[:, :2].double(), loc.double(), p=2)
        index = distances.argsort(dim=1)[:, :3]

        x = index // self.locations.shape[0]
        y = index % self.locations.shape[0]

        dims = torch.stack([x, y], 2)

        min = dims.argmin(dim=1)
        max = dims.argmax(dim=1)

        min_x = min[:, 0]
        min_y = min[:, 1]
        max_x = max[:, 0]
        max_y = max[:, 1]

        min_x_dim = dims[range(dims.shape[0]), min_x, :]
        min_y_dim = dims[range(dims.shape[0]), min_y, :]
        max_x_dim = dims[range(dims.shape[0]), max_x, :]
        max_y_dim = dims[range(dims.shape[0]), max_y, :]

        min_x_val = self.data[min_x_dim[:, 0], min_x_dim[:, 1]]
        min_y_val = self.data[min_y_dim[:, 0], min_y_dim[:, 1]]
        max_x_val = self.data[max_x_dim[:, 0], max_x_dim[:, 1]]
        max_y_val = self.data[max_y_dim[:, 0], max_y_dim[:, 1]]

        grad = -1 * torch.stack([max_y_val - min_y_val, max_x_val - min_x_val], dim=1)

        return grad


def plot_ram_scatter_cv(
    cfg,
    model,
    logger,
    checkpoint_path,
    projection_dataset,
):
    if cfg.data.molecule == "alanine":
        plot_path = plot_ad_scatter_cv(
            model,
            logger,
            checkpoint_path,
            projection_dataset,
        )
        
    else:
        raise ValueError(f"Ramachandran plot for molecule {cfg.data.molecule} TBA...")
        
    return plot_path

def plot_ram_hex_cv(
    cfg,
    model,
    logger,
    checkpoint_path,
    projection_dataset,
):  
    if cfg.data.molecule == "alanine":
        plot_path = plot_ad_hex_cv(
            model,
            logger,
            checkpoint_path,
            projection_dataset,
            cv_dim = 1,
        )
        
    else:
        raise ValueError(f"Ramachandran plot for molecule {cfg.data.molecule} TBA...")
        
    return plot_path


def plot_ad_scatter_cv(
    model,
    logger,
    checkpoint_path,
    projection_dataset,
    cv_dim = 1,
):  
    # Compute projection dataset
    cv = model.forward(projection_dataset).cpu().detach()
    psi_list = np.load(f"../simulation/data/alanine/uniform-psi.npy")
    phi_list = np.load(f"../simulation/data/alanine/uniform-phi.npy")
    c5 = torch.load(f"../simulation/data/alanine/c5.pt")
    c7ax = torch.load(f"../simulation/data/alanine/c7ax.pt")
    phi_start, psi_start = c5["phi"], c5["psi"]
    phi_goal, psi_goal = c7ax["phi"], c7ax["psi"]
    
    # Plot CV scatter plot
    # fig, axs = plt.subplots(2, 2, figsize = ( 15, 12 ) )
    # axs = axs.ravel()
    # for i in range(min(cv_dim, 4)):
    #     ax = axs[i]
    #     im = ax.scatter(phi_list, psi_list, c = cv[:, i], cmap=COLORMAP, s=8, alpha=0.8)
    #     plt.colorbar(im, ax=ax)
    #     ax.scatter(phi_start, psi_start, edgecolors="black", c="w", zorder=101, s=100)
    #     ax.scatter(phi_goal, psi_goal, edgecolors="black", c="w", zorder=101, s=300, marker="*")
    #     ax.set_xlabel('phi')
    #     ax.set_ylabel('psi')
    #     ax.set_xticks([])
    #     ax.set_yticks([])
    #     ax.set_xlim(-np.pi, np.pi)
    #     ax.set_ylim(-np.pi, np.pi)
    # fig.savefig(checkpoint_path + "/cv-plot.png")
    # plt.close()
    # logger.info(f"CV plot saved at {checkpoint_path + '/cv-plot.png'}")
    
    # MLCV dim 0
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111)
    im = ax.scatter(phi_list, psi_list, c = cv[:, 0], cmap=COLORMAP, s=8, alpha=0.8)
    ax.margins(0) 
    ax.tick_params(
        left = False,
        right = False ,
        labelleft = True , 
        labelbottom = True,
        bottom = False
    ) 
    ax.scatter(phi_start, psi_start, edgecolors="black", c="w", zorder=101, s=300)
    ax.scatter(phi_goal, psi_goal, edgecolors="black", c="w", zorder=101, s=800, marker="*")
    ax.set_xlabel('$\phi$', fontsize=FONTSIZE)
    ax.set_ylabel('$\psi$', fontsize=FONTSIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=FONTSIZE_SMALL)
    plt.tight_layout()
    
    plot_path = os.path.join(checkpoint_path, 'cv-scatter-plot.png')
    plt.savefig(plot_path)
    plt.savefig(plot_path.replace('.png', '.pdf'))
    plt.close()
    logger.info(f">> CV scatter plot saved at {plot_path}")
    
    return plot_path
   

def plot_ad_hex_cv(
    model,
    logger,
    checkpoint_path,
    projection_dataset,
    cv_dim = 1,
):  
    # Compute projection dataset
    cv = model(projection_dataset).cpu().detach()
    psi_list = np.load(f"../simulation/data/alanine/uniform-psi.npy")
    phi_list = np.load(f"../simulation/data/alanine/uniform-phi.npy")
    c5 = torch.load(f"../simulation/data/alanine/c5.pt")
    c7ax = torch.load(f"../simulation/data/alanine/c7ax.pt")
    phi_start, psi_start = c5["phi"], c5["psi"]
    phi_goal, psi_goal = c7ax["phi"], c7ax["psi"]
    
    # # Plot CV hexa
    # for i in range(min(cv_dim, 4)):
    #     ax = axs[i]
    #     hb = ax.hexbin(
    #         phi_list, psi_list, C=cv[:, i],  # data
    #         gridsize=30,                     # controls resolution
    #         reduce_C_function=np.mean,       # compute average per hexagon
    #         cmap=COLORMAP,                  # colormap
    #         extent=[-3.15, 3.15, -3.15, 3.15]
    #     )
    #     ax.scatter(phi_start, psi_start, edgecolors="black", c="w", zorder=101, s=100)
    #     ax.scatter(phi_goal, psi_goal, edgecolors="black", c="w", zorder=101, s=300, marker="*")
    #     ax.set_xlabel('phi')
    #     ax.set_ylabel('psi')
    #     ax.set_title(f'CV Dimension {i}')
    #     cbar = plt.colorbar(hb, ax=ax)
    #     cbar.set_label('CV Value')
    # plt.savefig(checkpoint_path + "/cv-hexplot.png")
    # plt.close()
    # logger.info(f">> CV hexplot saved at {checkpoint_path + '/cv-hexplot.png'}")
    
    # MLCV dim 0
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    hb = ax.hexbin(
        phi_list, psi_list, C=cv[:, 0],  # data
        gridsize=HEX_GRIDSIZE,                     # controls resolution
        reduce_C_function=np.mean,       # compute average per hexagon
        cmap=COLORMAP,                  # colormap
        extent=[-np.pi, np.pi, -np.pi, np.pi]
    )
    ax.margins(0) 
    ax.tick_params(
        left = False,
        right = False ,
        labelleft = True , 
        labelbottom = True,
        bottom = False
    ) 
    ax.scatter(phi_start, psi_start, edgecolors="black", c="w", zorder=101, s=300)
    ax.scatter(phi_goal, psi_goal, edgecolors="black", c="w", zorder=101, s=800, marker="*")
    ax.set_xlabel('$\phi$', fontsize=FONTSIZE)
    ax.set_ylabel('$\psi$', fontsize=FONTSIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    # cbar = plt.colorbar(hb, ax=ax)
    # cbar.ax.tick_params(labelsize=FONTSIZE_SMALL)
    # cbar = plt.colorbar(hb, ax=ax, orientation='horizontal')

    plot_path = os.path.join(checkpoint_path, 'cv-hexplot.png')
    plt.savefig(plot_path)
    plt.savefig(plot_path.replace('.png', '.pdf'))
    plt.close()
    logger.info(f">> CV hexplot saved at {plot_path}")
    
    # # Plot colorbar
    # fig, ax = plt.subplots(figsize=(18, 1.5))
    # # fig.subplots_adjust(bottom=0.5)
    # cmap = plt.get_cmap("viridis")
    # norm = plt.Normalize(vmin=np.min(cv.numpy()), vmax=np.max(cv.numpy()))
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # cbar = plt.colorbar(sm, cax=ax, orientation='horizontal')
    # # cbar.set_label('MLCV', fontsize=FONTSIZE)
    # cbar.ax.tick_params(labelsize=FONTSIZE)
    # cbar.ax.xaxis.set_ticks_position('bottom')
    # # cbar.ax.xaxis.set_label_position('bottom')
    # plt.tight_layout()
    
    # plot_path = os.path.join(checkpoint_path, 'cv-colorbar3.png')
    # plt.savefig(plot_path)
    # plt.savefig(plot_path.replace('.png', '.pdf'))
    # plt.close()
    # logger.info(f">> CV colrbar saved at {plot_path}")
    
    return plot_path


def plot_sensitivity(
    cfg,
    model,
    logger,
    checkpoint_path,
    projection_dataset,
):
    top_k = 10
    projection_dataset = projection_dataset.clone().cpu()
    
    # Set descriptor
    if cfg.model.representation == "heavy_atom_distance":
        descriptors = np.array([f"{a} - {b}" for a, b in combinations(ALANINE_HEAVY_ATOM_DESCRIPTOR, 2)], dtype=str)  
    elif cfg.model.representation == "heavy_atom_coordinate":
        descriptors = np.repeat(np.array([f"{a}" for a in ALANINE_HEAVY_ATOM_DESCRIPTOR], dtype=str), 3)
    elif cfg.model.representation == "heavy_atom_coordinate_distance":
        descriptors = np.concatenate([
            np.repeat(np.array([f"{a}" for a in ALANINE_HEAVY_ATOM_DESCRIPTOR], dtype=str), 3),
            np.array([f"{a} - {b}" for a, b in combinations(ALANINE_HEAVY_ATOM_DESCRIPTOR, 2)], dtype=str)
        ])
    else:
        raise ValueError(f"Representation {cfg.model.representation} not found for sensitivity plot")
        
    # Compute senstivity by mlcolvar
    device = model.device
    model_cpu = model.cpu()
    model_cpu.eval()
    results = sensitivity_analysis(
        model_cpu,
        DictDataset(
            {"data": projection_dataset.reshape(projection_dataset.shape[0], -1)},
            feature_names = descriptors
        ),
        metric="mean_abs_val",   # metric to use to compute the sensitivity per feature (e.g. mean absolute value or root mean square)
        per_class=False,
        plot_mode=None
    )
    model = model.to(device)
    del model_cpu
    sensitivity = results['sensitivity']['Dataset'][::-1]
    feature_names = results['feature_names'][::-1]
    if cfg.model.representation == "heavy_atom_coordinate":
        agg = defaultdict(float)
        for k, v in zip(feature_names, sensitivity):
            agg[k] += v
        feature_names = np.array(list(agg.keys()))
        sensitivity  = np.array(list(agg.values()))
    sensitivity = sensitivity[:top_k]
    feature_names = feature_names[:top_k]
    
    # Plot sensitivity
    plt.clf()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    ax.tick_params(
        left = False,
        right = False ,
        labelleft = True , 
        labelbottom = True,
        bottom = False
    ) 
    y = np.arange(top_k)
    colors = [COLORS[1] if i < 3 else COLORS[-1] for i in range(top_k)]
    plt.barh(y, sensitivity, color=colors)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE_SMALL)
    ax.set_yticks(y, feature_names, fontsize=FONTSIZE_SMALL)
    ax.set_xlabel('Sensitivity', fontsize=FONTSIZE)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    plot_path = os.path.join(checkpoint_path, 'sensitivity.png')
    plt.savefig(plot_path)
    plt.savefig(plot_path.replace('.png', '.pdf'))
    plt.close()
    logger.info(f">> Sensitivity plot saved at {plot_path}")
    
    return plot_path


def plot_paths(
    cfg,
    trajectory_list,
    logger,
    hit_mask,
    hit_index,
    seed,
    checkpoint_path,
):
    molecule = cfg.steeredmd.molecule
    
    if molecule == "alanine":
        start_state_info = torch.load(f"../simulation/data/{molecule}/{cfg.steeredmd.start_state}.pt")
        goal_state_info = torch.load(f"../simulation/data/{molecule}/{cfg.steeredmd.goal_state}.pt")
        phi_start, psi_start = start_state_info["phi"], start_state_info["psi"]
        phi_goal, psi_goal = goal_state_info["phi"], goal_state_info["psi"]
    
        # Compute phi, psi from trajectory_list
        phi_traj_list = [compute_dihedral(trajectory[:, ALDP_PHI_ANGLE]) for trajectory in trajectory_list]
        psi_traj_list = [compute_dihedral(trajectory[:, ALDP_PSI_ANGLE]) for trajectory in trajectory_list]
        ram_plot_img = plot_ad_traj(
            logger,
            checkpoint_path = checkpoint_path,
            traj_dihedral = (phi_traj_list, psi_traj_list),
            start_dihedral = (phi_start, psi_start),
            goal_dihedral = (phi_goal, psi_goal),
            type = "all",
            seed = seed,
        )
        
        hit_phi_traj_list = [phi_traj_list[i][:hit_index[i]] for i in range(len(phi_traj_list)) if hit_mask[i]]
        hit_psi_traj_list = [psi_traj_list[i][:hit_index[i]] for i in range(len(psi_traj_list)) if hit_mask[i]]
        transition_path_plot_img = plot_ad_traj(
            logger,
            checkpoint_path = checkpoint_path,
            traj_dihedral = (hit_phi_traj_list, hit_psi_traj_list),
            start_dihedral = (phi_start, psi_start),
            goal_dihedral = (phi_goal, psi_goal),
            type = "hit",
            seed = seed,
        )

    elif molecule == "chignolin":
        raise ValueError(f"Projection for molecule {molecule} TBA...")
    
    else:
        raise ValueError(f"Ramachandran plot for molecule {molecule} TBA...")
    
    return ram_plot_img, transition_path_plot_img


def plot_ad_traj(
    logger,
    checkpoint_path,
    traj_dihedral,
    start_dihedral,
    goal_dihedral,
    type,
    seed,
):
    traj_list_phi = traj_dihedral[0]
    traj_list_psi = traj_dihedral[1]
    sample_num = len(traj_dihedral[0])

    # Plot the potential
    xs = np.arange(-np.pi, np.pi + 0.1, 0.1)
    ys = np.arange(-np.pi, np.pi + 0.1, 0.1)
    x, y = np.meshgrid(xs, ys)
    inp = torch.tensor(np.array([x, y])).view(2, -1).T
    potential = AlaninePotential(f"../simulation/data/alanine/final_frame.dat")
    z = potential.potential(inp)
    z = z.view(y.shape[0], y.shape[1])

    # Plot the trajectory
    plt.clf()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    plt.contourf(xs, ys, z, levels=100, zorder=0)
    cm = plt.get_cmap("gist_rainbow")
    ax.set_prop_cycle(
        color=[cm(1.0 * i / sample_num) for i in range(sample_num)]
    )
    for idx in range(sample_num):
        ax.plot(
            traj_list_phi[idx].cpu().detach().numpy(),
            traj_list_psi[idx].cpu().detach().numpy(),
            marker="o",
            linestyle="None",
            markersize=3,
            alpha=1.0,
            zorder=100
        )

    # Plot start, goal states, hit boundary
    ax.scatter(start_dihedral[0], start_dihedral[1], edgecolors="black", c="w", zorder=101, s=300)
    ax.scatter(goal_dihedral[0], goal_dihedral[1], edgecolors="black", c="w", zorder=101, s=800, marker="*")
    hitbox = plt.Circle(
        (goal_dihedral[0], goal_dihedral[1] ),
        radius = ALANINE_CV_BOUND,
        color='r', fill=False, linewidth = 6,
        zorder=101
    )
    plt.gca().add_patch(hitbox)
    
    # Formatting
    ax.margins(0) 
    ax.tick_params(
        left = False,
        right = False ,
        labelleft = True , 
        labelbottom = True,
        bottom = False
    ) 
    ax.set_xlabel('$\phi$', fontsize=FONTSIZE)
    ax.set_ylabel('$\psi$', fontsize=FONTSIZE)
    ax.set_xticks([])
    ax.set_yticks([])
    
    plot_path = os.path.join(checkpoint_path, f'{seed}-{type}-paths.png')
    plt.savefig(plot_path)
    plt.savefig(plot_path.replace('.png', '.pdf'))
    plt.close()
    logger.info(f">> Transition path plot saved at {plot_path}")
    
    return wandb.Image(plot_path)
    
    
def plot_mlcv(
    mlcv_list,
    goal_mlcv,
    logger,
    repeat_idx,
    checkpoint_path
):    
    n_trajectories, time_horizon, cv_dim = mlcv_list.shape
    mlcv_list = mlcv_list.cpu().detach().numpy()
    start_mlcv = mlcv_list[0][0]
    start_mlcv = start_mlcv.reshape(-1)
    start_color = COLORS[2]
    goal_mlcv = goal_mlcv.squeeze().cpu().detach().numpy()
    goal_mlcv = goal_mlcv.reshape(-1)
    goal_color = COLORS[1]
    
    # Plot
    dim = 0 
    plt.clf()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    for traj_idx in range(n_trajectories):
        ax.plot(range(time_horizon), 
            mlcv_list[traj_idx, :, dim],
            linewidth=2
        )
    ax.axhline(
        y=start_mlcv[dim], 
        color=start_color, 
        linestyle='--', 
        label='Start MLCV',
        linewidth=4
    )
    ax.axhline(
        y=goal_mlcv[dim], 
        color=goal_color, 
        linestyle='--', 
        label='Goal MLCV',
        linewidth=4
    )
    ax.tick_params(
        left = False,
        right = False ,
        labelleft = True , 
        labelbottom = True,
        bottom = False
    ) 
    
    # Format
    ax.set_xlabel('Time Step (fs)', fontsize=FONTSIZE)
    ax.set_ylabel(f'MLCV', fontsize=FONTSIZE)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE_SMALL)
    ax.grid(True, alpha=0.5, linestyle="dotted")
    ax.legend(fontsize=FONTSIZE_SMALL)
    plt.tight_layout()
    
    plot_path = f"{checkpoint_path}/{repeat_idx}-mlcv.png"
    plt.savefig(plot_path)
    plt.savefig(plot_path.replace('.png', '.pdf'))
    plt.close()
    logger.info(f">> MLCV trajectories plot saved at {plot_path}")
    
    return wandb.Image(plot_path)


def plot_path_energy(
    logger,
    hit_mask,
    path_energy,
    repeat_idx,
    checkpoint_path
):
    n_trajectories, time_horizon, _ = path_energy.shape
    
    # Plot all path energy trajectories
    plt.clf()
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    for traj_idx in range(n_trajectories):
        ax.plot(
            range(time_horizon), 
            path_energy[traj_idx],
            linewidth=2,
        )
    ax.tick_params(
        left = False,
        right = False ,
        labelleft = True , 
        labelbottom = True,
        bottom = False
    ) 
    ax.set_xlabel('Time Step (fs)', fontsize=FONTSIZE)
    ax.set_ylabel(f'Energy (kJ/mol)', fontsize=FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE_SMALL)
    ax.grid(True, alpha=0.5, linestyle="dotted")
    plt.tight_layout()
    
    plot_path_all = f"{checkpoint_path}/{repeat_idx}-all-path-energy.png"
    plt.savefig(plot_path_all)
    plt.savefig(plot_path_all.replace('.png', '.pdf'))
    plt.close()
    logger.info(f">> Energy trajectories plot saved at {plot_path_all}")
    
    # Plot hitting path energy trajectories
    plt.clf()
    fig = plt.figure(figsize=(9, 6))
    ax = fig.add_subplot(111)
    for traj_idx in hit_mask.nonzero().reshape(-1):
        ax.plot(
            range(time_horizon), 
            path_energy[traj_idx],
            linewidth=2,
        )
    ax.tick_params(
        left = False,
        right = False ,
        labelleft = True , 
        labelbottom = True,
        bottom = False
    ) 
    ax.set_xlabel('Time Step (fs)', fontsize=FONTSIZE)
    ax.set_ylabel(f'Energy (kJ/mol)', fontsize=FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=FONTSIZE_SMALL)
    ax.grid(True, alpha=0.5, linestyle="dotted")
    plt.tight_layout()
    
    plot_path_hitting = f"{checkpoint_path}/{repeat_idx}-hit-path-energy.png"
    plt.savefig(plot_path_hitting)
    plt.savefig(plot_path_hitting.replace('.png', '.pdf'))
    plt.close()
    logger.info(f">> Energy hitting trajectories plot saved at {plot_path_hitting}")
    
    return wandb.Image(plot_path_all), wandb.Image(plot_path_hitting)