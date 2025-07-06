import torch
import wandb
import hydra
import logging

from numbers import Number
from pprint import pformat

from .load import *
from .metric import *
from .util.plot import *
from .simulation import simulate_steered_md


def eval(
    cfg,
    model,
    logger,
    checkpoint_path
):
    # Evaluation
    checkpoint_path = f"{checkpoint_path}/{cfg.steeredmd.simulation.k}"
    eval_dict = {}
    
    
    for repeat_idx in range(0, cfg.steeredmd.repeat):
        np.random.seed(repeat_idx)
        torch.manual_seed(repeat_idx)
        logger.info(f">> Steered MD evaluation #{repeat_idx}")
        trajectory_list, mlcv_list, goal_mlcv = simulate_steered_md(cfg, model, logger, repeat_idx, checkpoint_path)
        steered_md_metric = evalute_steered_md(cfg, logger, repeat_idx, checkpoint_path, trajectory_list, mlcv_list, goal_mlcv)
        
        logger.info(f">> Steered MD result: {steered_md_metric}")
        wandb.log(steered_md_metric)
        eval_dict.update(steered_md_metric)
        
    if cfg.steeredmd.repeat > 0:
        eval_dict["steered_md/thp/average"] = np.mean([
            value for key, value in eval_dict.items()
            if key.startswith("steered_md/thp") and value != None
        ])
        
        epd_list = [
            value for key, value in eval_dict.items()
            if key.startswith("steered_md/epd/avg_") and isinstance(value, Number)
        ]
        if epd_list:
            eval_dict["steered_md/epd/avg"] = np.mean(epd_list)
            epd_list_std = np.array([
                value for key, value in eval_dict.items()
                if key.startswith("steered_md/epd/std_") and isinstance(value, Number)
            ])
            eval_dict["steered_md/epd/std"] = np.sqrt(np.mean(epd_list_std ** 2) + np.var(epd_list))
        
        rmsd_list = [
            value for key, value in eval_dict.items()
            if key.startswith("steered_md/rmsd/avg_") and isinstance(value, Number)
        ]
        if rmsd_list:
            eval_dict["steered_md/rmsd/avg"] = np.mean(rmsd_list)
            rmsd_lsit_sd = np.array([
                value for key, value in eval_dict.items()
                if key.startswith("steered_md/rmsd/std_") and isinstance(value, Number)
            ])
            eval_dict["steered_md/rmsd/std"] = np.sqrt(np.mean(rmsd_lsit_sd ** 2) + np.var(rmsd_list))
        
        # All paths max energy
        max_energy_list_avg = np.array([
            value for key, value in eval_dict.items()
            if key.startswith("steered_md/all_paths/max_energy/avg_") and isinstance(value, Number)
        ])
        if max_energy_list_avg.shape[0] > 0:
            eval_dict["steered_md/all_paths/max_energy/avg"] = np.mean(max_energy_list_avg) 
            max_energy_list_std = np.array([
                value for key, value in eval_dict.items()
                if key.startswith("steered_md/all_paths/max_energy/std_") and isinstance(value, Number)
            ])
            eval_dict["steered_md/all_paths/max_energy/std"] = np.sqrt(np.mean(max_energy_list_std ** 2) + np.var(max_energy_list_avg))
        
        # Hitting paths max energy
        max_energy_list_avg = np.array([
            value for key, value in eval_dict.items()
            if key.startswith("steered_md/hitting_paths/max_energy/avg_") and isinstance(value, Number)
        ])
        if max_energy_list_avg.shape[0] > 0:
            eval_dict["steered_md/hitting_paths/max_energy/avg"] = np.mean(max_energy_list_avg) 
            max_energy_list_std = np.array([
                value for key, value in eval_dict.items()
                if key.startswith("steered_md/hitting_paths/max_energy/std_") and isinstance(value, Number)
            ])
            eval_dict["steered_md/hitting_paths/max_energy/std"] = np.sqrt(np.mean(max_energy_list_std ** 2) + np.var(max_energy_list_avg))
        
        # All paths final energy
        final_energy_list_avg = np.array([
            value for key, value in eval_dict.items()
            if key.startswith("steered_md/all_paths/final_energy/avg_") and isinstance(value, Number)
        ])
        if final_energy_list_avg.shape[0] > 0:
            eval_dict["steered_md/all_paths/final_energy/avg"] = np.mean(final_energy_list_avg) 
            final_energy_list_std = np.array([
                value for key, value in eval_dict.items()
                if key.startswith("steered_md/all_paths/final_energy/std_") and isinstance(value, Number)
            ])
            eval_dict["steered_md/all_paths/final_energy/std"] = np.sqrt(np.mean(final_energy_list_std ** 2) + np.var(final_energy_list_avg))
            
        # Hitting paths final energy
        final_energy_list_avg = np.array([
            value for key, value in eval_dict.items()
            if key.startswith("steered_md/hitting_paths/final_energy/avg_") and isinstance(value, Number)
        ])
        if final_energy_list_avg.shape[0] > 0:
            eval_dict["steered_md/hitting_paths/final_energy/avg"] = np.mean(final_energy_list_avg) 
            final_energy_list_std = np.array([
                value for key, value in eval_dict.items()
                if key.startswith("steered_md/hitting_paths/final_energy/std_") and isinstance(value, Number)
            ])
            eval_dict["steered_md/hitting_paths/final_energy/std"] = np.sqrt(np.mean(final_energy_list_std ** 2) + np.var(final_energy_list_avg))
        
        
    keys_with_average = [key for key in eval_dict.keys() if key.endswith("avg") or key.endswith("std") or key.endswith("average")]
    eval_dict_avereage = {key: eval_dict[key] for key in keys_with_average}
    wandb.log(eval_dict_avereage)
    logger.info(f">> Steered MD average result\n{pformat(eval_dict_avereage, sort_dicts=True)}")
    
    return


def evalute_steered_md(cfg, logger, repeat_idx, checkpoint_path, trajectory_list, mlcv_list, goal_mlcv):
    result_dict = {}

    result_dict[f"steered_md/thp/avg_{repeat_idx}"], hit_mask, hit_index = compute_thp(cfg, trajectory_list)
    result_dict[f"steered_md/epd/avg_{repeat_idx}"], result_dict[f"steered_md/epd/std_{repeat_idx}"], \
    result_dict[f"steered_md/rmsd/avg_{repeat_idx}"], result_dict[f"steered_md/rmsd/std_{repeat_idx}"] \
    = compute_epd(cfg, trajectory_list, logger, hit_mask, hit_index)
    result_dict[f"steered_md/all_paths/max_energy/avg_{repeat_idx}"], result_dict[f"steered_md/all_paths/max_energy/std_{repeat_idx}"], \
    result_dict[f"steered_md/all_paths/final_energy/avg_{repeat_idx}"], result_dict[f"steered_md/all_paths/final_energy/std_{repeat_idx}"], \
    result_dict[f"steered_md/hitting_paths/max_energy/avg_{repeat_idx}"], result_dict[f"steered_md/hitting_paths/max_energy/std_{repeat_idx}"], \
    result_dict[f"steered_md/hitting_paths/final_energy/avg_{repeat_idx}"], result_dict[f"steered_md/hitting_paths/final_energy/std_{repeat_idx}"], \
    all_path_max_energy_idx, hitting_path_max_energy_idx, path_energy_list = compute_energy(cfg, trajectory_list, hit_mask)
    
    result_dict[f"steered_md/all_paths/mlcv/{repeat_idx}"] = plot_mlcv(mlcv_list, goal_mlcv, logger, repeat_idx, checkpoint_path)
    result_dict[f"steered_md/all_paths/{repeat_idx}"], result_dict[f"steered_md/hitting_paths/{repeat_idx}"] = plot_paths(cfg, trajectory_list, logger, hit_mask, hit_index, repeat_idx, checkpoint_path)
    result_dict[f"steered_md/all_paths/energy/{repeat_idx}"], result_dict[f"steered_md/hitting_paths/energy/{repeat_idx}"] = plot_path_energy(logger, hit_mask, path_energy_list, repeat_idx, checkpoint_path)
    result_dict[f"steered_md/committor_analysis/{repeat_idx}"] = committor_analysis(cfg, logger, hit_mask, all_path_max_energy_idx, hitting_path_max_energy_idx, repeat_idx, checkpoint_path)
    
    return result_dict


def committor_analysis(cfg, logger, hit_mask, all_path_max_energy_idx, hitting_path_max_energy_idx, repeat_idx, checkpoint_path):
    # Load simulation
    # committor_simulation = CommittorSimulation(cfg = cfg)
    # time_horizon = cfg.steeredmd.simulation.time_horizon
        
    # # Shape: (sample_num, time_horizon, atom_num, 3)
    # trajectory_list = torch.empty(size=(sample_num, time_horizon, cfg.data.atom, 3)).to(model.device)
    # mlcv_list = torch.empty(size=(sample_num, time_horizon, goal_mlcv.shape[1])).to(model.device)
    # current_position, current_mlcv = steered_md_simulation.report()
    # trajectory_list[:, 0, : ] = current_position
    # mlcv_list[:, 0, : ] = current_mlcv
    
    # # simulate
    # try:    
    #     for step in tqdm(
    #         range(1, time_horizon+1),
    #         desc = f"Genearting {sample_num} trajectories for {time_horizon} steps",
    #     ):
    #         steered_md_simulation.step(step)
    #         current_position, current_mlcv = steered_md_simulation.report()
    #         trajectory_list[:, step - 1, : ] = current_position
    #         mlcv_list[:, step - 1, : ] = current_mlcv
    
    return


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="basic"
)
def main(cfg):
    wandb.init(
        project = "mlcv",
        entity = "eddy26",
    )
    logger = logging.getLogger("MLCVs")
    logger.info(">> Configs")
    logger.info(OmegaConf.to_yaml(cfg))
    device = "cuda"
    
    if cfg.model.name in ["dihedral", "rmsd"]:
        checkpoint_path = f"./model/{cfg.model.name}/"
        logger.info(">> Evaluation for dihedral angles")
        eval(
            cfg = cfg,
            model = model,
            logger = logger
        )
    else:
        # Load model and data
        checkpoint_path = f"./model/{cfg.model.name}/{cfg.data.version}"
        logger.info(">> Loading model from checkpoint and datamodule")
        model = load_model(cfg).to(device)
        model = load_checkpoint(cfg, model)
        model.eval()
        
        # Evaluate
        eval(
            cfg = cfg,
            model = model,
            logger = logger,
            checkpoint_path = checkpoint_path
        )

if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()
