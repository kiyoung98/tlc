import os
import json
import hydra
import torch
import logging

from omegaconf import OmegaConf
import wandb

from tqdm import tqdm
from dynamics import SMDs
from constant import METHODS

@hydra.main(version_base=None)
def main(cfg):
    # Load configs and components
    wandb.init(
        project = cfg.logger.project,
        tags = cfg.logger.tags,
        config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )
    logger = logging.getLogger("MLCVs")
    logger.info(">> Configs")
    logger.info(OmegaConf.to_yaml(cfg))

    res_path = f"./res/{cfg.molecule}/{cfg.model.name}/{cfg.simulation.k}"
    os.makedirs(res_path, exist_ok=True)
    
    if cfg.model.name in ["rmsd"]:
        model = None
    elif cfg.model.name in METHODS:
        checkpoint_file = f"./models/{cfg.molecule}/{cfg.model.name}.pt"
        logger.info(f">> Load model from {checkpoint_file}")
        
        # Load JIT model directly
        logger.info(f">> Load JIT model from {checkpoint_file}")
        model = torch.jit.load(checkpoint_file, map_location='cuda')
        model.device = torch.device('cuda')
        model.eval()
    
    # Simulation
    logger.info(">> Start simulation")
    smds = SMDs(cfg, model)

    positions = torch.empty(size=(cfg.num_samples, cfg.simulation.num_steps, smds.num_atoms, 3), dtype=torch.float32, device=smds.device)
    potentials = torch.empty(size=(cfg.num_samples, cfg.simulation.num_steps), dtype=torch.float32, device=smds.device)
    position, potential = smds.report()
    positions[:, 0] = position
    potentials[:, 0] = potential

    for step in tqdm(range(cfg.simulation.num_steps), desc = f"Simulating {cfg.num_samples} paths"):
        smds.step(step, position)
        position, potential = smds.report()
        positions[:, step] = position
        potentials[:, step] = potential

    torch.save(positions, f"{res_path}/positions.pt")
    torch.save(potentials, f"{res_path}/potentials.pt")
    logger.info(f">> {positions.shape[0]} paths saved at: {res_path}/positions.pt")
    logger.info(f">> {positions.shape[0]} potentials saved at: {res_path}/potentials.pt")
    logger.info(">> Finish simulation")
    logger.info(">> Start evaluation")

    # Evaluation
    res_metrics = smds.metrics(positions[:, -1], potentials, cfg.rmsd_threshold)
    res_plots = smds.plot(positions, potentials)
    
    # Save metrics
    with open(f"{res_path}/metrics.json", 'w') as f:
        json.dump(res_metrics, f, indent=2)
    
    # Save plot image
    res_plots["paths"].savefig(f"{res_path}/paths.png", dpi=300, bbox_inches='tight')
    res_plots["potentials"].savefig(f"{res_path}/potentials.png", dpi=300, bbox_inches='tight')
    logger.info(f">> Paths plot saved at: {f'{res_path}/paths.png'}")
    logger.info(f">> Potentials plot saved at: {f'{res_path}/potentials.png'}")
    logger.info(f">> Metrics saved at: {f'{res_path}/metrics.json'}")
    logger.info(f">> SMD result: {res_metrics}")
    logger.info(">> Finish evaluation")

    wandb.log(res_metrics)
    wandb.log(
        {
            "paths": wandb.Image(f"{res_path}/paths.png"),
            "potentials": wandb.Image(f"{res_path}/potentials.png")
        }
    )
    wandb.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()