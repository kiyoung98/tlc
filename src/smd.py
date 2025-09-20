import os
import json
import hydra
import torch
import numpy as np
import logging

from omegaconf import OmegaConf
import wandb

from tqdm import tqdm
from dynamics import SMDs
from constant import METHODS

@hydra.main(version_base=None)
def main(cfg):
    # Load configs and components
    if cfg.logger.enable:
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
    num_cas = len(smds.ca_indices)
    
    ca_positions = torch.empty(size=(cfg.num_samples, cfg.simulation.num_steps, num_cas, 3), dtype=torch.float32, device=smds.device)
    potentials = torch.empty(size=(cfg.num_samples, cfg.simulation.num_steps), dtype=torch.float32, device=smds.device)
    biased_potentials = torch.empty(size=(cfg.num_samples, cfg.simulation.num_steps), dtype=torch.float32, device=smds.device)
    total_potentials = torch.empty(size=(cfg.num_samples, cfg.simulation.num_steps), dtype=torch.float32, device=smds.device)
    cvs = torch.empty(size=(cfg.num_samples, cfg.simulation.num_steps), dtype=torch.float32, device=smds.device)
    target_cvs = torch.empty(size=(cfg.num_samples, cfg.simulation.num_steps), dtype=torch.float32, device=smds.device)

    for step in tqdm(range(cfg.simulation.num_steps), desc = f"Simulating {cfg.num_samples} paths"):
        ca_position, potential, biased_potential, total_potential = smds.report()
        bias_ca_force, cv, target_cv = smds.bias_ca_force(step, ca_position)
        smds.step(bias_ca_force)

        ca_positions[:, step] = ca_position
        potentials[:, step] = potential
        biased_potentials[:, step] = biased_potential
        total_potentials[:, step] = total_potential
        cvs[:, step] = cv
        target_cvs[:, step] = target_cv

    # Evaluation
    metrics, rmsds, etss = smds.metrics(ca_positions[:, -1], potentials, cfg.rmsd_threshold)
    plots = smds.plot(ca_positions, potentials, biased_potentials, total_potentials, cvs, target_cvs)
        
    np.save(f"{res_path}/ca_positions.npy", ca_positions.detach().cpu().numpy())
    np.save(f"{res_path}/potentials.npy", potentials.detach().cpu().numpy())
    np.save(f"{res_path}/biased_potentials.npy", biased_potentials.detach().cpu().numpy())
    np.save(f"{res_path}/total_potentials.npy", total_potentials.detach().cpu().numpy())
    np.save(f"{res_path}/cvs.npy", cvs.detach().cpu().numpy())
    np.save(f"{res_path}/target_cvs.npy", target_cvs.detach().cpu().numpy())
    np.save(f"{res_path}/start_ca_positions.npy", smds.start_ca_position.detach().cpu().numpy())
    np.save(f"{res_path}/goal_ca_positions.npy", smds.goal_ca_position.detach().cpu().numpy())
    np.save(f"{res_path}/rmsds.npy", rmsds.detach().cpu().numpy())
    np.save(f"{res_path}/etss.npy", etss.detach().cpu().numpy())
    logger.info(f">> {cvs.shape[0]} paths saved at: {res_path}/ca_positions.npy")
    logger.info(f">> {cvs.shape[0]} potentials saved at: {res_path}/potentials.npy")
    logger.info(f">> {cvs.shape[0]} biased potentials saved at: {res_path}/biased_potentials.npy")
    logger.info(f">> {cvs.shape[0]} total potentials saved at: {res_path}/total_potentials.npy")
    logger.info(f">> {cvs.shape[0]} cvs saved at: {res_path}/cvs.npy")
    logger.info(f">> {cvs.shape[0]} target cvs saved at: {res_path}/target_cvs.npy")
    logger.info(f">> {cvs.shape[0]} start ca positions saved at: {res_path}/start_ca_positions.npy")
    logger.info(f">> {cvs.shape[0]} goal ca positions saved at: {res_path}/goal_ca_positions.npy")
    logger.info(f">> {cvs.shape[0]} rmsds saved at: {res_path}/rmsds.npy")
    logger.info(f">> {cvs.shape[0]} etss saved at: {res_path}/etss.npy")
    logger.info(">> Finish simulation")
    logger.info(">> Start evaluation")
    
    # Save metrics
    with open(f"{res_path}/metrics.json", 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save plot image
    plots["paths"].savefig(f"{res_path}/paths.png", dpi=300, bbox_inches='tight')
    plots["potentials"].savefig(f"{res_path}/potentials.png", dpi=300, bbox_inches='tight')
    plots["biased_potentials"].savefig(f"{res_path}/biased_potentials.png", dpi=300, bbox_inches='tight')
    plots["total_potentials"].savefig(f"{res_path}/total_potentials.png", dpi=300, bbox_inches='tight')
    plots["cvs"].savefig(f"{res_path}/cvs.png", dpi=300, bbox_inches='tight')
    logger.info(f">> Paths plot saved at: {f'{res_path}/paths.png'}")
    logger.info(f">> Potentials plot saved at: {f'{res_path}/potentials.png'}")
    logger.info(f">> Biased potentials plot saved at: {f'{res_path}/biased_potentials.png'}")
    logger.info(f">> Total potentials plot saved at: {f'{res_path}/total_potentials.png'}")
    logger.info(f">> Cvs plot saved at: {f'{res_path}/cvs.png'}")
    logger.info(f">> Metrics saved at: {f'{res_path}/metrics.json'}")
    logger.info(f">> SMD result: {metrics}")
    logger.info(">> Finish evaluation")

    if cfg.logger.enable:
        wandb.log(metrics)
        wandb.log(
            {
                "paths": wandb.Image(f"{res_path}/paths.png"),
                "potentials": wandb.Image(f"{res_path}/potentials.png"),
                "biased_potentials": wandb.Image(f"{res_path}/biased_potentials.png"),
                "total_potentials": wandb.Image(f"{res_path}/total_potentials.png"),
                "cvs": wandb.Image(f"{res_path}/cvs.png")
            }
        )
        wandb.finish()


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()