import os
import pytz
import hydra
import torch
import logging
import lightning

from datetime import datetime
from omegaconf import OmegaConf,open_dict
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import Trainer

from mlcolvar.utils.trainer import MetricsCallback
from mlcolvar.core.transform import Statistics

from src import *


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="basic"
)
def main(cfg):
    # Load configs and components
    lightning_logger = load_lightning_logger(cfg)
    logger = logging.getLogger("MLCVs")
    logger.info(">> Configs")
    logger.info(OmegaConf.to_yaml(cfg))
    device = "cuda"
    model = load_model(cfg).to(device)
    datamodule = load_data(cfg)
    checkpoint_path = f"./res/{cfg.model.name}/{cfg.data.version}"
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    if not os.path.exists(checkpoint_path + f"/{cfg.steeredmd.simulation.k}/SteeredMD"):
        os.makedirs(checkpoint_path + f"/{cfg.steeredmd.simulation.k}/SteeredMD")
    
    # Load from checkpoint
    if cfg.model.checkpoint and cfg.model.name not in ["dihedral"]:
        logger.info(f">> Loading model from checkpoint {cfg.model.checkpoint_name}.pt")
        if cfg.model.name not in ["tbgcv", "tbgcv-xyz", "tbgcv-nolag", "tbgcv-xyzhad"]:
            model.postprocessing = PostProcess().to(model.device)
        model.load_state_dict(torch.load(f"{checkpoint_path}/{cfg.model.checkpoint_name}.pt"))
    elif cfg.model.checkpoint and cfg.model.name in ["dihedral"]:
        pass
    # or train
    else:
        logger.info(">> Training...")
        metrics = MetricsCallback()
        early_stopping = EarlyStopping(**cfg.model.trainer.early_stopping)
        trainer = lightning.Trainer(
            callbacks=[metrics, early_stopping],
            logger=lightning_logger,
            max_epochs=cfg.model.trainer.max_epochs,
            enable_checkpointing=False,
            log_every_n_steps=1,
        )
        trainer.fit(model, datamodule)
        logger.info(">> Training complete.!!")
    model.eval()
    
    # Load projection data
    if cfg.model.name in ["dihedral"]:
        pass
    else:
        projection_dataset = load_projection_data(cfg, model.device)
        with torch.no_grad():
            cv = model(projection_dataset)
        stats = Statistics(cv.cpu()).to_dict()
        wandb.log({f"cv/{key}":item for key, item in stats.items()})
    
    # Normalization on MLCV values it posprocessing module not configuredd
    if not cfg.model.checkpoint or cfg.model.name in ["tbgcv", "tbgcv-xyz", "tbgcv-nolag", "tbgcv-xyzhad"]:
        c5_state = torch.load(f"../simulation/data/{cfg.data.molecule}/{cfg.steeredmd.start_state}.pt")['xyz'].to(model.device)
        model.postprocessing = PostProcess(stats, model(input2representation(cfg, c5_state, c5_state[0]))).to(model.device)
        logger.info(f">> Post-processing module added")
        cv_normalized = model.postprocessing(cv.to(model.device))
        logger.info(f">> Max CV: {cv_normalized.max()}, Min CV: {cv_normalized.min()}")
        
        
    # Basic plots
    if cfg.model.name in ["dihedral"]:
        pass
    else:
        cv_scatter_plot = plot_ram_scatter_cv(
            cfg,
            model,
            logger,
            checkpoint_path + f"/{cfg.steeredmd.simulation.k}",
            projection_dataset
        )
        cv_hexplot = plot_ram_hex_cv(
            cfg,
            model,
            logger,
            checkpoint_path + f"/{cfg.steeredmd.simulation.k}",
            projection_dataset
        )
        sensitivity = plot_sensitivity(
            cfg,
            model,
            logger,
            checkpoint_path + f"/{cfg.steeredmd.simulation.k}",
            projection_dataset
        )
        wandb.log({
            "plot/cv-plot": wandb.Image(cv_scatter_plot),
            "plot/cv-hexplot": wandb.Image(cv_hexplot),
            "plot/sensitivity": wandb.Image(sensitivity),
        })

    # Save model if trained
    if not cfg.model.checkpoint:
        if not "checkpoint_name" in cfg.model:
            kst = pytz.timezone('Asia/Seoul')
            now = datetime.now(kst)
            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                cfg.model.checkpoint_name = now.strftime("%m%d_%H%M%S")
        torch.save(model.state_dict(), checkpoint_path + f"/{cfg.model.checkpoint_name}.pt")
    
    # Save jit model (always)
    if cfg.model.name in ["dihedral"]:
        pass
    else:
        model.trainer = Trainer(logger=False, enable_checkpointing=False, enable_model_summary=False)
        input_dim = datamodule.dataset["data"].shape[1]
        random_input = torch.rand(1, input_dim).to(model.device)
        traced_script_module = torch.jit.trace(model, random_input)
        traced_script_module.save(checkpoint_path + f"/{cfg.model.checkpoint_name}-jit.pt")
        logger.info(f">> Model saved at {checkpoint_path}/{cfg.model.checkpoint_name}-jit.pt")
    
    # Evaluation
    logger.info(">> Evaluating...")
    eval(
        cfg = cfg,
        model = model,
        logger = logger,
        checkpoint_path = checkpoint_path
    )
    wandb.finish()
    logger.info(">> Evaluation complete.!!")


if __name__ == "__main__":
    torch.set_float32_matmul_precision('high')
    main()