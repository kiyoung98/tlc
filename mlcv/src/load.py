import os
import torch
import wandb
import numpy as np

from torch import optim
from omegaconf import OmegaConf
from itertools import combinations

from mlcolvar.data import DictDataset, DictModule
from mlcolvar.cvs import DeepLDA, DeepTDA, DeepTICA

from .util.rotate import kabsch
from .util.constant import *
from .data import *
from .model import *


model_dict = {
    "lda": LDA,
    "dihedral": DeepLDA,
    "deeplda": DeepLDA,
    "deeptda": DeepTDA,
    "deeptica": DeepTICA,
    "tae": TAE,
    "tae-xyz": TAE,
    "vae": VAE,
    "vde": VariationalDynamicsEncoder,
    "tbgcv": TBGCV,
    "tbgcv-xyz": TBGCV,
    "tbgcv-nolag": TBGCV,
    "tbgcv-xyzhad": TBGCV,
}


def load_model(cfg):
    if cfg.model.name in ["dihedral", "rmsd"]:
        model = model_dict[cfg.model.name](**cfg.model.model)

    elif cfg.model.name == "gnncv-tica":
        import mlcolvar.graph as mg
        mg.utils.torch_tools.set_default_dtype('float32')
        optimizer_options = {
            'optimizer': cfg.model.trainer.optimizer.optimizer,
            'lr_scheduler': {
                'scheduler': optim.lr_scheduler.ExponentialLR,
                'gamma': cfg.model.trainer.optimizer.lr_scheduler.gamma
            }
        }
        model = mg.cvs.GraphDeepTICA(
            n_cvs=cfg.n_cvs,
            cutoff=cfg.cutoff,
            atomic_numbers=cfg.atomic_numbers,
            model_options=dict(cfg.model.model),
            optimizer_options=optimizer_options,
        )
    
    elif cfg.model.name in model_dict:
        model = model_dict[cfg.model.name](**cfg.model.model)
    
    else:
        raise ValueError(f"Model {cfg.model.name} not found")
    
    print(">> Model")
    print(model)
    return model


def load_lightning_logger(cfg):
    if cfg.model.logger.name == "wandb":
        from lightning.pytorch.loggers import WandbLogger
        wandb.init(
            project = cfg.model.logger.project,
            entity = "eddy26",
            tags = cfg.model.logger.tags,
            config = OmegaConf.to_container(
                cfg, resolve=True, throw_on_missing=True
            )
        )
        lightning_logger = WandbLogger(
            project = cfg.model.logger.project,
            log_model = cfg.model.logger.log_model
        )
    else:
        lightning_logger = None
        
    return lightning_logger


def load_data(cfg):
    if cfg.model.name == "dihedral":
        return
    
    data_dir = os.path.join(
        cfg.data.dir,
        cfg.data.molecule,
        str(cfg.data.temperature),
        cfg.data.version
    )
    
    if cfg.model.name in ["lda", "deeplda", "deeptda"]:
        custom_data = torch.load(os.path.join(data_dir, "current-distance.pt"))
        custom_label = torch.load(os.path.join(data_dir, "current-label.pt"))
        dataset = DictDataset(
            {
                "data": custom_data,
                "labels": custom_label
            },
            feature_names = np.array([f"{a} - {b}" for a, b in combinations(ALANINE_HEAVY_ATOM_DESCRIPTOR, 2)], dtype=str)  
        )
        datamodule = DictModule(
            dataset = dataset,
            lengths = [0.8,0.2],
            batch_size=cfg.model.trainer.batch_size
        )
        
    elif cfg.model.name in ["deeptica", "tae-xyz", "vde"]:
        custom_data = torch.load(os.path.join(data_dir, "current-distance.pt"))
        custom_data_lag = torch.load(os.path.join(data_dir, "timelag-distance.pt"))
        dataset = DictDataset(
            {
                "data": custom_data,
                "data_lag": custom_data_lag,
                "weights": torch.ones(custom_data.shape[0], dtype=torch.float32, device=custom_data.device),
                "weights_lag": torch.ones(custom_data_lag.shape[0], dtype=torch.float32, device=custom_data_lag.device)
            },
            feature_names = np.array([f"{a} - {b}" for a, b in combinations(ALANINE_HEAVY_ATOM_DESCRIPTOR, 2)], dtype=str)  
        )
        datamodule = DictModule(
            dataset = dataset,
            lengths = [0.8,0.2],
            batch_size=cfg.model.trainer.batch_size
        )
    
    elif cfg.model.name in ["tae", "vae"]:
        custom_data = torch.load(os.path.join(data_dir, "current-xyz-aligned.pt"))
        custom_data_lag = torch.load(os.path.join(data_dir, "timelag-xyz-aligned.pt"))
        heavy_atom_data = custom_data[:, ALANINE_HEAVY_ATOM_IDX]
        heavy_atom_data_lag = custom_data_lag[:, ALANINE_HEAVY_ATOM_IDX]
        heavy_atom_data_lag.requires_grad = True
        dataset = DictDataset(
            {
                "data": heavy_atom_data.reshape(heavy_atom_data.shape[0], -1),
                "target": heavy_atom_data_lag.reshape(heavy_atom_data.shape[0], -1),
            },
            feature_names = np.array([f"{a}" for a in ALANINE_HEAVY_ATOM_DESCRIPTOR], dtype=str)
        )
        datamodule = DictModule(
            dataset = dataset,
            lengths = [0.8,0.2],
            batch_size=cfg.model.trainer.batch_size
        )
        
    elif cfg.model.name in "tbgcv":
        custom_data = torch.load(os.path.join(data_dir, "current-distance.pt"))
        custom_data_lag = torch.load(os.path.join(data_dir, "timelag-distance.pt"))
        dataset = DictDataset(
            {
                "data": custom_data.reshape(custom_data.shape[0], -1),
                "data_lag": custom_data_lag.reshape(custom_data.shape[0], -1),
            },
            feature_names = np.array([f"{a} - {b}" for a, b in combinations(ALANINE_HEAVY_ATOM_DESCRIPTOR, 2)], dtype=str)
        )
        datamodule = DictModule(
            dataset = dataset,
            lengths = [0.8,0.2],
            batch_size=cfg.model.trainer.batch_size
        )
        
    elif cfg.model.name == "tbgcv-xyz":
        custom_data = torch.load(os.path.join(data_dir, "current-xyz-aligned.pt"))
        custom_data_lag = torch.load(os.path.join(data_dir, "timelag-distance.pt"))
        heavy_atom_data = custom_data[:, ALANINE_HEAVY_ATOM_IDX]
        dataset = DictDataset(
            {
                "data": heavy_atom_data.reshape(heavy_atom_data.shape[0], -1),
                "data_lag": custom_data_lag.reshape(custom_data.shape[0], -1),
            },
            feature_names = np.array([f"{a}" for a in ALANINE_HEAVY_ATOM_DESCRIPTOR], dtype=str)
        )
        datamodule = DictModule(
            dataset = dataset,
            lengths = [0.8,0.2],
            batch_size=cfg.model.trainer.batch_size
        )
    
    elif cfg.model.name == "tbgcv-nolag":
        custom_data = torch.load(os.path.join(data_dir, "current-xyz-aligned.pt"))
        custom_data_lag = torch.load(os.path.join(data_dir, "timelag-xyz.pt"))
        heavy_atom_data = custom_data[:, ALANINE_HEAVY_ATOM_IDX]
        dataset = DictDataset(
            {
                "data": heavy_atom_data.reshape(heavy_atom_data.shape[0], -1),
                "data_lag": custom_data_lag.reshape(custom_data.shape[0], -1),
            },
            feature_names = np.array([f"{a}" for a in ALANINE_HEAVY_ATOM_DESCRIPTOR], dtype=str)
        )
        datamodule = DictModule(
            dataset = dataset,
            lengths = [0.8,0.2],
            batch_size=cfg.model.trainer.batch_size
        )
    
    elif cfg.model.name == "tbgcv-xyzhad":
        custom_data_xyz = torch.load(os.path.join(data_dir, "current-xyz-aligned.pt"))
        heavy_atom_data = custom_data_xyz[:, ALANINE_HEAVY_ATOM_IDX].reshape(custom_data_xyz.shape[0], -1)
        custom_data_had = torch.load(os.path.join(data_dir, "current-distance.pt"))
        custom_data_lag = torch.load(os.path.join(data_dir, "timelag-xyz.pt"))
        custom_data_input = torch.cat([heavy_atom_data, custom_data_had], dim=1)
        dataset = DictDataset(
            {
                "data": custom_data_input,
                "data_lag": custom_data_lag.reshape(custom_data_lag.shape[0], -1),
            },
            feature_names = np.concatenate((
                np.repeat(np.array([f"{a}" for a in ALANINE_HEAVY_ATOM_DESCRIPTOR], dtype=str), 3),
                np.array([f"{a} - {b}" for a, b in combinations(ALANINE_HEAVY_ATOM_DESCRIPTOR, 2)], dtype=str)
            ))
        )
        datamodule = DictModule(
            dataset = dataset,
            lengths = [0.8,0.2],
            batch_size=cfg.model.trainer.batch_size
        )
        
    elif cfg.model.name == "gnncv-tica":
        import mlcolvar.graph as mg
        mg.utils.torch_tools.set_default_dtype('float32')
        graph_dataset = torch.load(os.path.join(data_dir, "graph-dataset.pt"))
        datasets = mg.utils.timelagged.create_timelagged_datasets(
            graph_dataset, lag_time=2
        )
        datamodule = mg.data.GraphCombinedDataModule(
            datasets,
            random_split = False,
            batch_size = cfg.model.trainer.batch_size
        )
    
    else:
        raise ValueError(f"Data not found for model {cfg.model.name}")
    
    print(">> Dataset")
    print(datamodule)
    return datamodule


def load_checkpoint(cfg, model):
    checkpoint_path = f"./model/{cfg.model.name}/{cfg.data.version}"
    
    if not "checkpoint" in cfg.model or cfg.model["checkpoint"]:
        raise ValueError(f"Checkpoint path disabled, check config")
    
    if cfg.model.name in ["deeplda", "deeptda"]:
        model.load_state_dict(torch.load(checkpoint_path))
    
    else:
        raise ValueError(f"Checkpoint not found for model {cfg.model.name}")
    
    return model


def load_projection_data(cfg, device):
    representation = cfg.model.representation
    
    if representation in ['heavy_atom_distance']:
        dataset = torch.load(f"../simulation/data/alanine/uniform-heavy-atom-distance.pt").to(device)
        if cfg.model.name == "tbgcv":
            dataset = dataset / ALANINE_TBGCV_SCALING
        
    elif representation == 'heavy_atom_coordinate':
        dataset = torch.load(f"../simulation/data/alanine/uniform-aligned.pt").to(device)
        dataset = dataset[:, ALANINE_HEAVY_ATOM_IDX].reshape(dataset.shape[0], -1)
        
    elif representation == 'heavy_atom_coordinate_distance':
        dataset_xyz = torch.load(f"../simulation/data/alanine/uniform-aligned.pt").to(device)[:, ALANINE_HEAVY_ATOM_IDX]
        dataset_xyz = dataset_xyz.reshape(dataset_xyz.shape[0], -1)
        dataset_had = torch.load(f"../simulation/data/alanine/uniform-heavy-atom-distance.pt").to(device)
        dataset = torch.cat([dataset_xyz, dataset_had], dim=1)
        
    elif representation == 'backbone_atom_coordinate':
        dataset = torch.load(f"../simulation/data/alanine/uniform.pt").to(device)
        dataset = dataset[:, ALANINE_BACKBONE_ATOM_IDX].reshape(dataset.shape[0], -1)
        
    else:
        raise ValueError(f"Representation {representation} not found")    
    
    return dataset
