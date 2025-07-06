import torch

from .constant import *
from .rotate import kabsch
from mlcolvar.core.transform.descriptors import PairwiseDistances


def input2representation(cfg, input, reference_frame=None):
    if cfg.model.representation == "heavy_atom_distance":
        # representation_list = torch.empty(size=(input.shape[0], ALANINE_HEAVY_ATOM_NUM * (ALANINE_HEAVY_ATOM_NUM - 1) // 2), device=input.device, dtype=input.dtype)
        # for idx, data in enumerate(input):
        #     if cfg.model.name in ["tbgcv"]:
        #         data = data / ALANINE_TBGCV_SCALING
        #     representation_list[idx] = coordinate2distance(data)
        cell = torch.tensor([25.58, 25.58, 25.58], device=input.device, dtype=input.dtype)
        ComputeDistances = PairwiseDistances(
            n_atoms=ALANINE_HEAVY_ATOM_NUM,
            PBC=True,
            cell=cell,
            scaled_coords=False
        )        
        representation = ComputeDistances(input[:, ALANINE_HEAVY_ATOM_IDX])
    
    elif cfg.model.representation == "heavy_atom_coordinate":
        input = input.reshape(input.shape[0], -1)
        representation = torch.empty(size=(input.shape[0], ALANINE_HEAVY_ATOM_NUM * 3), device=input.device, dtype=input.dtype)
        for idx, data in enumerate(input):
            if reference_frame is not None:
                data = kabsch(reference_frame, data.reshape(ALANINE_ATOM_NUM, 3))
            else:
                raise ValueError(f"Reference frame {reference_frame} not found")
            representation[idx]  = data[ALANINE_HEAVY_ATOM_IDX].reshape(1, -1)
    
    elif cfg.model.representation == "heavy_atom_coordinate_distance":
        if reference_frame is None:
            raise ValueError(f"Reference frame {reference_frame} not found")
        input = input.reshape(input.shape[0], -1)
        representation = torch.empty(size=(input.shape[0], ALANINE_HEAVY_ATOM_NUM * 3 + ALANINE_HEAVY_ATOM_NUM * (ALANINE_HEAVY_ATOM_NUM - 1) // 2), device=input.device, dtype=input.dtype)
        for idx, data in enumerate(input):
            data_xyz = kabsch(reference_frame, data.reshape(ALANINE_ATOM_NUM, 3))[ALANINE_HEAVY_ATOM_IDX].reshape(1, -1)
            data_had = coordinate2distance(data).reshape(1, -1)
            representation[idx] = torch.cat([data_xyz, data_had], dim=1)
    
    elif cfg.model.representation == "backbone":
        raise NotImplementedError(f"Representation {cfg.model.representation} not implemented")
        
    else:
        raise ValueError(f"Representation {cfg.model.representation} not found")    
    
    return representation


def coordinate2distance(position):
    position = position.reshape(-1, 3)
    heavy_atom_position = position[ALANINE_HEAVY_ATOM_IDX]
    num_heavy_atoms = len(heavy_atom_position)
    distance = []
    for i in range(num_heavy_atoms):
        for j in range(i+1, num_heavy_atoms):
            distance.append(torch.norm(heavy_atom_position[i] - heavy_atom_position[j]))
    distance = torch.stack(distance)
    
    return distance