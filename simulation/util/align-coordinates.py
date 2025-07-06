import os
import torch

import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset

from mlcolvar.data import DictDataset, DictModule


ALANINE_BACKBONE_ATOM_IDX = [1, 4, 6, 8, 10, 14, 16, 18]


class CL_dataset(Dataset):
    def __init__(
        self,
        data_list,
        data_augmented_list,
        data_augmented_hard_list,
        temperature_list,
    ):
        super(CL_dataset, self).__init__()
        self.x = data_list
        self.x_augmented = data_augmented_list
        self.x_augmented_hard = data_augmented_hard_list
        self.temperature = temperature_list
        
    def __getitem__(self, index):
	    return self.x[index], self.x_augmented[index], self.x_augmented_hard[index], self.temperature[index]
 
    def __len__(self):
	    return self.x.shape[0]
 

def kabsch(
	reference_position: torch.Tensor,
	position: torch.Tensor,
) -> torch.Tensor:
    '''
        Kabsch algorithm for aligning two sets of points
        Args:
            reference_position (torch.Tensor): Reference positions (N, 3)
            position (torch.Tensor): Positions to align (N, 3)
        Returns:
            torch.Tensor: Aligned positions (N, 3)
    '''
    # Compute centroids
    centroid_ref = torch.mean(reference_position, dim=0, keepdim=True)
    centroid_pos = torch.mean(position, dim=0, keepdim=True)
    ref_centered = reference_position - centroid_ref  
    pos_centered = position - centroid_pos

    # Compute rotation, translation matrix
    covariance = torch.matmul(ref_centered.T, pos_centered)
    U, S, Vt = torch.linalg.svd(covariance)
    d = torch.linalg.det(torch.matmul(Vt.T, U.T))
    if d < 0:
        Vt[-1] *= -1
    rotation = torch.matmul(Vt.T, U.T)

    # Align position to reference_position
    aligned_position = torch.matmul(pos_centered, rotation) + centroid_ref
    return aligned_position




custom_dataset = torch.load(f"../dataset/alanine/300.0/10n-v1/cl-xyz.pt")

custom_data = torch.cat([
    custom_dataset.x,
    custom_dataset.x_augmented,
    custom_dataset.x_augmented_hard,
], dim=0).reshape(-1, 22, 3)
reference_frame = custom_data[0]
data_num = custom_data.shape[0]

for i in tqdm(range(data_num)):
    custom_data[i] = kabsch(reference_frame, custom_data[i])
print(custom_data.shape)

path_to_save = f"../dataset/alanine/300.0/10n-v1/cl-xyz-aligned.pt"
if not os.path.exists(path_to_save):
    torch.save(custom_data, path_to_save)
