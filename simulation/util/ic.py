import numpy as np
import bgflow as bg

import torch

def load_ic_transform():
    z_matrix = np.array([
        [ 0,  1,  4,  6],
        [ 1,  4,  6,  8],
        [ 2,  1,  4,  0],
        [ 3,  1,  4,  0],
        [ 4,  6,  8, 14],
        [ 5,  4,  6,  8],
        [ 7,  6,  8,  4],
        [11, 10,  8,  6],
        [12, 10,  8, 11],
        [13, 10,  8, 11],
        [15, 14,  8, 16],
        [16, 14,  8,  6],
        [17, 16, 14, 15],
        [18, 16, 14,  8],
        [19, 18, 16, 14],
        [20, 18, 16, 19],
        [21, 18, 16, 19]
    ])
    rigid_block = np.array([ 6,  8,  9, 10, 14])
    
    coordinate_transform = bg.RelativeInternalCoordinateTransformation(
        z_matrix=z_matrix,
        fixed_atoms=rigid_block,
        normalize_angles = False,
        eps = 1e-10, 
    )
    
    def xyz2ic(xyz):
        bond, torsion, dihedral, fixed_x, _ = coordinate_transform(xyz)
        ic = torch.cat([bond, torsion, dihedral, fixed_x], dim=1)
        return ic
        
    return xyz2ic