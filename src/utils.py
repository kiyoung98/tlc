import torch


def kabsch_rmsd(P, Q):
    centroid_P = torch.mean(P, dim=-2, keepdims=True)
    centroid_Q = torch.mean(Q, dim=-2, keepdims=True)
    p = P - centroid_P
    q = Q - centroid_Q
    with torch.no_grad():
        H = torch.matmul(p.transpose(-2, -1), q)
        U, S, Vt = torch.linalg.svd(H)
        d = torch.det(torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1)))
        Vt[d < 0.0, -1] *= -1.0
        R = torch.matmul(Vt.transpose(-2, -1), U.transpose(-2, -1))
    p_rotated = torch.matmul(p, R)
    diff = p_rotated - q
    rmsd = diff.square().sum(-1).mean(-1).sqrt()
    return rmsd