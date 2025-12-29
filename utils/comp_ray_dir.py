import torch
import numpy as np


def get_rays(H, W, focal, c2w, device='cuda'):
    """
    PyTorch version of NeRF-style ray generator.
   
    Args:
        H (int): Image height
        W (int): Image width
        focal (float): Focal length
        c2w (torch.Tensor): Camera-to-world matrix (3x4 or 4x4)
        device (str): 'cpu' or 'cuda'
   
    Returns:
        rays_o: (H, W, 3) tensor of ray origins
        rays_d: (H, W, 3) tensor of ray directions
    """
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing='xy'
    )
 
    dirs = torch.stack([
        (i - W * 0.5) / focal, 
        -(j - H * 0.5) / focal, 
        -torch.ones(H, W, dtype=torch.float32, device=device)  # (H, W)

    ], dim=-1)  # (H, W, 3)
 
    if c2w.shape == torch.Size([3, 4]):
        R = c2w[:, :3].to(device)
        t = c2w[:, 3].to(device)
    elif c2w.shape == torch.Size([4, 4]):
        R = c2w[:3, :3].to(device)
        t = c2w[:3, 3].to(device)
    else:
        raise ValueError("c2w must be of shape (3, 4) or (4, 4)")

    
    rays_d = dirs @ R.T
    #rays_d =  dirs
 
    # Broadcast camera origin to all rays
    rays_o = t.expand_as(rays_d)
 
    return rays_o, rays_d