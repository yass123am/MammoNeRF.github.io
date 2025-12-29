import sys
import os
import argparse
from pathlib import Path
import datetime
import shutil
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import cv2
sys.path.append(os.path.join(sys.path[0], '../..'))
import imageio

from dataloader.any_folder import DataLoaderAnyFolder
from utils.training_utils import set_randomness, mse2psnr, save_checkpoint
from utils.pos_enc import encode_position
from utils.volume_op import volume_rendering , volume_sampling_ndc_before, volume_sampling_ndc
from utils.comp_ray_dir import get_rays
from models.nerf_models import OfficialNerf
from models.intrinsics import LearnFocal
from models.poses import LearnPose
import open3d as o3d  # pip install open3d
import torch
from scipy.spatial import cKDTree

#############
from PIL import Image
######################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=4000, type=int)
    parser.add_argument('--eval_interval', default=100, type=int, help='run eval every this epoch number')

    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--multi_gpu',  default=False, type=eval, choices=[True, False])
    parser.add_argument('--base_dir', type=str, default='./any_folder_demo')
    parser.add_argument('--scene_name', type=str, default='base')

    parser.add_argument('--nerf_lr', default=0.001, type=float)
    parser.add_argument('--nerf_milestones', default=list(range(0, 10000, 10)), type=int, nargs='+',
                        help='learning rate schedule milestones')
    parser.add_argument('--nerf_lr_gamma', type=float, default=0.9954, help="learning rate milestones gamma")

    parser.add_argument('--learn_focal', default=False, type=bool)
    parser.add_argument('--focal_order', default=2, type=int)
    parser.add_argument('--fx_only', default=True, type=eval, choices=[True, False])
    parser.add_argument('--focal_lr', default=0.001, type=float)
    parser.add_argument('--focal_milestones', default=list(range(0, 10000, 100)), type=int, nargs='+',
                        help='learning rate schedule milestones')
    parser.add_argument('--focal_lr_gamma', type=float, default=0.9, help="learning rate milestones gamma")

    parser.add_argument('--learn_R', default=True, type=eval, choices=[True, False])
    parser.add_argument('--learn_t', default=True, type=eval, choices=[True, False])
    parser.add_argument('--pose_lr', default=0.001, type=float)
    parser.add_argument('--pose_milestones', default=list(range(0, 10000, 100)), type=int, nargs='+',
                        help='learning rate schedule milestones')
    parser.add_argument('--pose_lr_gamma', type=float, default=0.9, help="learning rate milestones gamma")

    parser.add_argument('--resize_ratio', type=int, default=16, help='lower the image resolution with this ratio')
    parser.add_argument('--num_rows_eval_img', type=int, default=1, help='split a high res image to rows in eval')
    parser.add_argument('--hidden_dims', type=int, default=128, help='network hidden unit dimensions')
    parser.add_argument('--train_rand_rows', type=int, default=32, help='rand sample these rows to train')
    parser.add_argument('--train_rand_cols', type=int, default=32, help='rand sample these cols to train')
    parser.add_argument('--num_sample', type=int, default=2, help='number samples along a ray')

    parser.add_argument('--pos_enc_levels', type=int, default=10, help='number of freqs for positional encoding')
    parser.add_argument('--pos_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--use_dir_enc', type=bool, default=True, help='use pos enc for view dir?')
    parser.add_argument('--dir_enc_levels', type=int, default=4, help='number of freqs for positional encoding')
    parser.add_argument('--dir_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--train_img_num', type=int, default=-1, help='num of images to train, -1 for all')
    parser.add_argument('--train_load_sorted', type=bool, default=True)
    parser.add_argument('--train_start', type=int, default=0, help='inclusive')
    parser.add_argument('--train_end', type=int, default=-1, help='exclusive, -1 for all')
    parser.add_argument('--train_skip', type=int, default=1, help='skip every this number of imgs')

    parser.add_argument('--rand_seed', type=int, default=17)
    parser.add_argument('--true_rand', type=bool, default=False)

    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')

    parser.add_argument('--alias', type=str, default='', help="experiments alias")
    return parser.parse_args()


def gen_detail_name(args):
    outstr = 'lr_' + str(args.nerf_lr) + \
             '_gpu' + str(args.gpu_id) + \
             '_seed_' + str(args.rand_seed) + \
             '_resize_' + str(args.resize_ratio) + \
             '_Nsam_' + str(args.num_sample) + \
             '_Ntr_img_'+ str(args.train_img_num) + \
             '_freq_' + str(args.pos_enc_levels) + \
             '_' + str(args.alias) + \
             '_' + str(datetime.datetime.now().strftime('%y%m%d_%H%M'))
    return outstr


def model_render_image(c2w, rays_orig , rays_cam, t_vals, near, far, H, W, fxfy, model, perturb_t, sigma_noise_std,
                       args, rgb_act_fn):
    """Render an image or pixels.
    :param c2w:         (4, 4)                  pose to transform ray direction from cam to world.
    :param rays_cam:    (someH, someW, 3)       ray directions in camera coordinate, can be random selected
                                                rows and cols, or some full rows, or an entire image.
    :param t_vals:      (N_samples)             sample depth along a ray.
    :param fxfy:        a float or a (2, ) torch tensor for focal.
    :param perturb_t:   True/False              whether add noise to t.
    :param sigma_noise_std: a float             std dev when adding noise to raw density (sigma).
    :rgb_act_fn:        sigmoid()               apply an activation fn to the raw rgb output to get actual rgb.
    :return:            (someH, someW, 3)       volume rendered images for the input rays.
    """
    # (H, W, N_sample, 3), (H, W, 3), (H, W, N_sam)
    sample_pos, ray_ori_world, ray_dir_world, t_vals_noisy = volume_sampling_ndc_before(c2w, rays_orig, rays_cam, t_vals, near, far,
                                                                     H, W , fxfy, perturb_t)


 
    # encode position: (H, W, N_sample, (2L+1)*C = 63)
    pos_enc = encode_position(sample_pos, levels=args.pos_enc_levels, inc_input=args.pos_enc_inc_in)
    # encode direction: (H, W, N_sample, (2L+1)*C = 27)
    if args.use_dir_enc:
        ray_dir_world = F.normalize(ray_dir_world, p=2, dim=2)  # (H, W, 3)
        dir_enc = encode_position(ray_dir_world, levels=args.dir_enc_levels, inc_input=args.dir_enc_inc_in)  # (H, W, 27)
        dir_enc = dir_enc.unsqueeze(2).expand(-1, -1, args.num_sample, -1)  # (H, W, N_sample, 27)
    else:
        dir_enc = None

    # inference rgb and density using position and direction encoding.
    rgb_density = model(pos_enc, dir_enc)  # (H, W, N_sample, 4)

    #volume(sample_pos, rgb_density)
    #density(sample_pos, rgb_density)
    
    
    render_result = volume_rendering(rgb_density, t_vals_noisy, sigma_noise_std, rgb_act_fn)
    rgb_rendered = render_result['rgb']  # (H, W, 3)
    depth_map = render_result['depth_map']  # (H, W)

    result = {
        'rgb': rgb_rendered,  # (H, W, 3)
        'sample_pos': sample_pos,  # (H, W, N_sample, 3)
        'depth_map': depth_map,  # (H, W)
        'rgb_density': rgb_density,  # (H, W, N_sample, 4)
        'encoded_poses' : pos_enc, # (H, W, N_sample, 27)
        'ray_direction' : ray_dir_world,
        'ray_origine' : ray_ori_world,
    }

    return result



def volume_mlo(sample_pos, rgb_density, output_path='sample_points_mlo_nn.ply'):
    """
    Visualize the 3D points with their associated RGB color and save them in .ply format without flattening the arrays.
    
    :param sample_pos: (H, W, N_sample, 3) torch.Tensor
    :param rgb_density: (H, W, N_sample, 4) torch.Tensor, the first three channels are RGB
    """
    with torch.no_grad():
        # Convert to numpy arrays
        pts = sample_pos.detach().cpu().numpy()  # (H, W, N_sample, 3)
        rgb = rgb_density[..., :3].detach().cpu().numpy()  # (H, W, N_sample, 3)

        # Apply sigmoid activation if not done already to ensure RGB is in [0,1]
        rgb = 1 / (1 + np.exp(-rgb))  # Ensure values are between [0, 1]
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

        # Initialize lists for points and colors

        all_points = []
        all_colors = []

        # Iterate through the grid to collect points and their corresponding RGB values
        H, W, N_sample, _ = pts.shape
        for h in range(H):
            for w in range(W):
                for n in range(N_sample):
                    all_points.append(pts[h, w, n])
                    all_colors.append(rgb[h, w, n])


        # Convert the list of points and colors to numpy arrays
        all_points = np.array(all_points)
        all_colors = np.array(all_colors)

        # Create an Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(np.float32) / 255.0)

        # Save the point cloud to file
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved point cloud with {all_points.shape[0]} points to {output_path}")


def volume_cc(sample_pos, rgb_density, output_path='sample_points_cc_nn.ply'):
    """
    Visualize the 3D points with their associated RGB color and save them in .ply format.
    Rotates the volume 90 degrees around the Z axis.
    
    :param sample_pos: (H, W, N_sample, 3) torch.Tensor
    :param rgb_density: (H, W, N_sample, 4) torch.Tensor, the first three channels are RGB
    """
    with torch.no_grad():
        # Convert to numpy arrays
        pts = sample_pos.detach().cpu().numpy()  # (H, W, N_sample, 3)
        rgb = rgb_density[..., :3].detach().cpu().numpy()  # (H, W, N_sample, 3)

        # Apply sigmoid to ensure RGB values are in [0,1]
        rgb = 1 / (1 + np.exp(-rgb))
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

        # Initialize lists for points and colors
        all_points = []
        all_colors = []

        H, W, N_sample, _ = pts.shape
        for h in range(H):
            for w in range(W):
                for n in range(N_sample):
                    all_points.append(pts[h, w, n])
                    all_colors.append(rgb[h, w, n])

        all_points = np.array(all_points)
        all_colors = np.array(all_colors)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(np.float32) / 255.0)

        # Save to file
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved rotated point cloud with {all_points.shape[0]} points to {output_path}")


def save_rgb_volume_as_pointcloud(rgb_density, output_path='sample_points_cc_nn.ply'):
    """
    Convert an RGB tensor to a visible 3D grid point cloud and save as .ply.
    :param rgb_density: (H, W, N_sample, 4) torch.Tensor
    """
    with torch.no_grad():
        # Get RGB and apply sigmoid
        rgb = rgb_density[..., :3].detach().cpu().numpy()  # (H, W, N_sample, 3)
        rgb = 1 / (1 + np.exp(-rgb))  # sigmoid to [0, 1]
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)  # to [0, 255]

        H, W, N_sample, _ = rgb.shape
        points = []
        colors = []

        for h in range(H):
            for w in range(W):
                for n in range(N_sample):
                    # Assign 3D position in voxel grid: (w, h, n)
                    points.append([w, h, n])
                    colors.append(rgb[h, w, n])

        points = np.array(points, dtype=np.float32)
        colors = np.array(colors, dtype=np.float32) / 255.0  # normalize for Open3D

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Save to file
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved RGB volume with {points.shape[0]} points to '{output_path}'")


def volume_z_1(sample_pos, rgb_density, output_path='sample_points_z_1_nn.ply'):
    """
    Visualize the 3D points with their associated RGB color and save them in .ply format.
    Rotates the volume 90 degrees around the Z axis.
    
    :param sample_pos: (H, W, N_sample, 3) torch.Tensor
    :param rgb_density: (H, W, N_sample, 4) torch.Tensor, the first three channels are RGB
    """
    with torch.no_grad():
        # Convert to numpy arrays
        pts = sample_pos.detach().cpu().numpy()  # (H, W, N_sample, 3)
        rgb = rgb_density[..., :3].detach().cpu().numpy()  # (H, W, N_sample, 3)

        # Apply sigmoid to ensure RGB values are in [0,1]
        rgb = 1 / (1 + np.exp(-rgb))
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

        # Initialize lists for points and colors
        all_points = []
        all_colors = []

        H, W, N_sample, _ = pts.shape
        for h in range(H):
            for w in range(W):
                for n in range(N_sample):
                    all_points.append(pts[h, w, n])
                    all_colors.append(rgb[h, w, n])

        all_points = np.array(all_points)
        all_colors = np.array(all_colors)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(np.float32) / 255.0)

        # Save to file
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved rotated point cloud with {all_points.shape[0]} points to {output_path}")

def volume_z_2(sample_pos, rgb_density, output_path='sample_points_z_2_nn.ply'):
    """
    Visualize the 3D points with their associated RGB color and save them in .ply format.
    Rotates the volume 90 degrees around the Z axis.
    
    :param sample_pos: (H, W, N_sample, 3) torch.Tensor
    :param rgb_density: (H, W, N_sample, 4) torch.Tensor, the first three channels are RGB
    """
    with torch.no_grad():
        # Convert to numpy arrays
        pts = sample_pos.detach().cpu().numpy()  # (H, W, N_sample, 3)
        rgb = rgb_density[..., :3].detach().cpu().numpy()  # (H, W, N_sample, 3)

        # Apply sigmoid to ensure RGB values are in [0,1]
        rgb = 1 / (1 + np.exp(-rgb))
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

        # Initialize lists for points and colors
        all_points = []
        all_colors = []

        H, W, N_sample, _ = pts.shape
        for h in range(H):
            for w in range(W):
                for n in range(N_sample):
                    all_points.append(pts[h, w, n])
                    all_colors.append(rgb[h, w, n])

        all_points = np.array(all_points)
        all_colors = np.array(all_colors)

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors.astype(np.float32) / 255.0)

        # Save to file
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved rotated point cloud with {all_points.shape[0]} points to {output_path}")

def write_volume_density(sample_pos, rgb_density, output_path='sample_density_points.ply'):
    """
    Visualise les points 3D avec une couleur basée sur la densité (sigma) et les sauvegarde en .ply
    :param sample_pos: (H, W, N_sample, 3) torch.Tensor
    :param rgb_density: (H, W, N_sample, 4) torch.Tensor
    """
    with torch.no_grad():
        # Convertir en numpy
        pts = sample_pos.detach().cpu().numpy().reshape(-1, 3)
        density = rgb_density[..., 3].detach().cpu().numpy().reshape(-1)  # shape: (H*W*N_sample,)

        # Normalize density to [0, 1] for colormap
        density_normalized = (density - density.min()) / (density.max() - density.min() + 1e-5)

        # Filtrer les points avec une densité > 0
        #mask = density_normalized > 0
        #pts = pts[mask]
        #density = density[mask]

        # Map density to color using a matplotlib colormap (e.g., plasma, viridis, inferno)
        colors = plt.cm.plasma(density_normalized)[:, :3]  # get RGB from colormap (drop alpha)

        # Créer un nuage de points Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float32))

        # Sauvegarder
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved density-colored point cloud with {pts.shape[0]} points to {output_path}")
        
mask_mlo = cv2.imread('mask_mlo_processed.png', cv2.IMREAD_UNCHANGED)
img_mlo = np.rot90(mask_mlo, k=-1)  # k=1 pour -90°
img_mloo = cv2.flip(img_mlo, 1)
mask_mlo_fin = cv2.resize(img_mloo , (int(img_mloo.shape[1]/16), int(img_mloo.shape[0]/16)))



def volume_mlo_lesion(pts_lesion, rgb_lesion, save_ply=True, output_path='sample_points_mlo_n_lesion.ply'):
    # Optional: save to ply after moving data to cpu and converting to numpy
    if save_ply:
        pts_np = pts_lesion.detach().cpu().numpy()
        red_color = np.array([0, 255, 0], dtype=np.uint8)
        rgb_np = (rgb_lesion.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        rgb_np = np.tile(red_color, (rgb_np.shape[0], 1))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_np)
        pcd.colors = o3d.utility.Vector3dVector(rgb_np.astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved lesion point cloud with {pts_np.shape[0]} points to {output_path}")

    return pts_lesion, rgb_lesion

def volume_mlo_point_lesion(sample_pos, rgb_density, mask_mlo_fin):
    """
    Extract 3D points and RGB values corresponding to lesion areas, using all samples along rays.

    Args:
        sample_pos: (H, W, N_sample, 3) torch.Tensor 
            Sampled 3D points along rays (requires_grad=True).
        rgb_density: (H, W, N_sample, 4) torch.Tensor 
            RGB (3 channels) and density (1 channel) for each sample point (requires_grad=True).
        mask_cc_fin: (H, W) numpy array or torch.Tensor 
            Lesion mask with 255 indicating lesion pixels.

    Returns:
        pts_lesion: (N_points, 3) torch.Tensor 
            3D coordinates of lesion voxels.
        rgb_lesion: (N_points, 3) torch.Tensor 
            RGB values in [0, 1] for lesion points.
    """
    import torch

    # Ensure mask is torch tensor on the same device
    if not torch.is_tensor(mask_mlo_fin):
        mask_mlo_fin = torch.tensor(mask_mlo_fin, device=sample_pos.device)
    else:
        mask_mlo_fin = mask_mlo_fin.to(sample_pos.device)

    # Binary lesion mask (H, W)
    lesion_mask = (mask_mlo_fin == 255)

    H, W, N_sample, _ = sample_pos.shape

    # Expand lesion mask to (H, W, N_sample)
    lesion_mask_3d = lesion_mask.unsqueeze(-1).expand(-1, -1, N_sample)  # (H, W, N_sample)

    # Flatten everything
    pts_flat = sample_pos.view(-1, 3)                     # (H * W * N_sample, 3)
    rgb_flat = rgb_density[..., :3].view(-1, 3)           # (H * W * N_sample, 3)
    rgb_flat = torch.sigmoid(rgb_flat)                   # Normalize RGB to [0, 1]
    mask_flat = lesion_mask_3d.reshape(-1)               # (H * W * N_sample)

    # Apply lesion mask
    pts_lesion = pts_flat[mask_flat]
    rgb_lesion = rgb_flat[mask_flat]

    return pts_lesion, rgb_lesion

def volume_mlo_point_lesion__(sample_pos, rgb_density, mask_mlo_fin):
    """
    Differentiable version for training; optionally save point cloud as PLY.
    
    Args:
        sample_pos: (H, W, N_sample, 3) torch.Tensor (requires_grad=True)
        rgb_density: (H, W, N_sample, 4) torch.Tensor (requires_grad=True)
        mask_cc_fin: (H, W) numpy array or torch tensor, lesion mask with 255 in lesion pixels
        save_ply: bool, if True saves a PLY point cloud file
        output_path: str, filename to save PLY
    
    Returns:
        pts_lesion: (N_points, 3) torch.Tensor (on same device as inputs)
        rgb_lesion: (N_points, 3) torch.Tensor in [0, 1]
    """

    # Convert mask to torch bool tensor on device
    if not torch.is_tensor(mask_mlo_fin):
        mask_mlo_fin = torch.tensor(mask_mlo_fin, device=sample_pos.device)
    else:
        mask_mlo_fin = mask_mlo_fin.to(sample_pos.device)
    lesion_mask = (mask_mlo_fin == 255)  # bool mask (H, W)

    sample_idx = 4  # fixed sample index as before

    # Select points and RGB at sample index
    pts_selected = sample_pos[:, :, 2, :3]
    rgb_selected = rgb_density[:, :, 2, :3]

    # Apply sigmoid to RGB to restrict [0,1]
    rgb_selected = torch.sigmoid(rgb_selected)

    # Flatten spatial dims for boolean indexing
    H, W, _ = pts_selected.shape
    pts_flat = pts_selected.view(-1, 3)
    rgb_flat = rgb_selected.view(-1, 3)
    mask_flat = lesion_mask.view(-1)

    # Select lesion points and colors
    pts_lesion = pts_flat[mask_flat]
    rgb_lesion = rgb_flat[mask_flat]

    return pts_lesion, rgb_lesion

mask_cc = cv2.imread('mask_cc_processed.png', cv2.IMREAD_UNCHANGED)
img_cc = np.rot90(mask_cc, k=-1)  # k=1 pour -90°
cc = cv2.flip(img_cc, 1)
#mask_cc_fin = cv2.resize(img_cc , (int(img_cc.shape[1]/16), int(img_cc.shape[0]/16)))
mask_cc_fin = cv2.resize(cc , (int(cc.shape[1]/16), int(cc.shape[0]/16)))

def volume_cc_lesion(pts_lesion, rgb_lesion, save_ply=True, output_path='sample_points_cc_n_lesion.ply'):
    # Optional: save to ply after moving data to cpu and converting to numpy
    if save_ply:
        pts_np = pts_lesion.detach().cpu().numpy()
        red_color = np.array([255, 0, 0], dtype=np.uint8)
        rgb_np = (rgb_lesion.detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        rgb_np = np.tile(red_color, (rgb_np.shape[0], 1))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_np)
        pcd.colors = o3d.utility.Vector3dVector(rgb_np.astype(np.float64) / 255.0)
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved lesion point cloud with {pts_np.shape[0]} points to {output_path}")

    return pts_lesion, rgb_lesion

def volume_cc_point_lesion__(sample_pos, rgb_density, mask_cc_fin):
    """
    Differentiable version for training; optionally save point cloud as PLY.
    
    Args:
        sample_pos: (H, W, N_sample, 3) torch.Tensor (requires_grad=True)
        rgb_density: (H, W, N_sample, 4) torch.Tensor (requires_grad=True)
        mask_cc_fin: (H, W) numpy array or torch tensor, lesion mask with 255 in lesion pixels
        save_ply: bool, if True saves a PLY point cloud file
        output_path: str, filename to save PLY
    
    Returns:
        pts_lesion: (N_points, 3) torch.Tensor (on same device as inputs)
        rgb_lesion: (N_points, 3) torch.Tensor in [0, 1]
    """

    # Convert mask to torch bool tensor on device
    if not torch.is_tensor(mask_cc_fin):
        mask_cc_fin = torch.tensor(mask_cc_fin, device=sample_pos.device)
    else:
        mask_cc_fin = mask_cc_fin.to(sample_pos.device)
    lesion_mask = (mask_cc_fin == 255)  # bool mask (H, W)

    sample_idx = 2  # fixed sample index as before

    # Select points and RGB at sample index
    pts_selected = sample_pos[:, :, sample_idx,  :3]  # (H, W, 3)
    rgb_selected = rgb_density[:, :, sample_idx, :3]  # (H, W, 3)
    # Apply sigmoid to RGB to restrict [0,1]
    rgb_selected = torch.sigmoid(rgb_selected)

    # Flatten spatial dims for boolean indexing
    H, W, _ = pts_selected.shape
    pts_flat = pts_selected.view(-1, 3)
    rgb_flat = rgb_selected.view(-1, 3)
    mask_flat = lesion_mask.view(-1)

    # Select lesion points and colors
    pts_lesion = pts_flat[mask_flat]
    rgb_lesion = rgb_flat[mask_flat]

    return pts_lesion, rgb_lesion

def volume_cc_point_lesion(sample_pos, rgb_density, mask_cc_fin):
    """
    Extract 3D points and RGB values corresponding to lesion areas, using all samples along rays.

    Args:
        sample_pos: (H, W, N_sample, 3) torch.Tensor 
            Sampled 3D points along rays (requires_grad=True).
        rgb_density: (H, W, N_sample, 4) torch.Tensor 
            RGB (3 channels) and density (1 channel) for each sample point (requires_grad=True).
        mask_cc_fin: (H, W) numpy array or torch.Tensor 
            Lesion mask with 255 indicating lesion pixels.

    Returns:
        pts_lesion: (N_points, 3) torch.Tensor 
            3D coordinates of lesion voxels.
        rgb_lesion: (N_points, 3) torch.Tensor 
            RGB values in [0, 1] for lesion points.
    """
    import torch

    # Ensure mask is torch tensor on the same device
    if not torch.is_tensor(mask_cc_fin):
        mask_cc_fin = torch.tensor(mask_cc_fin, device=sample_pos.device)
    else:
        mask_cc_fin = mask_cc_fin.to(sample_pos.device)

    # Binary lesion mask (H, W)
    lesion_mask = (mask_cc_fin == 255)

    H, W, N_sample, _ = sample_pos.shape

    # Expand lesion mask to (H, W, N_sample)
    lesion_mask_3d = lesion_mask.unsqueeze(-1).expand(-1, -1, N_sample)  # (H, W, N_sample)

    # Flatten everything
    pts_flat = sample_pos.view(-1, 3)                     # (H * W * N_sample, 3)
    rgb_flat = rgb_density[..., :3].view(-1, 3)           # (H * W * N_sample, 3)
    rgb_flat = torch.sigmoid(rgb_flat)                   # Normalize RGB to [0, 1]
    mask_flat = lesion_mask_3d.reshape(-1)               # (H * W * N_sample)

    # Apply lesion mask
    pts_lesion = pts_flat[mask_flat]
    rgb_lesion = rgb_flat[mask_flat]

    return pts_lesion, rgb_lesion

mask_z_1 = cv2.imread('mask_cc_processed_1.png', cv2.IMREAD_UNCHANGED)
img_z_1 = np.rot90(mask_z_1, k=1)  # k=1 pour -90°
z_1 = cv2.flip(img_z_1, 1)
mask_z_1_fin = cv2.resize(z_1 , (int(z_1.shape[1]/16), int(z_1.shape[0]/16)))

def volume_z_1_lesion(sample_pos, rgb_density, output_path='sample_points_z_1_n_lesion.ply'):
    """
    Visualize the 3D points with their associated RGB color and save them in .ply format without flattening the arrays.
    
    :param sample_pos: (H, W, N_sample, 3) torch.Tensor
    :param rgb_density: (H, W, N_sample, 4) torch.Tensor, the first three channels are RGB
    """
    with torch.no_grad():
        # Convert to numpy arrays
        pts = sample_pos.detach().cpu().numpy()  # (H, W, N_sample, 3)
        rgb = rgb_density[..., :3].detach().cpu().numpy()  # (H, W, N_sample, 3)

        # Apply sigmoid activation if not done already to ensure RGB is in [0,1]
        rgb = 1 / (1 + np.exp(-rgb))  # Ensure values are between [0, 1]
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

        # Initialize lists for points and colors
        all_points_lesion = []
        all_color_lesion = []
        red_color = np.array([0, 0, 255])  # RGB for red
        # Iterate through the grid to collect points and their corresponding RGB values
        H, W, N_sample, _ = pts.shape
        for h in range(H):
            for w in range(W):
                for n in range(N_sample):
                    if mask_z_1_fin[h, w] == 255:
                        all_points_lesion.append(pts[h , w , n])
                        all_color_lesion.append(red_color)


        # Convert the list of points and colors to numpy arrays
        all_points_lesion = np.array(all_points_lesion)
        all_color_lesion = np.array(all_color_lesion)
        
        # Create an Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points_lesion)
        pcd.colors = o3d.utility.Vector3dVector(all_color_lesion.astype(np.float32) / 255.0)

        # Save the point cloud to file
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved point cloud with {all_points_lesion.shape[0]} points to {output_path}")


mask_z_2 = cv2.imread('mask_mlo_processed.png', cv2.IMREAD_UNCHANGED)
img_z_2 = np.rot90(mask_z_2, k=-1)  # k=1 pour -90°
z_2 = cv2.flip(img_z_2, 1)
mask_z_2_fin = cv2.resize(z_2 , (int(z_2.shape[1]/16), int(z_2.shape[0]/16)))

def volume_z_2_lesion(sample_pos, rgb_density, output_path='sample_points_z_2_n_lesion.ply'):
    """
    Visualize the 3D points with their associated RGB color and save them in .ply format without flattening the arrays.
    
    :param sample_pos: (H, W, N_sample, 3) torch.Tensor
    :param rgb_density: (H, W, N_sample, 4) torch.Tensor, the first three channels are RGB
    """
    with torch.no_grad():
        # Convert to numpy arrays
        pts = sample_pos.detach().cpu().numpy()  # (H, W, N_sample, 3)
        rgb = rgb_density[..., :3].detach().cpu().numpy()  # (H, W, N_sample, 3)

        # Apply sigmoid activation if not done already to ensure RGB is in [0,1]
        rgb = 1 / (1 + np.exp(-rgb))  # Ensure values are between [0, 1]
        rgb = (rgb * 255).clip(0, 255).astype(np.uint8)

        # Initialize lists for points and colors
        all_points_lesion = []
        all_color_lesion = []
        red_color = np.array([0, 255, 255])  # RGB for red
        # Iterate through the grid to collect points and their corresponding RGB values
        H, W, N_sample, _ = pts.shape
        for h in range(H):
            for w in range(W):
                for n in range(N_sample):
                    if mask_z_2_fin[h, w] == 255:
                        all_points_lesion.append(pts[h , w , n])
                        all_color_lesion.append(red_color)


        # Convert the list of points and colors to numpy arrays
        all_points_lesion = np.array(all_points_lesion)
        all_color_lesion = np.array(all_color_lesion)
        
        # Create an Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points_lesion)
        pcd.colors = o3d.utility.Vector3dVector(all_color_lesion.astype(np.float32) / 255.0)

        # Save the point cloud to file
        o3d.io.write_point_cloud(output_path, pcd)
        print(f"Saved point cloud with {all_points_lesion.shape[0]} points to {output_path}")

def eval_one_epoch(scene_train, model, focal_net, pose_param_net,
                   my_devices, args, rgb_act_fn):
    model.eval()
    focal_net.eval()
    pose_param_net.eval()

    fxfy = focal_net(0)
    #ray_dir_cam = comp_ray_dir_cam_fxfy(scene_train.H, scene_train.W, fxfy[0], fxfy[1])
    t_vals = torch.linspace(scene_train.near, scene_train.far, args.num_sample, device=my_devices)  # (N_sample,) sample position
    N_img, H, W = scene_train.N_imgs, scene_train.H, scene_train.W


    rendered_img_list = []
    rendered_depth_list = []
    sample_poses_list= []
    rgb_density_list = []
    rays_dirs_list = []
    rays_orig_list = []
    images_saved = []
    for i in range(N_img):
        c2w = pose_param_net(i)  # (4, 4)
        print(c2w)
        print(focal_net(0))
        #c2w = eval_c2ws[i].to(my_devices)  # (4, 4)
        ray_ori_cam , ray_dir_cam = get_rays(scene_train.H, scene_train.W, fxfy[0] , c2w)

        # split an image to rows when the input image resolution is high
        rendered_img = []
        rendered_depth = []
        render_result = model_render_image(c2w, ray_ori_cam, ray_dir_cam, t_vals, scene_train.near, scene_train.far,
                                               scene_train.H, scene_train.W, fxfy,
                                               model, False, 0.0, args, rgb_act_fn)
        
        rgb_rendered = render_result['rgb']  # (N_select_rows, N_select_cols, 3)
        sample_poses =  render_result['sample_pos']
        rgb_density = render_result['rgb_density']
        #ray_origine = render_result['ray_origine']
        #ray_direction = render_result['ray_direction']
        #poses_encoded = render_result['encoded_poses']
        sample_poses_list.append(sample_poses)
        rays_dirs_list.append(render_result['ray_direction'])
        rays_orig_list.append(render_result['ray_origine'])
        rgb_density_list.append(rgb_density)
        #poses_encoded_list.append(poses_encoded)

        #print('poses in 3D correspendinf to this ray:', sample_poses[0][0])
        #print('the saphe of this poses:', sample_poses[0][0].shape)
        ################### rendering ############################
        # Assuming rgb_rendered is a NumPy array of shape (H, W, 3) and dtype float32 in [0, 1]
        # Step 1: Move to CPU and detach from computation graph
        
        rgb_np = rgb_rendered.detach().cpu().numpy()
        print('rgb min max', np.min(rgb_np), np.max(rgb_np))
        
        #i_min = np.min(rgb_np)
        #i_max = np.max(rgb_np)
        #rgb_np = (rgb_np -i_min)/(i_max - i_min)
        
        # Step 2: Ensure it's in the [0, 255] uint8 range
        rgb_np = (rgb_np * 255).astype('uint8') #.clip(0, 255).astype('uint8')

        #i_min = np.min(rgb_np)
        #i_max = np.max(rgb_np)
        
        print('rgb min max', np.min(rgb_np), np.max(rgb_np))
        
        
        # Step 3: Convert to PIL Image
        image = Image.fromarray(rgb_np)
        images_saved.append(image)

        # Step 4: Save the image
        #image.save('/home/yassine.ameskine/lustre/med_img-z2y8h4a967e/code_yassine/nerfstudio/nerfmm/rendered_image_cc.png')


    images_saved[0].save('rendered_image_cc.png')
    images_saved[1].save('rendered_image_mlo.png')

    np.savez('rays_res/dirs_cc.npz', rays_dirs_list[0].cpu().numpy() )
    np.savez('rays_res/dirs_mlo.npz', rays_dirs_list[1].cpu().numpy() )
    np.savez('rays_res/orig_cc.npz', rays_orig_list[0].cpu().numpy() )
    np.savez('rays_res/orig_mlo.npz', rays_orig_list[1].cpu().numpy() )

    full_pose = torch.cat((sample_poses_list[0], sample_poses_list[1]), dim=2)
    full_rgb = torch.cat((rgb_density_list[0], rgb_density_list[1]), dim=2)

    volume_cc(sample_poses_list[0], rgb_density_list[0])
    #save_rgb_volume_as_pointcloud(rgb_density_list[0])
    volume_mlo(sample_poses_list[1], rgb_density_list[1])
    write_volume_density(sample_poses_list[1], rgb_density_list[1])
    #volume_cc(full_pose, full_rgb)
    mlo_intensity_points, mlo_rgb_lesion = volume_mlo_point_lesion(sample_poses_list[1], rgb_density_list[1], mask_mlo_fin)
    cc_intensity_points , cc_rgb_lesion = volume_cc_point_lesion(sample_poses_list[0], rgb_density_list[0], mask_cc_fin)
    print('shape of point , shape of rgb', sample_poses_list[0].shape, rgb_density_list[0].shape)

    volume_mlo_lesion(mlo_intensity_points, mlo_rgb_lesion)
    volume_cc_lesion(cc_intensity_points , cc_rgb_lesion)
    #volume_mlo_lesion(sample_poses_list[1], rgb_density_list[1], mask_mlo_fin, output_path=True)
    #volume_cc_lesion(sample_poses_list[0], rgb_density_list[0], mask_cc_fin, output_path=True)
    #volume_z_1(sample_poses_list[2], rgb_density_list[2])
    #volume_z_2(sample_poses_list[3], rgb_density_list[3])
    #volume_mlo_lesion(sample_poses_list[1], rgb_density_list[1])
    #volume_cc_lesion(sample_poses_list[0], rgb_density_list[0])
    #volume_z_1_lesion(sample_poses_list[2], rgb_density_list[2])
    #volume_z_2_lesion(sample_poses_list[3], rgb_density_list[3])
    array_cc = rgb_density_list[0][..., :3].cpu().numpy()
    array_lesion_cc = cc_intensity_points.cpu().numpy() 
    array_mlo = rgb_density_list[1][..., :3].cpu().numpy()
    array_lesion_mlo = mlo_intensity_points.cpu().numpy() 
    # Apply sigmoid activation if not done already to ensure RGB is in [0,1]
    array_cc = 1 / (1 + np.exp(-array_cc))  # Ensure values are between [0, 1]
    array_cc = (array_cc * 255).clip(0, 255).astype(np.uint8)
    array_mlo = 1 / (1 + np.exp(-array_mlo))  # Ensure values are between [0, 1]
    array_mlo = (array_mlo * 255).clip(0, 255).astype(np.uint8)
    np.savez('cc_rgb.npz', data= array_cc)
    np.savez('cc_lesion_rgb.npz', data= array_lesion_cc)
    np.savez('mlo_rgb.npz', data= array_mlo)
    np.savez('mlo_lesion_rgb.npz', data= array_lesion_mlo)
    np.savez('cc_point.npz', data= sample_poses_list[0].cpu().numpy() )
    #np.savez('cc_rgb_d.npz', data= rgb_density_list[0].cpu().numpy() )
    np.savez('mlo_point.npz', data= sample_poses_list[1].cpu().numpy() )
    #np.savez('mlo_rgb_d.npz', data= rgb_density_list[1].cpu().numpy() )


        #rgb_rendered_rows = render_result['rgb']  # (num_rows_eval_img, W, 3)
        #depth_map = render_result['depth_map']  # (num_rows_eval_img, W)

        #rendered_img.append(rgb_rendered_rows)
        #rendered_depth.append(depth_map)

        # combine rows to an image
        #rendered_img = torch.cat(rendered_img, dim=0)
        #rendered_depth = torch.cat(rendered_depth, dim=0).unsqueeze(0)  # (1, H, W)

        # for vis
        #rendered_img_list.append(rendered_img.cpu().numpy())
        #rendered_depth_list.append(rendered_depth.cpu().numpy())

    return


def tttrain_one_epoch(scene_train, optimizer_nerf, optimizer_focal, optimizer_pose, model, focal_net, pose_param_net,
                    my_devices, args, rgb_act_fn):
    model.train()
    focal_net.train()
    pose_param_net.train()

    t_vals = torch.linspace(scene_train.near, scene_train.far, args.num_sample, device=my_devices)  # (N_sample,) sample position
    N_img, H, W = scene_train.N_imgs, scene_train.H, scene_train.W
    L2_loss_epoch = []

    # shuffle the training imgs
    ids = np.arange(N_img)
    sample_poses_list = []
    poses_encoded_list = []

    rgb_density_list = []
    #images_saved = []
    for i in ids:
        fxfy = focal_net(0)
        c2w = pose_param_net(i)

        #ray_dir_cam = comp_ray_dir_cam_fxfy(H, W, fxfy[0], fxfy[1])
        ray_orig_cam , ray_dir_cam = get_rays(H, W, fxfy[0] , c2w)
        img = scene_train.imgs[i].to(my_devices)  # (H, W, 3)


        ray_selected_cam = ray_dir_cam  # (H, W, 3)  #is varibal beaceause depend on fxfy
        ray_ori_selected_cam = ray_orig_cam
        img_selected = img[:, :, :3]    # (H, W, 3), discard alpha if present
        
        
        # render an image using selected rays, pose, sample intervals, and the network
        render_result = model_render_image(c2w, ray_ori_selected_cam, ray_selected_cam, t_vals, scene_train.near, scene_train.far,
                                           scene_train.H, scene_train.W, fxfy,
                                           model, True, 0.0, args, rgb_act_fn)  # (N_select_rows, N_select_cols, 3)
                        
        
        rgb_rendered = render_result['rgb']  # (N_select_rows, N_select_cols, 3)
        sample_poses =  render_result['sample_pos']
        rgb_density = render_result['rgb_density']
        ray_origine = render_result['ray_origine']
        ray_direction = render_result['ray_direction']
        sample_poses_list.append(sample_poses)

        rgb_density_list.append(rgb_density)

        
        L2_loss = F.mse_loss(rgb_rendered, img_selected)  # loss for one image

        L2_loss.backward()
        optimizer_nerf.step()
        optimizer_focal.step()
        optimizer_pose.step()
        optimizer_nerf.zero_grad()
        optimizer_focal.zero_grad()
        optimizer_pose.zero_grad()

        L2_loss_epoch.append(L2_loss.item())


    mlo_intensity_points = volume_mlo_point_lesion(sample_poses_list[1], rgb_density_list[1])
    cc_intensity_points = volume_cc_point_lesion(sample_poses_list[0], rgb_density_list[0])
    center_cc = np.mean(cc_intensity_points, axis=0)
    center_mlo = np.mean(mlo_intensity_points, axis=0)
    new_mse_loss = np.mean((center_cc - center_mlo) ** 2)
    new_mse_loss.backward()

    #volume_cc(poses_encoded_list[0], rgb_density_list[0])
    #volume_mlo(poses_encoded_list[1], rgb_density_list[1])
    L2_loss_epoch_mean = np.mean(L2_loss_epoch)  # loss for all images.
    mean_losses = {
        'L2': L2_loss_epoch_mean,
    }
    return mean_losses

def train_one_epoch(scene_train, optimizer_nerf, optimizer_focal, optimizer_pose, model, focal_net, pose_param_net,
                    my_devices, args, rgb_act_fn):
    model.train()
    focal_net.train()
    pose_param_net.train()

    t_vals = torch.linspace(scene_train.near, scene_train.far, args.num_sample, device=my_devices)  # (N_sample,)
    N_img, H, W = scene_train.N_imgs, scene_train.H, scene_train.W
    L2_loss_epoch = []

    ids = np.arange(N_img)
    sample_poses_list = []
    rgb_density_list = []

    for i in ids:
        fxfy = focal_net(0)
        c2w = pose_param_net(i)

        ray_orig_cam , ray_dir_cam = get_rays(H, W, fxfy[0], c2w)
        img = scene_train.imgs[i].to(my_devices)  # (H, W, 3)

        ray_selected_cam = ray_dir_cam
        ray_ori_selected_cam = ray_orig_cam
        img_selected = img[:, :, :3]

        render_result = model_render_image(c2w, ray_ori_selected_cam, ray_selected_cam, t_vals, scene_train.near, scene_train.far,
                                           scene_train.H, scene_train.W, fxfy,
                                           model, True, 0.0, args, rgb_act_fn)

        rgb_rendered = render_result['rgb']  # (H, W, 3)
        sample_poses = render_result['sample_pos']
        rgb_density = render_result['rgb_density']

        sample_poses_list.append(sample_poses)
        rgb_density_list.append(rgb_density)

        L2_loss = F.mse_loss(rgb_rendered, img_selected)  # scalar tensor

        L2_loss_epoch.append(L2_loss)


    # Compute the lesion volumes points — assuming these return numpy arrays
    mlo_points_np, mlo_rgb_np = volume_mlo_point_lesion(sample_poses_list[1], rgb_density_list[1], mask_mlo_fin)
    cc_points_np, cc_rgb_np = volume_cc_point_lesion(sample_poses_list[0], rgb_density_list[0], mask_cc_fin)

    center_cc = torch.mean(cc_points_np, dim=0)
    center_mlo = torch.mean(mlo_points_np, dim=0)

    new_mse_loss = F.mse_loss(center_cc, center_mlo)

    # Sum all image losses and the new loss
    #total_loss =  torch.stack(L2_loss_epoch).mean()
    total_loss =  torch.stack(L2_loss_epoch).mean() + new_mse_loss 
    # Backpropagate combined loss once
    total_loss.backward()

    # Step optimizers once
    optimizer_nerf.step()
    optimizer_focal.step()
    optimizer_pose.step()

    # Zero gradients
    optimizer_nerf.zero_grad()
    optimizer_focal.zero_grad()
    optimizer_pose.zero_grad()

    mean_losses = {
        'L2': torch.stack(L2_loss_epoch).mean().item(),
        'new_mse_loss': new_mse_loss.item(),
        'total_loss': total_loss.item()
    }
    return mean_losses


def main(args):
    my_devices = torch.device('cuda:' + str(args.gpu_id))
    '''Create Folders'''
    exp_root_dir = Path(os.path.join('./logs/any_folder', args.scene_name))
    exp_root_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir = Path(os.path.join(exp_root_dir, gen_detail_name(args)))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy('./models/nerf_models.py', experiment_dir)
    shutil.copy('./models/intrinsics.py', experiment_dir)
    shutil.copy('./models/poses.py', experiment_dir)
    shutil.copy('./tasks/any_folder/train.py', experiment_dir)

    '''LOG'''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(os.path.join(experiment_dir, 'log.txt'))
    file_handler.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.info(args)



    '''Data Loading'''
    scene_train = DataLoaderAnyFolder(base_dir=args.base_dir,
                                      scene_name=args.scene_name,
                                      res_ratio=args.resize_ratio,
                                      num_img_to_load=args.train_img_num,
                                      start=args.train_start,
                                      end=args.train_end,
                                      skip=args.train_skip,
                                      load_sorted=args.train_load_sorted)

    print('Train with {0:6d} images.'.format(scene_train.imgs.shape[0]))

    # We have no eval pose in this any_folder task. Eval with a 4x4 identity pose.
    #eval_c2ws = torch.eye(4).unsqueeze(0).float()  # (1, 4, 4)

    '''Model Loading'''
    pos_enc_in_dims = (2 * args.pos_enc_levels + int(args.pos_enc_inc_in)) * 3  # (2L + 0 or 1) * 3
    if args.use_dir_enc:
        dir_enc_in_dims = (2 * args.dir_enc_levels + int(args.dir_enc_inc_in)) * 3  # (2L + 0 or 1) * 3
    else:
        dir_enc_in_dims = 0

    model = OfficialNerf(pos_enc_in_dims, dir_enc_in_dims, args.hidden_dims)
    if args.multi_gpu:
        model = torch.nn.DataParallel(model).to(device=my_devices)
    else:
        model = model.to(device=my_devices)

    # learn focal parameter
    focal_net = LearnFocal(scene_train.H, scene_train.W, args.learn_focal, args.fx_only, order=args.focal_order, init_focal=138)
    if args.multi_gpu:
        focal_net = torch.nn.DataParallel(focal_net).to(device=my_devices)
    else:
        focal_net = focal_net.to(device=my_devices)

    # learn pose for each image
    print("number of images" , scene_train.N_imgs)
    #pose_param_net = LearnPose(scene_train.N_imgs, args.learn_R, args.learn_t, None)

    cc_c2w = torch.tensor([[1.0000e+00, 0, 0,  0],
        [0,  0,  1.0000e+00, 0],
        [ 0, -1.0000e+00,  0, 0],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])

    hmlo_c2w = torch.tensor([[0.7, 0, 0.7,  3.5],
        [-0.7,  0,  0.7, 0],
        [ 0, -1,  0, -1.9014e-05],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    
    xmlo_c2w = torch.tensor([[0.7, 0, 0.7,  2.5],
        [-0.7,  0,  0.7  , 1.4],
        [ 0, -1,  0, -1.9014e-05],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    
    mlo_c2w = torch.tensor([[0.7, 0, 0.7,  2.5],
        [-0.7,  0,  0.7, 1],
        [ 0, -1,  0, -1.9014e-05],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])

    
    z_1_c2w = torch.tensor([[1.0000e+00, 0, 0,  0],
        [0,  0,  1.0000e+00, 11],
        [ 0, -1.0000e+00,  0, 0],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    
    z_2_c2w = torch.tensor([[-0.7, 0, -0.7,  -2.94],
        [-0.7,  0,  0.7, 0],
        [ 0, -1,  0, -1.9014e-05],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    
    ccc_c2w = torch.tensor([[1.0000e+00, 0, 0,  0],
        [0,  0,  1.0000e+00, -0.99],
        [ 0, -1.0000e+00,  0, 0],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    
    mlo_c2w_inverse = torch.tensor([[0.7, 0, 0.7,  3.7],
        [-0.7,  0,  0.7, 0],
        [ 0, -1,  0, -1.9014e-05],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])

    cggmlo_c2w = torch.tensor([[0.9, 0, 0.9,  4],
        [-0.9,  0,  0.9, 1],
        [ 0, -1,  0, -1.9014e-05],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])
    
    cmlo_c2w = torch.tensor([[0.8, 0, 0.8,  4],
        [-0.8,  0,  0.8, 0],
        [ 0, -1,  0, -1.9014e-05],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])

    init_c2w = torch.stack([cc_c2w, mlo_c2w])
    #init_c2w = torch.stack([cc_c2w, mlo_c2w, -z_1_c2w,  z_2_c2w])

    pose_param_net = LearnPose(scene_train.N_imgs, learn_R=True, learn_t=True, init_c2w=init_c2w)
    #eval_c2ws = torch.stack([pose_param_net(i) for i in range(scene_train.N_imgs)])


    if args.multi_gpu:
        pose_param_net = torch.nn.DataParallel(pose_param_net).to(device=my_devices)
    else:
        pose_param_net = pose_param_net.to(device=my_devices)
    

    '''Set Optimiser'''
    optimizer_nerf = torch.optim.Adam(model.parameters(), lr=args.nerf_lr)
    optimizer_focal = torch.optim.Adam(focal_net.parameters(), lr=args.focal_lr)
    optimizer_pose = torch.optim.Adam(pose_param_net.parameters(), lr=args.pose_lr)

    scheduler_nerf = torch.optim.lr_scheduler.MultiStepLR(optimizer_nerf, milestones=args.nerf_milestones,
                                                          gamma=args.nerf_lr_gamma)
    scheduler_focal = torch.optim.lr_scheduler.MultiStepLR(optimizer_focal, milestones=args.focal_milestones,
                                                           gamma=args.focal_lr_gamma)
    scheduler_pose = torch.optim.lr_scheduler.MultiStepLR(optimizer_pose, milestones=args.pose_milestones,
                                                          gamma=args.pose_lr_gamma)

    total_loss_list = []
    l2_loss_list = []
    l2_loss_list_1 = []
    new_mse_loss_list = []
    '''Training'''
    for epoch_i in tqdm(range(args.epoch), desc='epochs'):
        rgb_act_fn = torch.sigmoid
        train_epoch_losses = train_one_epoch(scene_train, optimizer_nerf, optimizer_focal, optimizer_pose,
                                             model, focal_net, pose_param_net, my_devices, args, rgb_act_fn)
        
        train_L2_loss = train_epoch_losses['L2']
        train_psnr = mse2psnr(train_L2_loss)
        l2_loss_list.append(train_psnr)
        l2_loss_list_1.append(train_L2_loss)
        total_loss_list.append(train_epoch_losses['total_loss'])
        new_mse_loss_list.append(train_epoch_losses['new_mse_loss'])
        scheduler_nerf.step()
        scheduler_focal.step()
        scheduler_pose.step()

        total_loss_array = np.array(total_loss_list)
        l2_loss_array = np.array(l2_loss_list)
        l2_loss_array_1 = np.array(l2_loss_list_1)
        new_mse_loss_array = np.array(new_mse_loss_list)
        np.save('total_loss.npy', total_loss_array)
        np.save('psnr.npy', l2_loss_array)
        np.save('bounding_box_loss.npy', new_mse_loss_array)
        np.save('mse for images.npy', l2_loss_array_1)

        if epoch_i % args.eval_interval == 0 and epoch_i > 0:
            with torch.no_grad():
                eval_one_epoch(scene_train, model, focal_net, pose_param_net, my_devices, args, rgb_act_fn)

                fxfy = focal_net(0)
                tqdm.write('Est fx: {0:.2f}, fy {1:.2f}'.format(fxfy[0].item(), fxfy[1].item()))
                logger.info('Est fx: {0:.2f}, fy {1:.2f}'.format(fxfy[0].item(), fxfy[1].item()))

                # save the latest model
                save_checkpoint(epoch_i, model, optimizer_nerf, experiment_dir, ckpt_name='latest_nerf')
                save_checkpoint(epoch_i, focal_net, optimizer_focal, experiment_dir, ckpt_name='latest_focal')
                save_checkpoint(epoch_i, pose_param_net, optimizer_pose, experiment_dir, ckpt_name='latest_pose')
    return


if __name__ == '__main__':
    args = parse_args()
    set_randomness(args)
    main(args)
