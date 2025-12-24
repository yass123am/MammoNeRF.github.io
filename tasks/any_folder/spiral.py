import sys
import os
import argparse
from pathlib import Path
import torch.nn.functional as F

import torch
import numpy as np
from tqdm import tqdm
import imageio
import math
sys.path.append(os.path.join(sys.path[0], '../..'))

from dataloader.any_folder import DataLoaderAnyFolder
from utils.training_utils import set_randomness, load_ckpt_to_net
from utils.pose_utils import create_spiral_poses
from utils.lie_group_helper import convert3x4_4x4
from models.nerf_models import OfficialNerf
from tasks.any_folder.train import model_render_image
from models.intrinsics import LearnFocal
from models.poses import LearnPose

from utils.volume_op import volume_rendering , volume_sampling_ndc_before, volume_sampling_ndc
from utils.comp_ray_dir import get_rays
import open3d as o3d  # pip install open3d
import torch

#############
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--multi_gpu',  default=False, action='store_true')
    parser.add_argument('--base_dir', type=str, default='./any_folder_demo')
    parser.add_argument('--scene_name', type=str, default='base')

    parser.add_argument('--learn_focal', default=False, type=bool)
    parser.add_argument('--focal_order', default=2, type=int)
    parser.add_argument('--fx_only', default=True, type=eval, choices=[True, False])

    parser.add_argument('--learn_R', default=False, type=bool)
    parser.add_argument('--learn_t', default=False, type=bool)

    parser.add_argument('--resize_ratio', type=int, default=16, help='lower the image resolution with this ratio')
    parser.add_argument('--num_rows_eval_img', type=int, default=1, help='split a high res image to rows in eval')
    parser.add_argument('--hidden_dims', type=int, default=128, help='network hidden unit dimensions')
    parser.add_argument('--num_sample', type=int, default=5, help='number samples along a ray')

    parser.add_argument('--pos_enc_levels', type=int, default=10, help='number of freqs for positional encoding')
    parser.add_argument('--pos_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--use_dir_enc', type=bool, default=True, help='use pos enc for view dir?')
    parser.add_argument('--dir_enc_levels', type=int, default=4, help='number of freqs for positional encoding')
    parser.add_argument('--dir_enc_inc_in', type=bool, default=True, help='concat the input to the encoding')

    parser.add_argument('--rand_seed', type=int, default=17)
    parser.add_argument('--true_rand', type=bool, default=False)

    parser.add_argument('--train_img_num', type=int, default=-1, help='num of images to train')
    parser.add_argument('--train_load_sorted', type=bool, default=False)
    parser.add_argument('--train_start', type=int, default=0, help='inclusive')
    parser.add_argument('--train_end', type=int, default=-1, help='exclusive, -1 for all')
    parser.add_argument('--train_skip', type=int, default=1, help='skip every this number of imgs')

    parser.add_argument('--spiral_mag_percent', type=float, default=50, help='for np.percentile')
    parser.add_argument('--spiral_axis_scale', type=float, default=[1.0, 1.0, 1.0], nargs=3,
                        help='applied on top of percentile, useful in zoom in motion')
    parser.add_argument('--N_img_per_circle', type=int, default=60)
    parser.add_argument('--N_circle_traj', type=int, default=2)

    parser.add_argument('--ckpt_dir', type=str, default='./lr_0.001_gpu0_seed_17_resize_16_Nsam_5_Ntr_img_-1_freq_10__250914_2335')
    return parser.parse_args()


import torch
import torch.nn.functional as F

def slerp(q1, q2, t):
    """Spherical linear interpolation (SLERP) for quaternions"""
    dot = (q1 * q2).sum(dim=-1, keepdim=True)
    dot = torch.clamp(dot, -1.0, 1.0)

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)

    s1 = torch.sin((1.0 - t) * theta) / sin_theta
    s2 = torch.sin(t * theta) / sin_theta

    return s1 * q1 + s2 * q2

def rotation_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (w, x, y, z)"""
    trace = R.trace()
    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    else:
        i = torch.argmax(torch.tensor([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    return F.normalize(torch.tensor([w, x, y, z], dtype=torch.float32), dim=0)

def quaternion_to_rotation(q):
    """Convert quaternion (w, x, y, z) to rotation matrix"""
    w, x, y, z = q
    return torch.tensor([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
    ], dtype=torch.float32)

def interpolate_camera_poses(pose1, pose2, num_steps):
    """Interpolate between two 4x4 poses using SLERP + linear translation"""
    R1, t1 = pose1[:3, :3], pose1[:3, 3]
    R2, t2 = pose2[:3, :3], pose2[:3, 3]

    q1 = rotation_to_quaternion(R1)
    q2 = rotation_to_quaternion(R2)

    interpolated_poses = []

    for i in range(num_steps):
        t = i / (num_steps - 1)
        t_tensor = torch.tensor(t)

        qt = slerp(q1, q2, t_tensor)
        Rt = quaternion_to_rotation(qt)

        tt = (1 - t) * t1 + t * t2

        pose = torch.eye(4)
        pose[:3, :3] = Rt
        pose[:3, 3] = tt
        interpolated_poses.append(pose)

    return torch.stack(interpolated_poses, dim=0)  # shape: (num_steps, 4, 4)

def interpolate_poses_y(pose1, pose_2 , n):
    all_poses = []
    for i in range(1, n):
        angle_rad =  torch.tensor(math.pi / i) 

        rota_y = torch.tensor([[torch.cos(angle_rad), 0,  torch.sin(angle_rad),  2.5],

                 [0,  1,  0 , 1.9014e-05],
        
                 [-torch.sin(angle_rad) , 0 , torch.cos(angle_rad), 1],
        
                 [ 0.0,  0.0,  0.0,  1.0]], dtype=torch.float32, device= 'cpu')

        new_pose = pose1 @ rota_y

        all_poses.append(new_pose)

    return torch.stack(all_poses, dim=0)

def create_half_arc_poses(start_pose, end_pose, n_steps=60):
    t1 = start_pose[:3, 3]
    t2 = end_pose[:3, 3]

    # Compute center of arc
    center = (t1 + t2) / 2

    # Vectors from center
    r1 = t1 - center
    r2 = t2 - center

    # Angle between
    cos_theta = torch.dot(r1, r2) / (torch.norm(r1) * torch.norm(r2))
    theta = torch.acos(torch.clamp(cos_theta, -1.0, 1.0))

    # Compute axis of rotation (normal to plane of arc)
    rot_axis = torch.cross(r1, r2)
    if torch.norm(rot_axis) < 1e-6:
        rot_axis = torch.tensor([0.0, 1.0, 0.0])  # fallback
    else:
        rot_axis = rot_axis / torch.norm(rot_axis)

    def rotation_matrix(axis, angle):
        axis = axis / torch.norm(axis)
        cos = torch.cos(angle)
        sin = torch.sin(angle)
        ux, uy, uz = axis
        return torch.tensor([
            [cos + ux**2 * (1 - cos), ux*uy*(1 - cos) - uz*sin, ux*uz*(1 - cos) + uy*sin],
            [uy*ux*(1 - cos) + uz*sin, cos + uy**2*(1 - cos), uy*uz*(1 - cos) - ux*sin],
            [uz*ux*(1 - cos) - uy*sin, uz*uy*(1 - cos) + ux*sin, cos + uz**2*(1 - cos)]
        ])

    poses = []
    for i in range(n_steps):
        angle = (theta / 2) * (i / (n_steps - 1))  # Half arc
        R = rotation_matrix(rot_axis, angle)
        trans = R @ r1 + center

        pose = torch.eye(4)
        pose[:3, :3] = R @ start_pose[:3, :3]
        pose[:3, 3] = trans
        poses.append(pose)

    return torch.stack(poses)

import torch

def create_full_arc_poses_yaw45(start_pose, end_pose, n_steps=60):
    """
    Generate poses along a full 180° arc (semicircle) between start and end translations,
    while interpolating ORIENTATION only around the global Y axis from start_pose to end_pose.

    Assumes the relative rotation from start to end is approximately a Y-axis (yaw) rotation.
    - Translation path: semicircle in the plane perpendicular to the chord (t2 - t1), centered at (t1 + t2)/2.
    - Rotation path: yaw from R1 to R2 about world Y only.

    Args:
        start_pose: (4,4) torch tensor (SE3)
        end_pose:   (4,4) torch tensor (SE3)
        n_steps:    number of poses along the arc (>= 2)

    Returns:
        (n_steps, 4, 4) torch tensor of poses
    """
    assert start_pose.shape == (4, 4) and end_pose.shape == (4, 4)
    device = start_pose.device
    dtype = start_pose.dtype

    t1 = start_pose[:3, 3]
    t2 = end_pose[:3, 3]
    R1 = start_pose[:3, :3]
    R2 = end_pose[:3, :3]

    # ---------------------- helpers (device/dtype safe) ---------------------- #
    def safe_vec(vals):
        return torch.tensor(vals, device=device, dtype=dtype)

    def rot_axis_angle(axis_vec, angle):
        # axis_vec: (3,), unit; angle: scalar tensor
        ux, uy, uz = axis_vec
        c = torch.cos(angle)
        s = torch.sin(angle)
        oc = 1 - c
        return torch.stack([
            torch.stack([c + ux*ux*oc,      ux*uy*oc - uz*s, ux*uz*oc + uy*s]),
            torch.stack([uy*ux*oc + uz*s,   c + uy*uy*oc,    uy*uz*oc - ux*s]),
            torch.stack([uz*ux*oc - uy*s,   uz*uy*oc + ux*s, c + uz*uz*oc   ])
        ])

    def rot_y(yaw):
        # world-Y rotation; yaw is scalar tensor
        c = torch.cos(yaw)
        s = torch.sin(yaw)
        z = torch.zeros((), device=device, dtype=dtype)
        o = torch.ones((), device=device, dtype=dtype)
        return torch.stack([
            torch.stack([ c, z,  s]),
            torch.stack([ z, o,  z]),
            torch.stack([-s, z,  c]),
        ])

    # ---------------------- translation: full semicircle --------------------- #
    center = (t1 + t2) * 0.5
    r1 = t1 - center
    r_norm = torch.norm(r1)

    # If start and end positions coincide, arc collapses; we’ll just rotate in place.
    if r_norm < 1e-9:
        r1_unit = safe_vec([1.0, 0.0, 0.0])
        r_norm = torch.tensor(0.0, device=device, dtype=dtype)
    else:
        r1_unit = r1 / r_norm

    # Plane normal = something perpendicular to the chord
    chord = t2 - t1
    chord_n = torch.norm(chord)
    if chord_n < 1e-9:
        chord_unit = safe_vec([0.0, 0.0, 1.0])
    else:
        chord_unit = chord / chord_n

    world_z = safe_vec([0.0, 0.0, 1.0])
    world_x = safe_vec([1.0, 0.0, 0.0])

    axis = torch.cross(chord_unit, world_z)
    if torch.norm(axis) < 1e-6:
        axis = torch.cross(chord_unit, world_x)
    if torch.norm(axis) < 1e-9:
        axis = torch.cross(chord_unit, r1_unit)
    if torch.norm(axis) < 1e-9:
        axis = safe_vec([0.0, 1.0, 0.0])  # final fallback
    axis = axis / (torch.norm(axis) + 1e-12)

    # ---------------------- rotation: yaw-only from R1 to R2 ----------------- #
    # Relative rotation
    R_rel = R2 @ R1.transpose(-1, -2)
    # Extract yaw assuming Y-up:
    # Ry(y) = [[ cos y, 0, sin y],
    #          [ 0,     1, 0    ],
    #          [-sin y, 0, cos y]]
    yaw_rel = torch.atan2(R_rel[0, 2], R_rel[2, 2])  # ~ 45° if your assumption holds

    # ---------------------- generate poses ----------------------------------- #
    poses = []
    denom = max(1, n_steps - 1)
    for i in range(n_steps):
        # 1) translate along 0..π (full semicircle)
        alpha = torch.tensor(torch.pi * i / denom, device=device, dtype=dtype)
        R_arc = rot_axis_angle(axis, alpha)
        trans = center + (R_arc @ r1_unit) * r_norm

        # 2) yaw-only interpolate from 0..yaw_rel
        yaw_i = yaw_rel * (i / denom)
        R_yaw_i = rot_y(yaw_i)
        R_i = R_yaw_i @ R1

        pose = torch.eye(4, device=device, dtype=dtype)
        pose[:3, :3] = R_i
        pose[:3, 3] = trans
        poses.append(pose)

    return torch.stack(poses)




def test_one_epoch(eval_c2ws, scene_train, model, focal_net, pose_param_net,
                   my_devices, args, rgb_act_fn):
    model.eval()
    focal_net.eval()
    pose_param_net.eval()

    fxfy = focal_net(0)
    #ray_dir_cam = comp_ray_dir_cam_fxfy(scene_train.H, scene_train.W, fxfy[0], fxfy[1])
    t_vals = torch.linspace(scene_train.near, scene_train.far, args.num_sample, device=my_devices)  # (N_sample,) sample position
    N_img, H, W = eval_c2ws.shape[0], scene_train.H, scene_train.W


    rendered_img_list = []
    rendered_depth_list = []
    sample_poses_list= []
    rgb_density_list = []
    images_saved = []
    for i in range(N_img):
        c2w = eval_c2ws[i].to(my_devices)  # (4, 4)
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

        rgb_density_list.append(rgb_density)
        #poses_encoded_list.append(poses_encoded)

        #print('poses in 3D correspendinf to this ray:', sample_poses[0][0])
        #print('the saphe of this poses:', sample_poses[0][0].shape)
        ################### rendering ############################
        # Assuming rgb_rendered is a NumPy array of shape (H, W, 3) and dtype float32 in [0, 1]
        # Step 1: Move to CPU and detach from computation graph
        
        rgb_np = rgb_rendered.detach().cpu().numpy()
        print('rgb min max', np.min(rgb_np), np.max(rgb_np))
        
        i_min = np.min(rgb_np)
        i_max = np.max(rgb_np)
        rgb_np = (rgb_np -i_min)/(i_max - i_min)
        
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


        image.save('logs/any_folder/base/results/' + str(i) + '.png')
    
    imageio.mimsave('logs/any_folder/base/video/test.mp4', images_saved, fps=30, codec='libx264')
    #imageio.mimwrite('logs/any_folder/base/video/test.mp4', images_saved , fps=30, quality=9)


    return

def main(args):
    my_devices = torch.device('cuda:' + str(args.gpu_id))

    '''Create Folders'''
    test_dir = Path(os.path.join(args.ckpt_dir, 'render_spiral'))
    img_out_dir = Path(os.path.join(test_dir, 'img_out'))
    depth_out_dir = Path(os.path.join(test_dir, 'depth_out'))
    video_out_dir = Path(os.path.join(test_dir, 'video_out'))
    #test_dir.mkdir(parents=True, exist_ok=True)
    #img_out_dir.mkdir(parents=True, exist_ok=True)
    #depth_out_dir.mkdir(parents=True, exist_ok=True)
    #video_out_dir.mkdir(parents=True, exist_ok=True)

    '''Load scene meta'''
    scene_train = DataLoaderAnyFolder(base_dir=args.base_dir,
                                      scene_name=args.scene_name,
                                      res_ratio=args.resize_ratio,
                                      num_img_to_load=args.train_img_num,
                                      start=args.train_start,
                                      end=args.train_end,
                                      skip=args.train_skip,
                                      load_sorted=args.train_load_sorted,
                                      load_img=False)

    print('H: {0:4d}, W: {1:4d}.'.format(scene_train.H, scene_train.W))
    print('near: {0:.1f}, far: {1:.1f}.'.format(scene_train.near, scene_train.far))


    cc_c2w = torch.tensor([[1.0000e+00, 0, 0,  0],
        [0,  0,  1.0000e+00, 0],
        [ 0, -1.0000e+00,  0, 0],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])


    mlo_c2w = torch.tensor([[0.7, 0, 0.7,  0],
        [-0.7,  0,  0.7, 3],
        [ 0, -1,  0, -1.9014e-05],
        [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]])


    init_c2w = torch.stack([cc_c2w, mlo_c2w])
        
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
    model = load_ckpt_to_net('logs/any_folder/base/lr_0.001_gpu0_seed_17_resize_16_Nsam_5_Ntr_img_-1_freq_10__250914_2335/latest_nerf.pth', model, map_location=my_devices)

    focal_net = LearnFocal(scene_train.H, scene_train.W, args.learn_focal, args.fx_only, order=args.focal_order)
    if args.multi_gpu:
        focal_net = torch.nn.DataParallel(focal_net).to(device=my_devices)
    else:
        focal_net = focal_net.to(device=my_devices)
    focal_net = load_ckpt_to_net('logs/any_folder/base/lr_0.001_gpu0_seed_17_resize_16_Nsam_5_Ntr_img_-1_freq_10__250914_2335/latest_focal.pth', focal_net, map_location=my_devices)

    pose_param_net = LearnPose(scene_train.N_imgs, args.learn_R, args.learn_t, init_c2w=init_c2w)
    if args.multi_gpu:
        pose_param_net = torch.nn.DataParallel(pose_param_net).to(device=my_devices)
    else:
        pose_param_net = pose_param_net.to(device=my_devices)
    pose_param_net = load_ckpt_to_net('logs/any_folder/base/lr_0.001_gpu0_seed_17_resize_16_Nsam_5_Ntr_img_-1_freq_10__250914_2335/latest_pose.pth', pose_param_net, map_location=my_devices)
    
    learned_poses = torch.stack([pose_param_net(i) for i in range(scene_train.N_imgs)])

    

    '''Generate camera traj'''
    # This spiral camera traj code is modified from https://github.com/kwea123/nerf_pl.
    # hardcoded, this is numerically close to the formula given in the original repo. Mathematically if near=1
    # and far=infinity, then this number will converge to 4. Borrowed from https://github.com/kwea123/nerf_pl
    interpolated_poses = interpolate_camera_poses(learned_poses[0] , learned_poses[1],  num_steps=200)
    #interpolated_poses = create_half_arc_poses(cc_c2w, mlo_c2w, n_steps=200)
    print(interpolated_poses)
    #radii = np.percentile(np.abs(learned_poses.cpu().numpy()[:, :3, 3]), args.spiral_mag_percent, axis=0)  # (3,)
    #radii *= np.array(args.spiral_axis_scale)
    #c2ws = create_spiral_poses(radii, focus_depth, n_circle=args.N_circle_traj, n_poses=N_novel_imgs)
    #c2ws = torch.from_numpy(c2ws).float()  # (N, 3, 4)

    #c2ws = convert3x4_4x4(c2ws)  # (N, 4, 4)

    '''Render'''
    fxfy = focal_net(0)
    rgb_act_fn = torch.sigmoid
    print('learned fx: {0:.2f}, fy: {1:.2f}'.format(fxfy[0].item(), fxfy[1].item()))
    test_one_epoch(interpolated_poses, scene_train, model, focal_net, pose_param_net, my_devices, args, rgb_act_fn)

    


    return


if __name__ == '__main__':
    args = parse_args()
    set_randomness(args)
    with torch.no_grad():
        main(args)
