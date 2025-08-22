import argparse
import copy
import os
import sys
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

sys.path.insert(0, _BASE_DIR)

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from diff_gaussian_rasterization import GaussianRasterizer as Renderer
from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera

from utils.common_utils import seed_everything
from utils.recon_helpers import setup_camera
from utils.slam_helpers import get_depth_and_silhouette
from utils.slam_external import build_rotation


def load_camera(cfg, scene_path):
    all_params = dict(np.load(scene_path, allow_pickle=True))
    params = all_params
    org_width = params['org_width']
    org_height = params['org_height']
    w2c = params['w2c']
    intrinsics = params['intrinsics']
    k = intrinsics[:3, :3]

    # scale intrinsics to match the visualization resolution
    k[0, :] *= cfg['viz_w'] / org_width
    k[1, :] *= cfg['viz_h'] / org_height
    return w2c, k


def load_scene_data(scene_path, first_frame_w2c, intrinsics, load_semantics=False):
    # load scene data
    all_params = dict(np.load(scene_path, allow_pickle=True))

    for k in all_params.keys():
        if k == 'semantic_id':
            all_params[k] = torch.tensor(all_params[k]).cuda().int()
        else:
            all_params[k] = torch.tensor(all_params[k]).cuda().float()

    intrinsics = torch.tensor(intrinsics).cuda().float()
    first_frame_w2c = torch.tensor(first_frame_w2c).cuda().float()

    keys = [k for k in all_params.keys() if
            k not in ['org_width', 'org_height', 'w2c', 'intrinsics',
                      'gt_w2c_all_frames', 'cam_unnorm_rots', 'cam_trans',
                      'keyframe_time_indices', 'semantic_id']]

    params = all_params
    for k in keys:
        if not isinstance(all_params[k], torch.Tensor):
            params[k] = torch.tensor(all_params[k]).cuda().float()
        else:
            params[k] = all_params[k].cuda().float()

    all_w2cs = []
    num_t = params['cam_unnorm_rots'].shape[-1]
    for t_i in range(num_t):
        cam_rot = F.normalize(params['cam_unnorm_rots'][..., t_i])
        cam_tran = params['cam_trans'][..., t_i]
        rel_w2c = torch.eye(4).cuda().float()
        rel_w2c[:3, :3] = build_rotation(cam_rot)
        rel_w2c[:3, 3] = cam_tran
        all_w2cs.append(rel_w2c.cpu().numpy())

    transformed_pts = params['means3D']

    rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': params['rgb_colors'],
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], device="cuda")
    }
    depth_rendervar = {
        'means3D': transformed_pts,
        'colors_precomp': get_depth_and_silhouette(transformed_pts, first_frame_w2c),
        'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], device="cuda")
    }

    if load_semantics:
        semantic_rendervar = {
            'means3D': transformed_pts,
            'colors_precomp': params['semantic_colors'],
            'rotations': torch.nn.functional.normalize(params['unnorm_rotations']),
            'opacities': torch.sigmoid(params['logit_opacities']),
            'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
            'means2D': torch.zeros_like(params['means3D'], device="cuda")
        }
        semantic_ids = params['semantic_ids'].cuda()
        if semantic_ids.dim() == 1:
            semantic_ids = semantic_ids.unsqueeze(1)
    else:
        semantic_rendervar, semantic_ids = None, None

    return rendervar, depth_rendervar, semantic_rendervar, all_w2cs, semantic_ids


def make_lineset(all_pts, all_cols, num_lines):
    linesets = []
    for pts, cols, num_lines in zip(all_pts, all_cols, num_lines):
        lineset = o3d.geometry.LineSet()
        lineset.points = o3d.utility.Vector3dVector(np.ascontiguousarray(pts, np.float64))
        lineset.colors = o3d.utility.Vector3dVector(np.ascontiguousarray(cols, np.float64))
        pt_indices = np.arange(len(lineset.points))
        line_indices = np.stack((pt_indices, pt_indices - num_lines), -1)[num_lines:]
        lineset.lines = o3d.utility.Vector2iVector(np.ascontiguousarray(line_indices, np.int32))
        linesets.append(lineset)
    return linesets


def render(w2c, k, timestep_data, timestep_depth_data, cfg, render_mask, device="cuda"):
    with torch.no_grad():
        mask_timestep_data = copy.deepcopy(timestep_data)
        for key, value in mask_timestep_data.items():
            mask_timestep_data[key] = value[render_mask]

        mask_timestep_depth_data = copy.deepcopy(timestep_depth_data)
        for key, value in mask_timestep_depth_data.items():
            mask_timestep_depth_data[key] = value[render_mask]

        cam = setup_camera(cfg['viz_w'], cfg['viz_h'], k, w2c, cfg['viz_near'],
                           cfg['viz_far'], device=device)
        white_bg_cam = Camera(
            image_height=cam.image_height,
            image_width=cam.image_width,
            tanfovx=cam.tanfovx,
            tanfovy=cam.tanfovy,
            bg=torch.tensor([1, 1, 1], dtype=torch.float32, device="cuda"),
            scale_modifier=cam.scale_modifier,
            viewmatrix=cam.viewmatrix,
            projmatrix=cam.projmatrix,
            sh_degree=cam.sh_degree,
            campos=cam.campos,
            prefiltered=cam.prefiltered
        )

        im, _, depth, = Renderer(raster_settings=white_bg_cam)(**mask_timestep_data)
        depth_sil, _, _, = Renderer(raster_settings=cam)(**mask_timestep_depth_data)
        differentiable_depth = depth_sil[0, :, :].unsqueeze(0)
        sil = depth_sil[1, :, :].unsqueeze(0)
        return im, depth, sil


def rgbd2pcd(color, depth, w2c, intrinsics, cfg):
    width, height = color.shape[2], color.shape[1]
    CX = intrinsics[0][2]
    CY = intrinsics[1][2]
    FX = intrinsics[0][0]
    FY = intrinsics[1][1]

    # pixel indices
    xx = torch.tile(torch.arange(width).cuda(), (height,))
    yy = torch.repeat_interleave(torch.arange(height).cuda(), width)
    xx = (xx - CX) / FX
    yy = (yy - CY) / FY
    z_depth = depth[0].reshape(-1)

    # camera-frame points
    pts_cam = torch.stack((xx * z_depth, yy * z_depth, z_depth), dim=-1)
    pix_ones = torch.ones(height * width, 1).cuda().float()
    pts4 = torch.cat((pts_cam, pix_ones), dim=1)
    c2w = torch.inverse(torch.tensor(w2c).cuda().float())
    pts = (c2w @ pts4.T).T[:, :3]

    # convert to open3d
    pts = o3d.utility.Vector3dVector(pts.contiguous().double().cpu().numpy())

    # colorize
    if cfg['render_mode'] == 'depth':
        cols = z_depth
        bg_mask = (cols < 15).float()
        cols = cols * bg_mask
        colormap = plt.get_cmap('jet')
        cNorm = plt.Normalize(vmin=0, vmax=torch.max(cols))
        scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=colormap)
        cols = scalarMap.to_rgba(cols.contiguous().cpu().numpy())[:, :3]
        bg_mask = bg_mask.cpu().numpy()
        cols = cols * bg_mask[:, None] + (1 - bg_mask[:, None]) * np.array([1.0, 1.0, 1.0])
        cols = o3d.utility.Vector3dVector(cols)
    else:
        cols = torch.permute(color, (1, 2, 0)).reshape(-1, 3)
        cols = o3d.utility.Vector3dVector(cols.contiguous().double().cpu().numpy())
    return pts, cols


def dim_outside_first_view(pcd, first_w2c, first_k, width, height, outside_factor=0.4):
    """
    Darken points that fall OUTSIDE the first camera view frustum.

    Args:
        pcd: open3d.geometry.PointCloud, colors in [0,1].
        first_w2c: (4,4) numpy world-to-camera of the first view.
        first_k: (3,3) numpy intrinsics scaled to (viz_w, viz_h).
        width, height: ints, first view image size (viz_w, viz_h).
        outside_factor: multiplier for points outside the frustum (0~1, smaller=darker).
    """
    if len(pcd.points) == 0:
        return

    outside_factor = float(np.clip(outside_factor, 0.0, 1.0))

    pts = np.asarray(pcd.points)           # (N, 3)
    cols = np.asarray(pcd.colors).copy()   # (N, 3) in [0,1]

    ones = np.ones((pts.shape[0], 1), dtype=pts.dtype)
    pts_h = np.concatenate([pts, ones], axis=1)      # (N, 4)
    cam = (first_w2c @ pts_h.T).T                    # (N, 4)
    X, Y, Z = cam[:, 0], cam[:, 1], cam[:, 2]

    z_ok = Z > 1e-6
    u = first_k[0, 0] * (X / Z) + first_k[0, 2]
    v = first_k[1, 1] * (Y / Z) + first_k[1, 2]

    in_w = (u >= 0) & (u < width)
    in_h = (v >= 0) & (v < height)
    inside_mask = z_ok & in_w & in_h
    outside_mask = ~inside_mask

    if np.any(outside_mask):
        cols[outside_mask] = np.clip(cols[outside_mask] * outside_factor, 0.0, 1.0)
        pcd.colors = o3d.utility.Vector3dVector(cols)


def rotate_camera_horizon(theta, view_w2c):
    rotation_matrix = np.array([
        [np.cos(theta), 0, np.sin(theta), 0],
        [0, 1, 0, 0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [0, 0, 0, 1]
    ])
    view_w2c = rotation_matrix @ view_w2c
    return view_w2c


def rotate_object_around_z(object_points, angle_increment_degrees, current_angle_degrees=0):
    angle_radians = np.radians(angle_increment_degrees + current_angle_degrees)
    Rz = np.array([[np.cos(angle_radians), -np.sin(angle_radians), 0],
                   [np.sin(angle_radians),  np.cos(angle_radians), 0],
                   [0,                      0,                     1]])
    object_center = np.mean(object_points, axis=0)
    points_centered = object_points - object_center
    points_rotated = np.dot(points_centered, Rz.T)
    rotated_points = points_rotated + object_center
    new_angle_degrees = (current_angle_degrees + angle_increment_degrees) % 360
    return rotated_points, new_angle_degrees


def visualize(scene_path, cfg):
    # load scene data
    w2c, k = load_camera(cfg, scene_path)

    # keep first view for frustum test
    first_w2c = w2c.copy()
    first_k = k.copy()

    load_semantics = cfg.get('load_semantics', False)

    scene_data, scene_depth_data, scene_semantic_data, all_w2cs, semantic_ids = load_scene_data(
        scene_path, w2c, k, load_semantics=load_semantics)

    # points to keep
    render_mask = torch.ones(scene_data['means3D'].shape[0], dtype=torch.bool).cuda()

    # open3d window
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=int(cfg['viz_w'] * cfg['view_scale']),
                      height=int(cfg['viz_h'] * cfg['view_scale']),
                      visible=True)

    im, depth, sil = render(w2c, k, scene_data, scene_depth_data, cfg, render_mask)
    init_pts, init_cols = rgbd2pcd(im, depth, w2c, k, cfg)
    pcd = o3d.geometry.PointCloud()
    pcd.points = init_pts
    pcd.colors = init_cols

    # darken points outside the first camera frustum on init
    dim_outside_first_view(
        pcd, first_w2c, first_k, cfg['viz_w'], cfg['viz_h'], cfg.get('outside_dim', 0.4)
    )

    vis.add_geometry(pcd)

    w = cfg['viz_w']
    h = cfg['viz_h']

    if cfg['visualize_cams']:
        frustum_size = 0.045
        num_t = len(all_w2cs)
        cam_centers = []
        cam_colormap = plt.get_cmap('cool')
        norm_factor = 0.5
        for i_t in range(num_t):
            frustum = o3d.geometry.LineSet.create_camera_visualization(w, h, k, all_w2cs[i_t], frustum_size)
            frustum.paint_uniform_color(np.array(cam_colormap(i_t * norm_factor / num_t)[:3]))
            vis.add_geometry(frustum)
            cam_centers.append(np.linalg.inv(all_w2cs[i_t])[:3, 3])

        num_lines = [1]
        total_num_lines = num_t - 1
        cols = []
        line_colormap = plt.get_cmap('cool')
        norm_factor = 0.5
        for line_t in range(total_num_lines):
            cols.append(np.array(line_colormap((line_t * norm_factor / total_num_lines) + norm_factor)[:3]))
        cols = np.array(cols)
        all_cols = [cols]
        out_pts = [np.array(cam_centers)]
        linesets = make_lineset(out_pts, all_cols, num_lines)
        lines = o3d.geometry.LineSet()
        lines.points = linesets[0].points
        lines.colors = linesets[0].colors
        lines.lines = linesets[0].lines
        vis.add_geometry(lines)

    # initialize view control
    view_k = k * cfg['view_scale']
    view_k[2, 2] = 1
    view_control = vis.get_view_control()
    cparams = o3d.camera.PinholeCameraParameters()

    if cfg['offset_first_viz_cam']:
        view_w2c = w2c
        view_w2c[:3, 3] = view_w2c[:3, 3] + np.array([0, 0, 0.5])
    else:
        view_w2c = w2c

    cparams.extrinsic = view_w2c
    cparams.intrinsic.intrinsic_matrix = view_k
    cparams.intrinsic.height = int(cfg['viz_h'] * cfg['view_scale'])
    cparams.intrinsic.width = int(cfg['viz_w'] * cfg['view_scale'])
    view_control.convert_from_pinhole_camera_parameters(cparams, allow_arbitrary=True)

    render_options = vis.get_render_option()
    render_options.point_size = cfg['view_scale']
    render_options.light_on = False

    # interactive rendering
    while True:
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        view_k = cam_params.intrinsic.intrinsic_matrix
        k = view_k / cfg['view_scale']
        k[2, 2] = 1
        w2c = cam_params.extrinsic

        if cfg['render_mode'] == 'centers':
            pts = o3d.utility.Vector3dVector(scene_data['means3D'][render_mask].contiguous().double().cpu().numpy())
            cols = o3d.utility.Vector3dVector(scene_data['colors_precomp'][render_mask].contiguous().double().cpu().numpy())
        elif cfg['render_mode'] == 'semantic_color':
            seg, depth, sil = render(w2c, k, scene_semantic_data, scene_depth_data, cfg, render_mask)
            pts, cols = rgbd2pcd(seg, depth, w2c, k, cfg)
        else:
            im, depth, sil = render(w2c, k, scene_data, scene_depth_data, cfg, render_mask)
            if cfg['show_sil']:
                im = (1 - sil).repeat(3, 1, 1)
            pts, cols = rgbd2pcd(im, depth, w2c, k, cfg)

        pcd.points = pts
        pcd.colors = cols

        # darken points outside the first camera frustum on every refresh
        dim_outside_first_view(
            pcd, first_w2c, first_k, cfg['viz_w'], cfg['viz_h'], cfg.get('outside_dim', 0.4)
        )

        vis.update_geometry(pcd)

        if not vis.poll_events():
            break
        vis.update_renderer()

    vis.destroy_window()
    del view_control
    del vis
    del render_options


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", type=str, help="Path to experiment file")
    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    seed_everything(seed=experiment.config["seed"])

    if "scene_path" not in experiment.config:
        results_dir = os.path.join(
            experiment.config["workdir"], experiment.config["run_name"]
        )
        scene_path = os.path.join(results_dir, "params.npz")
    else:
        scene_path = experiment.config["scene_path"]
    viz_cfg = experiment.config["viz"]

    # Visualize final reconstruction
    visualize(scene_path, viz_cfg)
