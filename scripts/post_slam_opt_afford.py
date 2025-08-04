import argparse
import os
import random
import sys
import shutil
from importlib.machinery import SourceFileLoader

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _BASE_DIR)

import cv2
import numpy as np
import torch
from tqdm import tqdm
import wandb
from affordance.afford_transfer import gsgrad_vote

from datasets.gradslam_datasets import (
    load_dataset_config,
    ICLDataset,
    ReplicaDataset,
    ReplicaCADDataset,
    AzureKinectDataset,
    ScannetDataset,
    Ai2thorDataset,
    Record3DDataset,
    RealsenseDataset,
    TUMDataset,
    ScannetPPDataset,
    NeRFCaptureDataset
)
from utils.common_utils import seed_everything, save_seq_params, save_params, save_params_ckpt, save_seq_params_ckpt
from utils.recon_helpers import setup_camera
from utils.gs_helpers import (
    params2rendervar, params2depthplussilhouette, semantics2rendervar,
    transformed_params2depthplussilhouette,
    transformed_semantics2rendervar,
    transform_to_frame, report_progress, eval,
    l1_loss_v1, matrix_to_quaternion
)
from utils.gs_external import (
    calc_ssim, build_rotation, densify,
    get_expon_lr_func, update_learning_rate
)

from diff_gaussian_rasterization import GaussianRasterizer as Renderer


def get_dataset(config_dict, basedir, sequence, **kwargs):
    if config_dict["dataset_name"].lower() in ["icl"]:
        return ICLDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replica"]:
        return ReplicaDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["replicacad"]:
        return ReplicaCADDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["azure", "azurekinect"]:
        return AzureKinectDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannet"]:
        return ScannetDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["ai2thor"]:
        return Ai2thorDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["record3d"]:
        return Record3DDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["realsense"]:
        return RealsenseDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["tum"]:
        return TUMDataset(config_dict, basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["scannetpp"]:
        return ScannetPPDataset(basedir, sequence, **kwargs)
    elif config_dict["dataset_name"].lower() in ["nerfcapture"]:
        return NeRFCaptureDataset(basedir, sequence, **kwargs)
    else:
        raise ValueError(f"Unknown dataset name {config_dict['dataset_name']}")


def initialize_optimizer(params, lrs_dict, params_opt_exclude):
    lrs = lrs_dict
    param_groups = [{'params': [v], 'name': k, 'lr': lrs[k]} for k, v in params.items() \
                    if k not in params_opt_exclude and k in lrs]

    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def initialize_first_timestep_from_ckpt(ckpt_path, dataset, num_frames, lrs_dict, mean_sq_dist_method,
                                        load_semantics=False, device="cuda"):
    # Params dont need gradient
    params_opt_exclude = set()
    # Get RGB-D Data & Camera Parameters
    if load_semantics:
        color, depth, intrinsics, pose, semantic_id, semantic_color = dataset[0]
        semantic_id = color.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
        semantic_color = semantic_color.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        params_opt_exclude.add("semantic_ids")
    else:
        color, depth, intrinsics, pose = dataset[0]

    # Process RGB-D Data
    color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
    depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
    
    # Process Camera Parameters
    intrinsics = intrinsics[:3, :3]
    w2c = torch.linalg.inv(pose)

    # Setup Camera
    cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(),
                       w2c.detach().cpu().numpy(), device=device)

    # Get Initial Point Cloud (PyTorch CUDA Tensor)
    mask = (depth > 0) # Mask out invalid depth values
    mask = mask.reshape(-1)

    # Initialize Parameters & Optimizer from Checkpoint
    # Load checkpoint
    print(f"Loading Params")
    params = dict(np.load(ckpt_path, allow_pickle=True))
    variables = {}

    for k in ['intrinsics', 'w2c', 'org_width', 'org_height', 'gt_w2c_all_frames', 'keyframe_time_indices']:
    # for k in ['timestep','intrinsics', 'w2c', 'org_width', 'org_height', 'gt_w2c_all_frames']:
        params.pop(k)

    print(params.keys())
    for k in params.keys():
        if k in params_opt_exclude:
            params[k] = torch.tensor(params[k]).cuda()
        else:
            params[k] = torch.tensor(params[k]).cuda().float().requires_grad_(True)

    if load_semantics and 'semantic_ids' in params.keys():
        if params['semantic_ids'].dim() == 1:
            params['semantic_ids'] = params['semantic_ids'].unsqueeze(1)
    
    variables['max_2D_radius'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
    variables['means2D_gradient_accum'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
    variables['denom'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
    # variables['timestep'] = torch.zeros(params['means3D'].shape[0]).cuda().float()
    variables['timestep'] = torch.tensor(params['timestep']).cuda().float()
    params.pop('timestep')
    optimizer = initialize_optimizer(params, lrs_dict, params_opt_exclude)

    # Initialize an estimate of scene radius for Gaussian-Splatting Densification
    variables['scene_radius'] = torch.max(depth)/2.0

    return params, variables, optimizer, params_opt_exclude, intrinsics, w2c, cam


def get_loss_gs(params, curr_data, variables, loss_weights, load_semantics=False):
    # Initialize Loss Dictionary
    losses = {}

    # Initialize Render Variables
    rendervar = params2rendervar(params)
    depth_sil_rendervar = params2depthplussilhouette(params, curr_data['w2c'])

    # RGB Rendering
    rendervar['means2D'].retain_grad()
    im, radius, _, = Renderer(raster_settings=curr_data['cam'])(**rendervar)
    variables['means2D'] = rendervar['means2D']  # Gradient only accum from colour render for densification

    # Depth & Silhouette Rendering
    depth_sil, _, _, = Renderer(raster_settings=curr_data['cam'])(**depth_sil_rendervar)
    depth = depth_sil[0, :, :].unsqueeze(0)
    silhouette = depth_sil[1, :, :]

    # Semantic Color Rendering
    if load_semantics:
        semantic_rendervar = semantics2rendervar(params)
        seg, _, _, = Renderer(raster_settings=curr_data['cam'])(**semantic_rendervar)
        # Semantic color loss
        losses['seg'] = 0.8 * l1_loss_v1(seg, curr_data['semantic_color']) + \
            0.2 * (1.0 - calc_ssim(seg, curr_data['semantic_color']))

    # Get invalid Depth Mask
    valid_depth_mask = (curr_data['depth'] != 0.0)
    depth = depth * valid_depth_mask

    # RGB Loss
    losses['im'] = 0.8 * l1_loss_v1(im, curr_data['im']) + 0.2 * (1.0 - calc_ssim(im, curr_data['im']))
    
    # Depth Loss
    losses['depth'] = l1_loss_v1(depth, curr_data['depth'])

    weighted_losses = {k: v * loss_weights[k] for k, v in losses.items()}
    loss = sum(weighted_losses.values())

    seen = radius > 0
    variables['max_2D_radius'][seen] = torch.max(radius[seen], variables['max_2D_radius'][seen])
    variables['seen'] = seen
    weighted_losses['loss'] = loss

    return loss, variables, weighted_losses


def initialize_new_params(new_pt_cld, mean3_sq_dist):
    num_pts = new_pt_cld.shape[0]
    means3D = new_pt_cld[:, :3] # [num_gaussians, 3]
    unnorm_rots = np.tile([1, 0, 0, 0], (num_pts, 1)) # [num_gaussians, 3]
    logit_opacities = torch.zeros((num_pts, 1), dtype=torch.float, device="cuda")
    params = {
        'means3D': means3D,
        'rgb_colors': new_pt_cld[:, 3:6],
        'unnorm_rotations': unnorm_rots,
        'logit_opacities': logit_opacities,
        'log_scales': torch.tile(torch.log(torch.sqrt(mean3_sq_dist))[..., None], (1, 1)),
    }
    for k, v in params.items():
        # Check if value is already a torch tensor
        if not isinstance(v, torch.Tensor):
            params[k] = torch.nn.Parameter(torch.tensor(v).cuda().float().contiguous().requires_grad_(True))
        else:
            params[k] = torch.nn.Parameter(v.cuda().float().contiguous().requires_grad_(True))
    return params


def infill_depth(depth, inpaint_radius=1):
    """
    Function to infill Depth for invalid regions
    Input:
        depth: Depth Image (Numpy)
        radius: Radius of the circular neighborhood for infilling
    Output:
        depth: Depth Image with invalid regions infilled (Numpy)
    """
    invalid_mask = (depth == 0)
    invalid_mask = invalid_mask.astype(np.uint8)
    filled_depth = cv2.inpaint(depth, invalid_mask, inpaint_radius, cv2.INPAINT_NS)

    return filled_depth


def convert_params_to_store(params):
    params_to_store = {}
    for k, v in params.items():
        if isinstance(v, torch.Tensor):
            params_to_store[k] = v.detach().clone()
        else:
            params_to_store[k] = v
    return params_to_store

def convert_store_to_params(params_to_store):
    params = {}
    for k, v in params_to_store.items():
        if isinstance(v, np.ndarray):
            params[k] = torch.tensor(v).cuda()
        else:
            params[k] = v  # 保留非张量项
    return params


def rgbd_slam(config: dict):
    # Print Config
    print("Loaded Config:")
    print(f"{config}")

    # Init WandB
    if config['use_wandb']:
        wandb_step = 0
        wandb_time_step = 0
        wandb_run = wandb.init(project=config['wandb']['project'],
                               entity=config['wandb']['entity'],
                               group=config['wandb']['group'],
                               name=config['wandb']['name'],
                               config=config)
        wandb_run.define_metric("Mapping_Iters")
        wandb_run.define_metric("Number of Gaussians - Densification", step_metric="Mapping_Iters")
        wandb_run.define_metric("Learning Rate - Means3D", step_metric="Mapping_Iters")

    # Get Device
    device = torch.device(config["primary_device"])
    if config["primary_device"].startswith("cuda:"):
        device_id = int(config["primary_device"].split(':')[1])
        torch.cuda.set_device(device_id)

    # Load Dataset
    print("Loading Dataset ...")
    dataset_config = config["data"]
    if "gradslam_data_cfg" not in dataset_config:
        gradslam_data_cfg = {}
        gradslam_data_cfg["dataset_name"] = dataset_config["dataset_name"]
    else:
        gradslam_data_cfg = load_dataset_config(dataset_config["gradslam_data_cfg"])
    if "ignore_bad" not in dataset_config:
        dataset_config["ignore_bad"] = False
    if "use_train_split" not in dataset_config:
        dataset_config["use_train_split"] = True
    if "load_semantics" not in dataset_config:
        load_semantics = False
        num_semantic_classes = 0
    else:
        load_semantics = dataset_config["load_semantics"]
        num_semantic_classes = dataset_config["num_semantic_classes"]
        semantic_color_all_frames_map = []
        semantic_id_all_frames_map = []
    if "load_affords" not in dataset_config:
        load_affords = False
    else:
        load_affords = dataset_config["load_affords"]

    # Poses are relative to the first frame
    afford_dataset = get_dataset(
        config_dict=gradslam_data_cfg,
        basedir=dataset_config["basedir"],
        sequence=os.path.basename(dataset_config["sequence"]),
        start=dataset_config["start"],
        end=dataset_config["end"],
        stride=dataset_config["eval_stride"],
        desired_height=dataset_config["desired_image_height"],
        desired_width=dataset_config["desired_image_width"],
        device=device,
        relative_pose=True,
        ignore_bad=dataset_config["ignore_bad"],
        use_train_split=dataset_config["use_train_split"],
        load_semantics=load_semantics,
        load_affords=load_affords,
        num_semantic_classes=num_semantic_classes,
    )

    num_frames = dataset_config["num_frames"]
    if num_frames == -1:
        num_frames = len(afford_dataset)
    # Initialize Parameters, Optimizer & Canoncial Camera parameters
    ckpt_path = config["data"]["param_ckpt_path"]
    params = dict(np.load(ckpt_path, allow_pickle=True))
    params = convert_store_to_params(params)
    vote_np = gsgrad_vote(afford_dataset, params, num_frames, device="cuda")
    output_dir = os.path.join(config["workdir"], config["run_name"])
    # np.save(os.path.join(output_dir, "vote.npy"), vote_np)
    # params = params
    params["vote"] = vote_np
    # Save Parameters
    save_params(params, output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("experiment", type=str, help="Path to experiment file")

    args = parser.parse_args()

    experiment = SourceFileLoader(
        os.path.basename(args.experiment), args.experiment
    ).load_module()

    # Set Experiment Seed
    seed_everything(seed=experiment.config['seed'])
    
    # Create Results Directory and Copy Config
    results_dir = os.path.join(
        experiment.config["workdir"], experiment.config["run_name"]
    )
    os.makedirs(results_dir, exist_ok=True)
    shutil.copy(args.experiment, os.path.join(results_dir, "config.py"))

    rgbd_slam(experiment.config)