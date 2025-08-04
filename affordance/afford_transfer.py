import torch
import os
import cv2
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from utils.recon_helpers import setup_camera
from diff_gaussian_rasterization import GaussianRasterizer as Renderer

def semantics2rendervar(params):
    rendervar = {
        'means3D': params['means3D'],
        'colors_precomp': params['semantic_colors'],
        'rotations': F.normalize(params['unnorm_rotations']),
        'opacities': torch.sigmoid(params['logit_opacities']),
        'scales': torch.exp(torch.tile(params['log_scales'], (1, 3))),
        'means2D': torch.zeros_like(params['means3D'], requires_grad=True, device="cuda") + 0
    }
    return rendervar
def gsgrad_vote(dataset, params, num_frames, device="cuda"):
    print("grad voting for affordance transfer ...")
    for time_idx in tqdm(range(num_frames)):
        color, depth, intrinsics, pose, semantic_id, semantic_color, afford_map = dataset[time_idx]
        semantic_id = semantic_id.permute(2, 0, 1) # (H, W, 1) -> (1, H, W)
        semantic_color = semantic_color.permute(2, 0, 1) # (H, W, C) -> (C, H, W)

        intrinsics = intrinsics[:3, :3]
        # Process RGB-D Data
        color = color.permute(2, 0, 1) / 255 # (H, W, C) -> (C, H, W)
        depth = depth.permute(2, 0, 1) # (H, W, C) -> (C, H, W)
        w2c = torch.linalg.inv(pose)
        cam = setup_camera(color.shape[2], color.shape[1], intrinsics.cpu().numpy(),
                           w2c.detach().cpu().numpy(), device=device)
        curr_data = {'cam': cam, 'im': color, 'depth': depth, 'id': time_idx, 'intrinsics': intrinsics, 'w2c': w2c}
        # rendervar = params2rendervar(params)
        rendervar = semantics2rendervar(params)

        renderer = Renderer(raster_settings=curr_data['cam'])
        rendervar["colors_precomp"].requires_grad = True

        votes = torch.zeros((rendervar["colors_precomp"].shape[0], 3)).to(device) # Hardcoding for now

        im, radius, _, = renderer(**rendervar)
        # Voting
        # mask = torch.from_numpy(mask).to(device).float()

        loss = (afford_map * im).mean()
        loss.backward(retain_graph=True)
        votes += rendervar["colors_precomp"].grad[..., :3]
        rendervar["colors_precomp"].grad.zero_()
    votes_np = votes.cpu().numpy()
    return votes_np