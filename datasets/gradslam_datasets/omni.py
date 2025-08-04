import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import cv2
import json
import imageio
from natsort import natsorted

from .basedataset import GradSLAMDataset
from .basedataset import readEXR_onlydepth, as_intrinsics_matrix
from . import datautils

from affordance.vrb.vrb_afford_extract import VRBExtractor

class OmniDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 720,
        desired_width: Optional[int] = 1280,
        load_semantics: Optional[bool] = False,
        load_embeddings: Optional[bool] = False,
        load_affords: Optional[bool] = False,
        num_semantic_classes: Optional[int] = 0,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "transforms_train.json")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_semantics=load_semantics,
            load_embeddings=load_embeddings,
            load_affords=load_affords,
            num_semantic_classes=num_semantic_classes,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        # color_paths = natsorted(glob.glob(f"{self.input_folder}/frames/frame*.jpg"))
        color_paths = natsorted(
            glob.glob(f"{self.input_folder}/images/*.jpg") + 
            glob.glob(f"{self.input_folder}/images/*.png")
        )
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/*.png"))
        semantic_id_paths = natsorted(glob.glob(f"{self.input_folder}/detic_semantic_ids/*.npy"))
        semantic_color_paths = natsorted(glob.glob(f"{self.input_folder}/detic_semantic_maps/*.npy"))
        afford_paths = natsorted(glob.glob(f"{self.input_folder}/affordance_maps/*.npy"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, semantic_id_paths, semantic_color_paths, afford_paths, embedding_paths

    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            data = json.load(f)
        for frame in data["frames"]:
            c2w = np.array(frame["transform_matrix"])
            c2w[:3, 1:3] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)
    
    # def _preprocess_semantic_id(self, semantic_ids: np.ndarray):
    #     semantic_ids = cv2.resize(
    #         semantic_ids,
    #         (self.desired_width, self.desired_height),
    #         interpolation=cv2.INTER_NEAREST,
    #     )
    #     semantic_ids = np.expand_dims(semantic_ids, -1) # (H, W) -> (H, W, 1)
    #     if self.channels_first:
    #         semantic_ids = datautils.channels_first(semantic_ids)
    #     return semantic_ids

    # def _preprocess_semantic_color(self, semantic_colors: np.ndarray):
    #     r"""Preprocesses the semantic colors by resizing, adding channel dimension. Optionally
    #     converts depth from channels last :math:`(H, W, 3)` to channels first :math:`(3, H, W)` representation.

    #     Args:
    #         semantic_labels (np.ndarray): Raw semantic image

    #     Returns:
    #         np.ndarray: Preprocessed semantic labels

    #     Shape:
    #         - semantic_labels: :math:`(H_\text{old}, W_\text{old})`
    #         - Output: :math:`(H, W, 3)` if `self.channels_first == False`, else :math:`(3, H, W)`.
    #     """
    #     semantic_color = cv2.resize(
    #         semantic_colors,
    #         (self.desired_width, self.desired_height),
    #         interpolation=cv2.INTER_NEAREST,
    #     )
    #     if self.normalize_color:
    #         semantic_color = datautils.normalize_image(semantic_color)
    #     if self.channels_first:
    #         semantic_color = datautils.channels_first(semantic_color)
    #     return semantic_color
    
    def __getitem__(self, index):
        # rewrite
        color_path = self.color_paths[index]
        depth_path = self.depth_paths[index]
        color = np.asarray(imageio.imread(color_path), dtype=float)
        color = self._preprocess_color(color)
        if ".png" in depth_path:
            # depth_data = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            depth = np.asarray(imageio.imread(depth_path), dtype=np.uint16)
        elif ".exr" in depth_path:
            depth = readEXR_onlydepth(depth_path)
        
        # load and preprocess semantic labels.
        if self.load_semantics:
            semantic_id_path = self.semantic_id_paths[index]
            # semantic_id = np.asarray(imageio.imread(semantic_id_path), dtype=np.int64)
            semantic_id = np.load(semantic_id_path).astype(np.int32)
            semantic_id = self._preprocess_semantic_id(semantic_id)
            semantic_id = torch.from_numpy(semantic_id)

            semantic_color_path = self.semantic_color_paths[index]
            # semantic_color = np.asarray(imageio.imread(semantic_color_path), dtype=float)
            semantic_color = np.load(semantic_color_path)
            semantic_color = self._preprocess_semantic_color(semantic_color)
            semantic_color = torch.from_numpy(semantic_color)
        
            if self.load_affords:
                affordmap_path = self.afford_paths[index]
                affordmap = np.load(affordmap_path)
                affordmap = self._preprocess_affordmap(affordmap)
                affordmap = torch.from_numpy(affordmap)


        K = as_intrinsics_matrix([self.fx, self.fy, self.cx, self.cy])
        if self.distortion is not None:
            # undistortion is only applied on color image, not depth!
            color = cv2.undistort(color, K, self.distortion)

        color = torch.from_numpy(color)
        K = torch.from_numpy(K)

        depth = self._preprocess_depth(depth)
        depth = torch.from_numpy(depth)

        K = datautils.scale_intrinsics(K, self.height_downsample_ratio, self.width_downsample_ratio)
        intrinsics = torch.eye(4).to(K)
        intrinsics[:3, :3] = K

        pose = self.transformed_poses[index]
        return_data = (
            color.to(self.device).type(self.dtype),
            depth.to(self.device).type(self.dtype),
            intrinsics.to(self.device).type(self.dtype),
            pose.to(self.device).type(self.dtype),
            # self.retained_inds[index].item(),
        )

        if self.load_semantics:
            return_data = return_data + (semantic_id.to(self.device),
                                         semantic_color.to(self.device).type(self.dtype)) # semantic_id has int dtype.
            if self.load_affords:
                return_data = return_data + (affordmap.to(self.device).type(self.dtype),)
        if self.load_embeddings:
            embedding = self.read_embedding_from_file(self.embedding_paths[index])
            return_data = return_data + (embedding.to(self.device),) # Allow embedding to be another dtype.

        return return_data
