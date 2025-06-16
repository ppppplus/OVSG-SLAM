import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import cv2
import imageio
from natsort import natsorted

from .basedataset import GradSLAMDataset
from .basedataset import readEXR_onlydepth, as_intrinsics_matrix
from . import datautils

class ReplicaDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_semantics: Optional[bool] = False,
        load_embeddings: Optional[bool] = False,
        num_semantic_classes: Optional[int] = 0,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = os.path.join(self.input_folder, "traj.txt")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_semantics=load_semantics,
            load_embeddings=load_embeddings,
            num_semantic_classes=num_semantic_classes,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = natsorted(glob.glob(f"{self.input_folder}/frames/frame*.jpg"))
        depth_paths = natsorted(glob.glob(f"{self.input_folder}/depths/depth*.png"))
        semantic_id_paths = natsorted(glob.glob(f"{self.input_folder}/detic_semantic_ids/*.npy"))
        semantic_color_paths = natsorted(glob.glob(f"{self.input_folder}/detic_semantic_maps/*.npy"))
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, semantic_id_paths, semantic_color_paths, embedding_paths

    def load_poses(self):
        poses = []
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        for i in range(self.num_imgs):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
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
            depth = np.asarray(imageio.imread(depth_path), dtype=np.int64)
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
        if self.load_embeddings:
            embedding = self.read_embedding_from_file(self.embedding_paths[index])
            return_data = return_data + (embedding.to(self.device),) # Allow embedding to be another dtype.

        return return_data
    
class ReplicaV2Dataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        use_train_split: Optional[bool] = True,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 480,
        desired_width: Optional[int] = 640,
        load_semantics: Optional[bool] = False,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.use_train_split = use_train_split
        if self.use_train_split:
            self.input_folder = os.path.join(basedir, sequence, "imap/00")
            self.pose_path = os.path.join(self.input_folder, "traj_w_c.txt")
        else:
            self.train_input_folder = os.path.join(basedir, sequence, "imap/00")
            self.train_pose_path = os.path.join(self.train_input_folder, "traj_w_c.txt")
            self.input_folder = os.path.join(basedir, sequence, "imap/01")
            self.pose_path = os.path.join(self.input_folder, "traj_w_c.txt")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_semantics=load_semantics,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def get_filepaths(self):
        if self.use_train_split:
            color_paths = natsorted(glob.glob(f"{self.input_folder}/rgb/rgb_*.png"))
            depth_paths = natsorted(glob.glob(f"{self.input_folder}/depth/depth_*.png"))
        else:
            first_train_color_path = f"{self.train_input_folder}/rgb/rgb_0.png"
            first_train_depth_path = f"{self.train_input_folder}/depth/depth_0.png"
            color_paths = [first_train_color_path] + natsorted(glob.glob(f"{self.input_folder}/rgb/rgb_*.png"))
            depth_paths = [first_train_depth_path] + natsorted(glob.glob(f"{self.input_folder}/depth/depth_*.png"))
        semantic_paths = None
        embedding_paths = None
        if self.load_embeddings:
            embedding_paths = natsorted(glob.glob(f"{self.input_folder}/{self.embedding_dir}/*.pt"))
        return color_paths, depth_paths, semantic_paths, embedding_paths

    def load_poses(self):
        poses = []
        if not self.use_train_split:
            with open(self.train_pose_path, "r") as f:
                train_lines = f.readlines()
            first_train_frame_line = train_lines[0]
            first_train_frame_c2w = np.array(list(map(float, first_train_frame_line.split()))).reshape(4, 4)
            first_train_frame_c2w = torch.from_numpy(first_train_frame_c2w).float()
            poses.append(first_train_frame_c2w)
        with open(self.pose_path, "r") as f:
            lines = f.readlines()
        if self.use_train_split:
            num_poses = self.num_imgs
        else:
            num_poses = self.num_imgs - 1
        for i in range(num_poses):
            line = lines[i]
            c2w = np.array(list(map(float, line.split()))).reshape(4, 4)
            # c2w[:3, 1] *= -1
            # c2w[:3, 2] *= -1
            c2w = torch.from_numpy(c2w).float()
            poses.append(c2w)
        return poses

    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path)
        return embedding.permute(0, 2, 3, 1)  # (1, H, W, embedding_dim)
    