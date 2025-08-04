import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import torch
from typing import Dict, Tuple, List
# from detic_extract import detic_extract
from encoder.detic_encoder.detic_extractor import DeticFeatureExtractor
from encoder.feat_comp.feature_compression import FeatureCompressor
from affordance.vrb.vrb_afford_extract import VRBExtractor
from affordance.vrb.vrb_afford_extract import compute_heatmap
from sklearn.decomposition import PCA

class ReplicaDatasetProcess():
    def __init__(self, frames_dir: str = "data/ReplicaCAD/apt_1_start/frames",
        semantic_id_dir: str = "data/ReplicaCAD/apt_1_start/detic_semantic_ids",
        semantic_map_dir: str = "data/ReplicaCAD/apt_1_start/detic_semantic_maps",
        afford_map_dir: str = "data/ReplicaCAD/apt_1_start/affordance_maps" 
        ) -> None:

        self.detic_processor = DeticFeatureExtractor()
        self.feature_comp = FeatureCompressor()
        self.afford_processor = VRBExtractor()
        self.frames_dir = frames_dir
        self.semantic_id_dir = semantic_id_dir
        self.semantic_map_dir = semantic_map_dir
        self.afford_map_dir = afford_map_dir
        os.makedirs(self.semantic_id_dir, exist_ok=True)
        os.makedirs(self.semantic_map_dir, exist_ok=True)
        os.makedirs(self.afford_map_dir, exist_ok=True)
    
    def extract_semantic_data(self, image_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract semantic IDs and feature-based semantic map from the input image.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Tuple containing:
                - semantic_ids: Array of semantic IDs for each pixel
                - semantic_map: RGB semantic map based on reduced features
        """
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # 使用detic_extract获取语义分割ID和特征
        data = self.detic_processor.extract_features(image)
        resimage = data["image"]
        seg_image = data["seg_image"]
        class_indices = data["class_indices"]
        feature_list = data["features_list"]
        pred_boxes = data["pred_boxes"]
        labels = data["class_names"]
        semantic_ids = np.zeros_like(seg_image, dtype=np.uint16)
    
        # 将索引映射到实际的类别ID
        for idx, class_id in enumerate(class_indices):
            mask = (seg_image == (idx+1))
            semantic_ids[mask] = class_id
        
        # semantic_ids = data[""]
        
        # 降维特征
        reduced_feature_list = self.feature_comp.encode(feature_list)
        reduced_feature_list_np =  [f.cpu().numpy() for f in reduced_feature_list]
        
        # 创建语义图
        semantic_map = self.create_semantic_map(seg_image, reduced_feature_list_np)
        
        return semantic_ids, semantic_map, resimage, pred_boxes, labels
    
    def create_semantic_map(self, seg_image: np.ndarray, reduced_features: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Create RGB semantic map from semantic IDs and reduced features.
        
        Args:
            seg_image: Array of semantic IDs for each pixel
            reduced_features: Dictionary mapping semantic IDs to 3D feature vectors
            
        Returns:
            RGB semantic map
        """
        height, width = seg_image.shape
        semantic_map = np.zeros((height, width, 3))
        # semantic_map = reduced_features[seg_image]
        for i in range(np.max(seg_image)):
            mask = (seg_image == (i+1))
            semantic_map[mask] = reduced_features[i]
        # 为每个语义ID填充对应的RGB值
        # for id_ in np.unique(seg_image):
        #     # if id_ in reduced_features:
        #         mask = (seg_image == id_)
        #         semantic_map[mask] = reduced_features[id_]
        
        return semantic_map # [h,w,3]

    def create_affordance_map(self, image: np.ndarray, all_boxes: np.ndarray, all_labels: np.ndarray, k_ratio: float = 10.0) -> np.ndarray:
        objects = [
            "dresser",
            "cabinet",
            # "vase"
            "door",
            "thermostat",
            "wall_socket"
        ]
        bboxes = []
        labels = []
        # print(all_labels)
        for i, label in enumerate(all_labels):
            if label in objects:
                bboxes.append(all_boxes[i])
                labels.append(label)
        if not labels:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        contact_points = []
        trajectories = []
        if all_boxes.shape[0] > 0:
            # box = boxes[0]
            for box in bboxes:
                # print(box)
                y1, x1, y2, x2 = box

                # bbox_offset = 20
                bbox_offset = 0
                y1, x1, y2, x2 = (
                    int(y1) - bbox_offset,
                    int(x1) - bbox_offset,
                    int(y2) + bbox_offset,
                    int(x2) + bbox_offset,
                )

                # width = y2 - y1
                # height = x2 - x1

                # diff = width - height
                # if width > height:
                #     y1 += int(diff / np.random.uniform(1.5, 2.5))
                #     y2 -= int((diff / (np.random.uniform(1.5, 2.5) + diff % 2)))
                # else:
                #     diff = height - width
                #     x1 += int(diff / np.random.uniform(1.5, 2.5))
                #     x2 -= int((diff / (np.random.uniform(1.5, 2.5) + diff % 2)))

                input_img = image[x1:x2, y1:y2]
                res = self.afford_processor.extract_pts_trajs(input_img, [x1, y1, x2, y2])
                # print(res["contact_points"], )
                contact_points.append(res["contact_points"])
                trajectories.append(res["trajectories"])
                    
        original_img = image.copy()
        # print(contact_points)
        hmap = compute_heatmap(
            np.vstack(contact_points),
            (original_img.shape[1], original_img.shape[0]),
            k_ratio=k_ratio,
        )
        # hmap = (hmap * 255).astype(np.uint8)

        return hmap

    def process(self) -> None:
        """
        Process all images in the Replica dataset frames directory.
        Extract semantic IDs and maps, save them to respective directories.
        
        Args:
            frames_dir: Directory containing the input frames
            semantic_id_dir: Directory to save the semantic IDs
            semantic_map_dir: Directory to save the semantic maps
            affor_map_dir: Directory to save the affordace maps
        """

        # 获取所有图片文件
        image_files = sorted([
            f for f in os.listdir(self.frames_dir) 
            if f.endswith(('.png', '.jpg', '.jpeg'))
        ])
        
        if not image_files:
            print(f"No image files found in {self.frames_dir}")
            return
        
        print(f"Found {len(image_files)} images to process")
        
        # 处理每张图片
        for img_file in tqdm(image_files, desc="Processing images"):
        # for img_file in ["frame000459.jpg"]:
            # 构建输入输出路径
            input_path = os.path.join(self.frames_dir, img_file)
            imgname = os.path.splitext(img_file)[0]  # "framexxxxxx"
            semantic_id_path = os.path.join(self.semantic_id_dir, f"{imgname}.npy")
            semantic_map_path = os.path.join(self.semantic_map_dir, f"{imgname}.npy")
            afford_map_path = os.path.join(self.afford_map_dir, f"{imgname}.npy")
            
            try:
                # 提取语义数据
                semantic_ids, semantic_map, image, pred_boxes, labels = self.extract_semantic_data(input_path)
                affordance_map = self.create_affordance_map(image, pred_boxes, labels)
                # 保存结果
                np.save(semantic_id_path, semantic_ids)
                np.save(semantic_map_path, semantic_map)
                np.save(afford_map_path, affordance_map)
                # print(semantic_ids, semantic_map)
                # break
                
            except Exception as e:
                print(f"Error processing {img_file}: {str(e)}")
                continue

if __name__ == "__main__":
    processor = ReplicaDatasetProcess()
    processor.process()
    print("Processing complete!")
