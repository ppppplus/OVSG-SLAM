import os
import yaml
import numpy as np
import cv2
import sys
import torch
from pathlib import Path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, CURRENT_DIR)
from detic_config import DeticConfig
from processor import DeticProcessor
import pickle

from detectron2.config import get_cfg
from typing import Dict, Any

from types import SimpleNamespace

# Add the parent directory to the path to import the base class
sys.path.append(str(Path(__file__).parent.parent))
from feature_extractor import BaseFeatureExtractor

# def load_config(config_path):
#     if not os.path.exists(config_path):
#         raise FileNotFoundError(f"Config file not found: {config_path}")
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)
#     return config
def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    else:
        return d

# 读取 YAML 配置文件
def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return dict_to_namespace(config_dict)

class DeticFeatureExtractor(BaseFeatureExtractor):
    """Detic feature extractor implementation."""
    
    def __init__(self, config_path: str = os.path.join(CURRENT_DIR, "configs/detic.yaml")):
        super().__init__()
        # self.config_path = config_path
        self.config = load_config(config_path)
        self.h = self.config.height
        self.w = self.config.width
        self.load_model()
    
    def load_model(self):
        """Load the Detic model and prepare configuration."""
        print("Loading model...")
        detic_config = DeticConfig(self.config)
        self.detic_processor = DeticProcessor(detic_config, self.config.vocabulary, self.config.custom_vocabulary)
    
    def extract_features(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extract features from input image using Detic.

        Args:
            image: OpenCV image as numpy array (H, W, C), BGR format

        Returns:
            Dict containing:
                - 'features': Tensor of shape (N, num_objects, feature_dim)
                - 'boxes': List of detected bounding boxes
                - 'scores': Detection confidence scores
                - 'classes': Predicted class labels
        """
        if image is None:
            print(f"Could not load image, skipping...")
            return None
        img_shape = image.shape[0:2]
        if image.shape[1] > self.w:
            # print("resize", image.shape, "to", (self.h, self.w))
            image = cv2.resize(image, (self.w,self.h), interpolation = cv2.INTER_AREA)    ### 
        raw_result = self.detic_processor.infer(image)
        # mask = raw_result.mask
        segmentation_raw_image = raw_result.segmentation_raw_image  # [H,W] ###
        class_indices = raw_result.class_indices    # List [num]    ###
        scores = raw_result.scores  # List [num]
        pred_boxes = raw_result.pred_boxes  # List [num(array[4])]  num*4   ### 
        detected_class_names = raw_result.detected_class_names  # List [num]    ###
        features_list = raw_result.features    # List[num(array[512])] num*512
        featuremap = raw_result.featuremap  ### 
        # print("Save results of '{t}'...")
        # print("embedding shape: ", featuremap.shape)
        # print("image_embedding_tensor_cropped: ", featuremap.shape)
        # torch.save(featuremap, os.path.join(args.output, f"{img_name}_fmap_CxHxW.pt"))
        data_dict = {
            "img_shape": img_shape,
            "image": image, # [h,w]
            "seg_image": segmentation_raw_image,    # [h,w]
            "class_indices": np.array(class_indices),   # [num,]
            "pred_boxes": np.array(pred_boxes), # [num,4]
            "class_names": np.array(detected_class_names), #[num,]
            "features_list": features_list, # List[num(Tensor[512])]
            "featuremap": featuremap #[512, h,w]
        } 
        return data_dict
    
def main(config):
    print("Loading model...")
    # 将配置字典转换成一个简单的对象，方便传递
    class ConfigNamespace:
        def __init__(self, d):
            self.__dict__.update(d)

    config_ns = ConfigNamespace(config)

    # 初始化模型配置和处理器
    detic_config = DeticConfig(config_ns)
    detic_processor = DeticProcessor(detic_config, config_ns.vocabulary, config_ns.custom_vocabulary)

    h, w = 480, 640
    input_path = config_ns.input
    output_path = config_ns.output

    if not os.path.isdir(input_path):
        targets = [input_path]
    else:
        targets = [
            f for f in os.listdir(input_path) if not os.path.isdir(os.path.join(input_path, f))
        ]
        targets = [os.path.join(input_path, f) for f in targets]

    os.makedirs(output_path, exist_ok=True)

    for t in targets:
        print(f"Processing '{t}'...")
        img_name = os.path.splitext(os.path.basename(t))[0]
        image = cv2.imread(t)
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        img_shape = image.shape[0:2]
        if image.shape[1] > w:
            # print("resize", image.shape, "to", (h, w))
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

        raw_result = detic_processor.infer(image)

        segmentation_raw_image = raw_result.segmentation_raw_image
        class_indices = raw_result.class_indices
        scores = raw_result.scores
        pred_boxes = raw_result.pred_boxes
        detected_class_names = raw_result.detected_class_names
        features_np = raw_result.features_np
        featuremap = raw_result.featuremap

        # print("embedding shape: ", featuremap.shape)

        data_dict = {
            "img_shape": img_shape,
            "image": image,
            "seg_image": segmentation_raw_image,
            "pred_boxes": np.array(pred_boxes),
            "class_names": np.array(detected_class_names),
            "features_list": features_np,
            "featuremap": np.array(featuremap.cpu().numpy())
        }

        # 根据需要保存结果，比如：
        # with open(os.path.join(output_path, f"{img_name}_dict.pkl"), "wb") as f:
        #     pickle.dump(data_dict, f)

if __name__ == "__main__":
    config_path = "configs/detic.yaml"  # 你可以改成你自己的配置文件路径
    # config = load_config(config_path)
    # main(config)
    extractor = DeticFeatureExtractor(config_path)
    img = cv2.imread("../test.png")
    data = extractor.extract_features(img)
    print(data["featuremap"].shape)