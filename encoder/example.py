from urllib.request import urlopen
from PIL import Image
import numpy as np
import torch
import cv2
import torch.nn.functional as F

# from detic_encoder.detic_extractor import DeticFeatureExtractor
from detic_encoder.detic_extractor import DeticFeatureExtractor
from feat_comp.feature_compression import FeatureCompressor

def main():
    config_path = "detic_encoder/configs/detic.yaml"  # 你可以改成你自己的配置文件路径
    # config = load_config(config_path)
    # main(config)
    extractor = DeticFeatureExtractor(config_path)
    img = cv2.imread("test.png")
    data = extractor.extract_features(img)
    featuremap = data["featuremap"]
    features = data["features_list"]
    # print(data["featuremap"].shape)'
    print(len(features))
    # feature1 = features_list[0]
    feature_comp = FeatureCompressor()
    encoded_features = feature_comp.encode(features)
    decoded_features = feature_comp.decode(encoded_features)
    # seg_image = data["seg_image"]
    seg_image = data["seg_image"]
    class_indices = data["class_indices"]
    print(seg_image, class_indices)
    #  = compress_feature(feature1)
    
    print(torch.max(featuremap), torch.min(featuremap))
    features_tensor = torch.stack(features, dim=0)
    encoded_features_tensor = torch.stack(encoded_features, dim=0)
    decoded_features_tensor = torch.stack(decoded_features, dim=0)
    print(features_tensor.shape, encoded_features_tensor.shape, decoded_features_tensor.shape)
    norm_features_tensor = F.normalize(features_tensor, p=2, dim=1).cuda()
    # print(features_tensor.device, norm_feature_tensor.device)
    # score = torch.dot(norm_features_tensor, decoded_features_tensor) 
    similarity = torch.sum(norm_features_tensor * decoded_features_tensor, dim=1) 

    print(torch.max(features_tensor), torch.min(features_tensor))
    print(torch.max(encoded_features_tensor), torch.min(encoded_features_tensor))
    print(torch.max(decoded_features_tensor), torch.min(decoded_features_tensor))
    print(similarity)

    # Load an example image

if __name__ == "__main__":
    main() 