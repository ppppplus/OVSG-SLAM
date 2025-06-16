import torch
import timm
from PIL import Image
from typing import Union, List
import sys
from pathlib import Path

# Add the parent directory to the path to import the base class
sys.path.append(str(Path(__file__).parent.parent))
from feature_extractor import BaseFeatureExtractor

class DINOv2FeatureExtractor(BaseFeatureExtractor):
    """DINOv2 feature extractor implementation."""
    
    def __init__(self, model_name: str = 'vit_large_patch14_dinov2.lvd142m'):
        super().__init__()
        self.model_name = model_name
        self.load_model()
    
    def load_model(self):
        """Load the DINOv2 model and prepare transforms."""
        self.model = timm.create_model(
            self.model_name,
            pretrained=True,
            num_classes=0  # remove classifier nn.Linear
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Get model specific transforms
        data_config = timm.data.resolve_model_data_config(self.model)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
    
    def extract_features(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """Extract features from input images using DINOv2.
        
        Args:
            images: A single PIL Image or a list of PIL Images
            
        Returns:
            torch.Tensor: Extracted features with shape (N, num_features)
        """
        if not isinstance(images, list):
            images = [images]
            
        # Transform all images
        transformed_images = torch.stack([self.transform(img) for img in images])
        transformed_images = transformed_images.to(self.device)
        
        with torch.no_grad():
            # Extract features
            features = self.model.forward_features(transformed_images)
            # Apply head to get final features
            features = self.model.forward_head(features, pre_logits=True)
            
        return features 