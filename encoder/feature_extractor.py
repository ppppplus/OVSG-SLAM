from abc import ABC, abstractmethod
import torch
from PIL import Image
from typing import Union, List, Dict, Any
import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

class BaseFeatureExtractor(ABC):
    """Base class for all feature extractors."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        # self.model = None
        # self.transform = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    @abstractmethod
    def load_model(self):
        """Load the model and its weights."""
        pass
    
    @abstractmethod
    def extract_features(self, images: Union[Image.Image, List[Image.Image]]) -> Dict[str, Any]:
        """Extract features from input images.
        
        Args:
            images: A single PIL Image or a list of PIL Images
            
        Returns:
            Dict[str, Any]: Extracted data
        """
        pass
    
    def to(self, device: str):
        """Move model to specified device."""
        self.device = torch.device(device)
        if self.model is not None:
            self.model = self.model.to(self.device)
        return self 

# from detic_encoder.detic_extractor import DeticFeatureExtractor