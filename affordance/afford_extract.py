from abc import ABC, abstractmethod
import torch
from typing import Union, List, Dict, Any
import os, sys
import numpy as np
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)

class BaseAffordExtractor(ABC):
    """Base class for all feature extractors."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_model()

    @abstractmethod
    def load_model(self):
        """Load the model and its weights."""
        pass
    
    @abstractmethod
    def extract_afford(self, image: np.ndarray) -> Dict[str, Any]:
        """Extract features from input images.
        
        Args:
            images: A single opencv Image or a list of opencv Images
            
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
