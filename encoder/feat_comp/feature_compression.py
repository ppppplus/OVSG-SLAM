import pickle
import numpy as np
import torch
import os, sys
from torch import Tensor
from typing import List
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)
from model import Autoencoder

class FeatureCompressor:
    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt_path = os.path.join(CURRENT_DIR, "comp_ckpt.pth")
        checkpoint = torch.load(ckpt_path)
        encoder_hidden_dims = [256, 128, 64, 32, 3]
        decoder_hidden_dims = [16, 32, 64, 128, 256, 256, 512]

        self.model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to("cuda:0")
        self.model.load_state_dict(checkpoint)
        self.model.eval()

    def encode(self, features: List[Tensor]) -> List[Tensor]:
        batch_features = torch.stack(features, dim=0).to(self.device)
        normalized_features = torch.nn.functional.normalize(batch_features, p=2, dim=1)
        # print(batch_features.shape, normalized_features.shape)
        # print(torch.min(batch_features), torch.max(batch_features))
        # print(torch.min(normalized_features), torch.max(normalized_features))


        with torch.no_grad():
            encoded_feature = self.model.encode(normalized_features)
        return list(encoded_feature.unbind(dim=0)) 
    
    def decode(self, features: List[Tensor]) -> List[Tensor]:
        batch_features = torch.stack(features, dim=0).to(self.device)
        with torch.no_grad():
            decoded_feature = self.model.decode(batch_features)
        return list(decoded_feature.unbind(dim=0)) 
