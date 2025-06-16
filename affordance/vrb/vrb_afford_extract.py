import cv2
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import os, sys
from pathlib import Path
from typing import Union, List, Dict, Any
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

sys.path.append(str(Path(__file__).parent.parent))
from afford_extract import BaseAffordExtractor
os.system("export TORCH_CUDA_ARCH_LIST='8.6'")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CURRENT_DIR)
from networks.model import VRBModel
from networks.traj import TrajAffCVAE
from sklearn.mixture import GaussianMixture
import yaml
from types import SimpleNamespace
from inference import compute_heatmap

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    else:
        return d

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return dict_to_namespace(config_dict)

class VRBExtractor(BaseAffordExtractor):
    def __init__(self, config_path: str = os.path.join(CURRENT_DIR, "configs/vrb.yaml")):
        super().__init__(config_path)
    
    def load_model(self):
        self.config = load_config(self.config_path)
        affhand_head = TrajAffCVAE(
            in_dim=2 * 5,
            hidden_dim=192,
            latent_dim=4,
            condition_dim=256,
            coord_dim=64,
            traj_len=5,
        )

        self.affVRBnet = VRBModel(
            src_in_features=512,
            num_patches=1,
            hidden_dim=192,
            hand_head=affhand_head,
            encoder_time_embed_type="sin",
            num_frames_input=10,
            resnet_type="resnet18",
            embed_dim=256,
            coord_dim=64,
            num_heads=8,
            enc_depth=6,
            attn_kp=1,
            attn_kp_fc=1,
            n_maps=5,
        )
        pth_path = os.path.join(CURRENT_DIR, self.config.model_path)
        dt = torch.load(
            pth_path,
            map_location=torch.device("cuda:0"),
        )
        self.affVRBnet.load_state_dict(dt)
        self.affVRBnet.to(torch.device("cuda:0"))

        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomGrayscale(p=0.05),
                transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.3, hue=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    
    def extract_afford(self, image: np.ndarray) -> Dict[str, Any]:
        # im_out = my_run_inference(self.affVRBnet, image)
        # return {"affordance_map": im_out}
        contact_points = []
        trajectories = []
        mixtures = []
        inp_img = Image.fromarray(image)
        inp_img = self.transform(inp_img).unsqueeze(0).to(torch.device("cuda:0"))
        gm = GaussianMixture(n_components=3, covariance_type="diag")
        centers = []
        trajs = []
        traj_scale = 0.1
        with torch.no_grad():
            ic, pc = self.affVRBnet.inference(inp_img, None, None)
            # pc = pc.cpu().numpy()
            # ic = ic.cpu().numpy()
            pc = pc.to(torch.float32).cpu().numpy()
            ic = ic.to(torch.float32).cpu().numpy()
            i = 0
            w, h = image.shape[:2]
            sm = pc[i, 0] * np.array([h, w])
            centers.append(sm)
            trajs.append(ic[0, 2:])
        gm.fit(np.vstack(centers))
        mixtures.append(sm)
        cp, indx = gm.sample(50)
        x2, y2 = np.vstack(trajs)[np.random.choice(len(trajs))]
        dx, dy = (
            np.array([x2, y2]) * np.array([h, w])
            + np.random.randn(2) * traj_scale
        )
        scale = 40 / max(abs(dx), abs(dy))
        # adjusted_cp = np.array([y1, x1]) + cp
        contact_points.append(cp)
        trajectories.append([x2, y2, dx, dy])

        original_img = np.asarray(image)
        hmap = compute_heatmap(
            np.vstack(contact_points),
            (original_img.shape[1], original_img.shape[0]),
            k_ratio=2,
        )
        hmap = (hmap * 255).astype(np.uint8)
        hmap = cv2.applyColorMap(hmap, colormap=cv2.COLORMAP_JET)
        overlay = (0.6 * original_img + 0.4 * hmap).astype(np.uint8)
        fig = Figure(figsize=(overlay.shape[1]/100, overlay.shape[0]/100), dpi=100)
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        ax.imshow(overlay)

        for i, cp in enumerate(contact_points):
            x2, y2, dx, dy = trajectories[i]
            scale = 60 / max(abs(dx), abs(dy))
            x, y = cp[:, 0], cp[:, 1]
            # ax.arrow(int(np.mean(x)), int(np.mean(y)),
            #         scale * dx, -scale * dy,
            #         color='white', linewidth=2.5, head_width=12)

        ax.axis('off')
        fig.tight_layout(pad=0)

        # 将画好的图保存到内存中并转换为 numpy 图像
        canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        result_img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
        result_img = result_img.reshape(int(height), int(width), 3)
        return result_img
        
