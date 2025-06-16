from sagesplat.vrb.networks.model import VRBModel
from sagesplat.vrb.networks.traj import TrajAffCVAE
from sagesplat.vrb.inference import run_inference, my_run_inference
import cv2
from PIL import Image
import torch
from torchvision import transforms
import numpy as np
import os
os.system("export TORCH_CUDA_ARCH_LIST='8.6'")
affhand_head = TrajAffCVAE(
            in_dim=2 * 5,
            hidden_dim=192,
            latent_dim=4,
            condition_dim=256,
            coord_dim=64,
            traj_len=5,
        )

affVRBnet = VRBModel(
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
dt = torch.load(
    f"/home/ubuntu/TJH/Work/aff_ws/Splat-MOVER/sagesplat/vrb/models/model_checkpoint_1249.pth.tar",
    map_location=torch.device("cuda:0"),
)
transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomGrayscale(p=0.05),
                transforms.ColorJitter(
                    brightness=0.4, contrast=0.3, saturation=0.3, hue=0.3
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
affVRBnet.load_state_dict(dt)
affVRBnet.to(torch.device("cuda:0"))
# img = cv2.imread("/home/ubuntu/TJH/Work/aff_ws/Splat-MOVER/data/asknerf_pot_burner_orange_2/images/frame_00027.png")
# image = Image.open("/home/ubuntu/TJH/Work/aff_ws/Splat-MOVER/data/asknerf_pot_burner_orange_2/images/frame_00027.png")  # 或其他图像来源
image = Image.open("/home/ubuntu/TJH/Work/aff_ws/Splat-MOVER/data/asknerf_pot_burner_orange_2/images/frame_00027.png").convert("RGB")

im_out = my_run_inference(affVRBnet, image)
