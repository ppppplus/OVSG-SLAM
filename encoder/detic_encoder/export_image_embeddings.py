import os,sys
# current_dir = os.path.dirname(os.path.abspath(__file__))
# # if current_dir not in sys.path:
# sys.path.insert(0, current_dir)

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

import argparse
# import os
from detic_config import DeticConfig
from processor import DeticProcessor

import pickle

parser = argparse.ArgumentParser(
    description=(
        "Get image embeddings of an input image or directory of images."
    )
)

parser.add_argument(
    "--input",
    type=str,
    required=True,
    help="Path to either a single input image or folder of images.",
)

parser.add_argument(
    "--output",
    type=str,
    required=True,
    help=(
        "Path to the directory where embeddings will be saved. Output will be either a folder "
        "of .pt per image or a single .pt representing image embeddings."
    ),
)

parser.add_argument(
    "--model-type",
    type=str,
    required=True,
    help="The type of model to load, in ['swin', 'convnet', 'res50', 'res18']",
)

parser.add_argument("--vocabulary", 
                    default="lvis", 
                    choices=['lvis', 'custom', 'icra23', 'lvis+icra23',
                                                                 'lvis+ycb_video', 'ycb_video', 'scan_net',
                                                                 'imagenet21k'],
                    help="name of glossary",
)

parser.add_argument("--custom_vocabulary", default="", help="comma separated words")

parser.add_argument("--pred_all_class", action='store_true')

parser.add_argument("--confidence_threshold", type=float, default=0.3)

parser.add_argument("--verbose", action='store_true')

parser.add_argument("--device", type=str, default="cuda", help="The device to run generation on.")


def main(args: argparse.Namespace) -> None:
    print("Loading model...")
    config = DeticConfig(args)
    detic_processor = DeticProcessor(config, args.vocabulary, args.custom_vocabulary)
    h,w = 480,640
    if not os.path.isdir(args.input):
        targets = [args.input]
    else:
        targets = [
            f for f in os.listdir(args.input) if not os.path.isdir(os.path.join(args.input, f))
        ]
        targets = [os.path.join(args.input, f) for f in targets]

    os.makedirs(args.output, exist_ok=True)

    for t in targets:
        print(f"Processing '{t}'...")
        img_name = t.split(os.sep)[-1].split(".")[0]
        image = cv2.imread(t) # (1423, 1908, 3)
        img_shape = image.shape[0:2]
        if image is None:
            print(f"Could not load '{t}' as an image, skipping...")
            continue
        if image.shape[1] > w:
            print("resize", image.shape, "to", (h, w))
            image = cv2.resize(image, (w,h), interpolation = cv2.INTER_AREA)    ### 
            # print(image[0].shape)
        raw_result = detic_processor.infer(image)
        # mask = raw_result.mask
        segmentation_raw_image = raw_result.segmentation_raw_image  # [H,W] ###
        class_indices = raw_result.class_indices    # List [num]    ###
        scores = raw_result.scores  # List [num]
        pred_boxes = raw_result.pred_boxes  # List [num(array[4])]  num*4   ### 
        detected_class_names = raw_result.detected_class_names  # List [num]    ###
        features_np = raw_result.features_np    # List[num(array[512])] num*512
        featuremap = raw_result.featuremap  ### 
        # print("Save results of '{t}'...")
        print("embedding shape: ", featuremap.shape)
        # print("image_embedding_tensor_cropped: ", featuremap.shape)
        # torch.save(featuremap, os.path.join(args.output, f"{img_name}_fmap_CxHxW.pt"))
        data_dict = {
            "img_shape": img_shape,
            "image": image, # [h,w]
            "seg_image": segmentation_raw_image,    # [h,w]
            # "class_indices": np.array(class_indices),   # [num,]
            "pred_boxes": np.array(pred_boxes), # [num,4]
            "class_names": np.array(detected_class_names), #[num,]
            "features_list": features_np, # List[num(array[512])]
            "featuremap": np.array(featuremap.cpu().numpy()) #[512, h,w]

        }
        # with open(os.path.join(args.output, f"{img_name}_dict.pkl"), "wb") as f:
        #     pickle.dump(data_dict, f)
        # print("save %s data dict"%img_name)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)


# from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# import cv2


# checkpoint = "checkpoints/sam_vit_h_4b8939.pth"
# model_type = "vit_h"
# sam = sam_model_registry[model_type](checkpoint=checkpoint)
# sam.to(device='cuda')
# predictor = SamPredictor(sam)

# image = cv2.imread("test/images/IMG_20220408_142309.png")
# predictor.set_image(image)
# image_embedding = predictor.get_image_embedding().cpu().numpy()
# print("embedding shape: ", image_embedding.shape)
# np.save("test/embedding.npy", image_embedding)