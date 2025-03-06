import cv2
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from typing import Any, Dict, List

from sam_segment import predict_masks_with_sam
from stable_diffusion_inpaint import fill_img_with_sd
from utils import load_img_to_array, save_array_to_img, dilate_mask, \
    show_mask, show_points, get_clicked_point

from PIL import Image

def setup_args(parser):
    parser.add_argument(
        "--input_img", type=str, required=True,
        help="Path to a single input img",
    )
    parser.add_argument(
        "--coords_type", type=str, required=True,
        default="key_in", choices=["click", "key_in"], 
        help="The way to select coords",
    )
    parser.add_argument(
        "--point_coords", type=float, nargs='+', required=True,
        help="The coordinate of the point prompt, [coord_W coord_H].",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+', required=True,
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--text_prompt", type=str, required=True,
        help="Text prompt",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
        help="The type of sam model to load. Default: 'vit_h"
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--seed", type=int,
        help="Specify seed for reproducibility.",
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use deterministic algorithms for reproducibility.",
    )

def replace_main(input_img_path, mask_path, output_dir, text_prompt, device):
    img = load_img_to_array(input_img_path)

    masks = Image.open(mask_path).convert("L")
    masks = np.array(masks) / 255.0
    masks = masks.astype(np.uint8) * 255
    masks = [masks]

    # # dilate mask to avoid unmasked edge effect
    # if args.dilate_kernel_size is not None:
    masks = [dilate_mask(mask, 0) for mask in masks]

    # visualize the segmentation results
    '''    
    img_stem = Path(input_img_path).stem
    out_dir = Path(output_dir) / img_stem
    out_dir.mkdir(parents=True, exist_ok=True)
    '''
    # for idx, mask in enumerate(masks):
    #     # path to the results
    #     mask_p = out_dir / f"mask_{idx}.png"
    #     img_points_p = out_dir / f"with_points.png"
    #     img_mask_p = out_dir / f"with_{Path(mask_p).name}"

    #     # save the mask
    #     save_array_to_img(mask, mask_p)

    #     # save the pointed and masked image
    #     dpi = plt.rcParams['figure.dpi']
    #     height, width = img.shape[:2]
    #     plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
    #     plt.imshow(img)
    #     plt.axis('off')
    #     # show_points(plt.gca(), [latest_coords], 1,
    #     #             size=(width*0.04)**2)
    #     plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
    #     show_mask(plt.gca(), mask, random_color=False)
    #     plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
    #     plt.close()

    # fill the masked image
    for idx, mask in enumerate(masks):
        # if args.seed is not None:
        #     torch.manual_seed(args.seed)
        # mask_p = out_dir / f"mask_{idx}.png"
        # img_filled_p = out_dir / f"filled_with_{Path(mask_p).name}"
        # print(img_filled_p)
        # img_filled_p = output_dir / f"{idx}_edited_image.png"
        img_filled_p = output_dir
        img_filled = fill_img_with_sd(
            img, mask, text_prompt, device=device)
        save_array_to_img(img_filled, img_filled_p)    
    return img_filled_p

