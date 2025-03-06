import cv2
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from depth_anything_v2.dpt import DepthAnythingV2


def depth_image(image_path, mask_path, device='cuda:0'):

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

    model = DepthAnythingV2(**model_configs[encoder])
    model.load_state_dict(torch.load(f'./Depth_Anything_V2/depth_anything_v2_vitl.pth', weights_only=True, map_location='cpu'))
    model = model.to(device).eval()
    
    
    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask) / 255.0
    mask = (mask > 0.5).astype(np.float32)
    mask = torch.tensor(mask, dtype=torch.float32)
    
    raw_img = cv2.imread(image_path)

    depth = model.infer_image(raw_img, device=device) # HxW raw depth map in numpy
    mask_center = torch.mean(torch.argwhere(mask > 0).float(), axis=0).int()
    depth_at_center = depth[mask_center[0], mask_center[1]]
    return depth_at_center
