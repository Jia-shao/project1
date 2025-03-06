import sys
import json
import os
sys.path.insert(0, './sam2')
sys.path.insert(0, './Depth_Anything_V2')
from sam2.mask_pipeline import sam_mask
from Depth_Anything_V2.depth_pipeline import depth_image
from owl_pipeline import label_image, find_label
from pathlib import Path
from PIL import Image
import numpy as np

def merge_masks_by_label(file_path, output_dir):
    with open(file_path, 'r') as file:
        json_data = json.load(file)

    label_groups = {}
    for key, info in json_data.items():
        label = info["semantic label"]
        if label not in label_groups:
            label_groups[label] = {"mask_paths": [], "depths": []}
        label_groups[label]["mask_paths"].append(info["path of mask"])
        label_groups[label]["depths"].append(info["depth"])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    new_json_data = {}
    for idx, (label, group_info) in enumerate(label_groups.items()):
        mask_paths = group_info["mask_paths"]
        depths = group_info["depths"]
        if mask_paths:
            merged_mask = np.array(Image.open(mask_paths[0]).convert("L"))
            
            for mask_path in mask_paths[1:]:
                mask = np.array(Image.open(mask_path).convert("L"))
                merged_mask = np.maximum(merged_mask, mask)  

            output_path = os.path.join(output_dir, f"1mask_result{idx}.png")
            merged_mask_image = Image.fromarray(merged_mask.astype(np.uint8))
            merged_mask_image.save(output_path)
            
            average_depth = np.mean(depths)

            new_json_data[str(idx)] = {
                "path of mask": output_path,
                "semantic label": label,
                "depth": float(average_depth)  
            }

    output_json_path = os.path.join(output_dir, "merged_masks.json")
    with open(output_json_path, 'w') as file:
        json.dump(new_json_data, file, indent=4)

    return output_json_path

def generate_json(image_path, text_prompt, sam_result_path, mask_output_path, dt_path, device):
    result = {}
    
    mask_list = sam_mask(image_path=image_path,
                save_path=sam_result_path,
                mask_output_path=mask_output_path,
                device=device)
    N = len(mask_list)
    boxes, scores, labels = label_image(image_path=image_path,
                                        text_prompt=text_prompt,
                                        device=device)
    for i in range(N):
        mask_path = mask_list[i]
        semantic_label = find_label(boxes, scores, labels, mask_path, text_prompt)
        depth = depth_image(image_path=image_path,
                            mask_path=mask_path,
                            device=device)
        depth = float(depth)
        result[i] = {
            "path of mask": mask_path,
            "semantic label": semantic_label,
            "depth": depth
        }
    output_path = dt_path / f'dt.json'
    with open(output_path,"w") as f:
        json.dump(result, f, indent=4)

    merged_dt_path = dt_path / f'merged'
    merged_dt_path.mkdir(parents=True, exist_ok=True)
    merged_dt_path = str(merged_dt_path)
    merged_output_path = merge_masks_by_label(file_path=output_path, output_dir=merged_dt_path)
    return merged_output_path
