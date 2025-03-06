import json
import os
from openai import OpenAI
import torch
from PIL import Image
import requests
import base64
import openai
import re
import numpy as np
from pathlib import Path
import glob
import ast
import gc
import sys
sys.path.insert(0, './digital_twin/Inpaint_Anything')
import torch
torch.cuda.empty_cache()
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8192"
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.regression import MeanSquaredError

from PIL import Image
import os

import base64
from openai import OpenAI

from PIL import Image
import tempfile
import os

class MetricsCalculator:
    def __init__(self, device) -> None:
        self.device = device

        # CLIP similarity
        self.clip_metric_calculator = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14").to(device)

        # background preservation
        self.psnr_metric_calculator = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.lpips_metric_calculator = LearnedPerceptualImagePatchSimilarity(net_type='squeeze').to(device)
        self.mse_metric_calculator = MeanSquaredError().to(device)
        self.ssim_metric_calculator = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    ####################################################################################
    # 1. CLIP similarity
    def calculate_clip_similarity(self, img, txt, mask=None):
        img = np.array(img)

        if mask is not None:
            mask = np.array(mask)
            img = np.uint8(img * mask)

        img_tensor = torch.tensor(img).permute(2, 0, 1).to(self.device)
        score = self.clip_metric_calculator(img_tensor, txt)
        score = score.cpu().item()
        return score

    # 2. PSNR
    def calculate_psnr(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt

        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        score = self.psnr_metric_calculator(img_pred_tensor, img_gt_tensor)
        score = score.cpu().item()
        return score

    # 3. LPIPS
    def calculate_lpips(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt

        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        score = self.lpips_metric_calculator(img_pred_tensor * 2 - 1, img_gt_tensor * 2 - 1)
        score = score.cpu().item()
        return score

    # 4. MSE
    def calculate_mse(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt

        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).to(self.device)
        score = self.mse_metric_calculator(img_pred_tensor.contiguous(), img_gt_tensor.contiguous())
        score = score.cpu().item()
        return score

    # 5. SSIM
    def calculate_ssim(self, img_pred, img_gt, mask_pred=None, mask_gt=None):
        img_pred = np.array(img_pred).astype(np.float32) / 255
        img_gt = np.array(img_gt).astype(np.float32) / 255
        assert img_pred.shape == img_gt.shape, "Image shapes should be the same."

        if mask_pred is not None:
            mask_pred = np.array(mask_pred).astype(np.float32)
            img_pred = img_pred * mask_pred
        if mask_gt is not None:
            mask_gt = np.array(mask_gt).astype(np.float32)
            img_gt = img_gt * mask_gt

        img_pred_tensor = torch.tensor(img_pred).permute(2, 0, 1).unsqueeze(0).to(self.device)
        img_gt_tensor = torch.tensor(img_gt).permute(2, 0, 1).unsqueeze(0).to(self.device)
        score = self.ssim_metric_calculator(img_pred_tensor, img_gt_tensor)
        score = score.cpu().item()
        return score


def describe_image_changes(image_path1, image_path2, api_key, model_name="qwen-vl-max"):
    prompt = """First, identify all objects present in the first image. \
Then, compare them with the objects in the second image. \
Describe which objects were added, which were removed, or which were replaced. \
Your answer should only contain one sentence. \
"""
    
    with open(image_path1, "rb") as image_file1:
        base64_image1 = base64.b64encode(image_file1.read()).decode("utf-8")
    
    with open(image_path2, "rb") as image_file2:
        base64_image2 = base64.b64encode(image_file2.read()).decode("utf-8")

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    'role': 'system',
                    'content': 'You are a helpful assistant.'
                },
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{base64_image1}"}},
                        {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{base64_image2}"}}
                    ]
                }
            ],
            stream=False,  # 不使用流式处理
            max_tokens=2048
        )
        print(f'--------the response is-------------')
        print(response.choices[0].message.content)
        return response.choices[0].message.content

    except Exception as e:
        print(f"{e}")
        print("https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
        return None

def call_llm(prompt, api_key, model_name="qwen-max"):
    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" 
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                # {
                #     'role': 'system',
                #     'content': 'You are a helpful assistant.'
                # },
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        
                    ]
                }
            ],
            stream=False, 
            max_tokens=100
        )
        return response.choices[0].message.content

    except Exception as e:
        print(f"{e}")
        print("https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
        return None
    
def llm_tt(instruction, sentence):
    api_key=""
    model_name=""
    prompt = f"""You will be given a instruction about image editing \
and a sentence decribing the difference between the original image and the edited image.\
Your task is to score the consistency of the instructions and the sentence on a scale of 1 to 5 points. 
The consistency refers to whether the changes required by the instruction for image editing are consistent \
with the changes in the two images described in the sentence.\
If you think the consistency between instructions and sentences is high, \
in other words, the sentences reflect the results of image editing instructions well, \
you need to score 5 points. On the contrary, give 1 point. \
In term of consistency, you just need to pay attention to whether the changes described in the two images are consistent. \n
The instruction is: {instruction}\n
The sentence is: {sentence}\n
Your answer should only include a single integer from 1 to 5.
"""
    answer = call_llm(prompt=prompt, api_key=api_key, model_name=model_name)
    print(answer)
    return answer

def llm_score_tt(image_path1, image_path2,instruction):
    difference = describe_image_changes(image_path1=image_path1, image_path2=image_path2)
    score = llm_tt(instruction=instruction, sentence=difference)
    return int(score)

def load_json_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def load_images_from_folders(base_folder_path):

    subfolders = [os.path.join(base_folder_path, folder) for folder in os.listdir(base_folder_path)
                  if os.path.isdir(os.path.join(base_folder_path, folder))]
    
    subfolders.sort()
    
    image_list = []
    
    for folder in subfolders:
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
        
        if files:
            image_path = os.path.join(folder, files[0]) 
            try:
                image = Image.open(image_path)
                image_list.append(image)
                print(f"load image: {image_path}")
            except Exception as e:
                print(f"Can't load image {image_path}: {e}")
        else:
            print(f"Folder {folder} has no image.")
    
    return image_list

def eval_single_image(original_image_path, edited_image_path, mask_path, instruction, clip_text, device):
    metrics_size = 512
    metrics_calculator = MetricsCalculator(device)
    original_image = Image.open(original_image_path).convert("RGB")
    original_image = original_image.resize((metrics_size, metrics_size), resample=Image.Resampling.BICUBIC)
    edited_image = Image.open(edited_image_path).convert("RGB")
    edited_image = edited_image.resize((metrics_size, metrics_size), resample=Image.Resampling.BICUBIC)
    mask_image = Image.open(mask_path).convert("L")
    mask_image = mask_image.resize((metrics_size, metrics_size), resample=Image.Resampling.NEAREST)
    
    # crop edited images -> find mask boundary
    mask_array = np.array(mask_image)
    y, x = np.where(mask_array)
    top = np.min(y)
    bottom = np.max(y)
    left = np.min(x)
    right = np.max(x)
    cropped_edited_image = edited_image.crop((left, top, right, bottom))
    cropped_edited_image = np.array(cropped_edited_image)
    cropped_edited_image = torch.tensor(cropped_edited_image).permute(2, 0, 1).to(device) 
    
    # process mask
    mask_image = np.asarray(mask_image, dtype=np.int64) / 255
    mask_image = 1 - mask_image
    mask_image = mask_image[:, :, np.newaxis].repeat([3], axis=2)

    # 1.1. PSNR
    psnr_unedit_part = metrics_calculator.calculate_psnr(img_pred=original_image, img_gt=edited_image,
                                                            mask_pred=mask_image, mask_gt=mask_image)
    # 1.2. LPIPS
    lpips_unedit_part = metrics_calculator.calculate_lpips(img_pred=original_image, img_gt=edited_image,
                                                            mask_pred=mask_image, mask_gt=mask_image)
    # 1.3. MSE
    mse_unedit_part = metrics_calculator.calculate_mse(img_pred=original_image, img_gt=edited_image,
                                                            mask_pred=mask_image, mask_gt=mask_image)
    # 1.4. SSIM
    ssim_unedit_part = metrics_calculator.calculate_ssim(img_pred=original_image, img_gt=edited_image,
                                                            mask_pred=mask_image, mask_gt=mask_image)
    # 1.5 LLM
    llm_score = llm_score_tt(image_path1=original_image_path, image_path2=edited_image_path, instruction=instruction)
    
    clip_metric_calculator = CLIPScore(model_name_or_path="openai/clip-vit-large-patch14-336").to(device)
    CLIP_score_IT = clip_metric_calculator(cropped_edited_image, clip_text)
    CLIP_score_IT = CLIP_score_IT.cpu().item()  
        
    print(f'---------------------------------the result is-----------------------------')
    print(f'PSNR:{psnr_unedit_part};  SSIM:{ssim_unedit_part};  LPIPS:{lpips_unedit_part};  LLM:{llm_score}; CLIP:{CLIP_score_IT}')
      
    
    torch.cuda.empty_cache()  # 清理未使用的 CUDA 缓存
    gc.collect()  # 强制进行垃圾回收
    return psnr_unedit_part, ssim_unedit_part, lpips_unedit_part, llm_score, CLIP_score_IT
