from dtreasoning_prompt import (split_instruction, judge_instruction_one_step, extract_bracket_content, 
                                find_question, find_semantic_label, 
                                find_position, need_which_infor, simplified_editing_prompt, call_llm, 
                                judge_dt_answer, check_first_word_is_no, check_first_word_is_yes, 
                                classify_instruction, mask_list)
from utils_my.str_process import remove_draw_prefix
from repaint_pipeline import inpaint_pipeline
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
sys.path.insert(0, './Inpaint_Anything')
from dttest import generate_json
from Inpaint_Anything.fill_anything import replace_main
from Inpaint_Anything.remove_anything import remove_main
from evaluation import MetricsCalculator
import torch
torch.cuda.empty_cache()
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:8192"
from torchmetrics.multimodal import CLIPScore
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.regression import MeanSquaredError

import subprocess
import os
import tempfile

from PIL import Image
import os
from timeit import default_timer as timer

def copy_image(source_path, destination_path):

    try:
        with Image.open(source_path) as img:
            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            img.save(destination_path)
    except FileNotFoundError:
        print(f"Error。")
    except Exception as e:
        print(f"{e}")

import base64
import io

def descri_img_by_qianwen(image_path, api_key="sk-59e1b47100c94a949c3078700cc0ec6f",
                          model_name="qwen-vl-max"):
    prompt = '''Your task is to list all the objects in the given image. \
Your answer should be in the form ['object1', 'object2', ...]. \
If there are multiple objects with the same label, distinguish them by their positions (e.g., top-left, top-right, bottom-right, bottom-left).'''

    image = Image.open(image_path)
    original_size = os.path.getsize(image_path)

    if original_size > 7.5 * 1024 * 1024:
        resize_ratio = 1.0
        while True:
            new_size = (int(image.width * resize_ratio), int(image.height * resize_ratio))
            resized_image = image.resize(new_size, Image.Resampling.BICUBIC)

            output = io.BytesIO()
            resized_image.save(output, format="JPEG", quality=80)  
            resized_image_data = output.getvalue()
            resized_size = len(resized_image_data)

            if resized_size < 7.5 * 1024 * 1024:
                print(f"图像已压缩至 {resized_size / (1024 * 1024):.2f} MB")
                break
            resize_ratio *= 0.9  

        base64_image = base64.b64encode(resized_image_data).decode("utf-8")
    else:
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode("utf-8")

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1" 
        )

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': prompt},
                        {'type': 'image_url', 'image_url': {'url': f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            stream=False,  
            max_tokens=1024
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"{e}")
        print("https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
        return None

def merge_masks_by_label(file_path, input_numbers, output_dir):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
        
    for number in input_numbers:
        if str(number) not in json_data:
            raise ValueError(f"Number {number} doesn't exsit!!")

    label_groups = {}
    for number in input_numbers:
        info = json_data[str(number)]
        label = info["semantic label"]
        if label not in label_groups:
            label_groups[label] = []
        label_groups[label].append(info["path of mask"])

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    result = {}
    for label, mask_paths in label_groups.items():
        if mask_paths:
            merged_mask = np.array(Image.open(mask_paths[0]).convert("L"))
            
            for mask_path in mask_paths[1:]:
                mask = np.array(Image.open(mask_path).convert("L"))
                merged_mask = np.maximum(merged_mask, mask)  

            merged_mask_image = Image.fromarray(merged_mask.astype(np.uint8))
            merged_mask_image.save(output_path)
            
            result[label] = output_path

    return result
   

def load_digital_twin(json_path):
    with open(json_path, 'r') as f:
        digital_twin = json.load(f)
    return digital_twin

def call_llm_result(instruction, digital_twin):
    prompt = ""
    prompt += f"You will be given an instruction about image editing and the digital twin information of the edited image."
    prompt += f"Your first task is to determine which type of editing the instruction belongs to."
    prompt += f" If the instruction is about remove something, please response the number \'1\';\n"
    prompt += f" If the instruction is about change or replace something, please response the number \'2\';\n"
    prompt += f" If the instruction is about add something, please response the number \'3\'.\n"
    prompt += f"Please note that you only need to response a number from 1 to 3 to finish the first task.\n \n"
    prompt += f"Your second task is to find the mask area that should be edited by the instruction. "
    prompt += f"You should only answer a list of number in the form [number1, number2, ...] to finish this task."
    prompt += f"If you find no mask area provided in the digital twin information should be edited, you can randomly select one."
    prompt += f"You can not answer a null list. \n"
    prompt += f"Your third task is to simplify the instruction."
    prompt += f"You should only answer the object that should be draw in the mask area you choose from the second task."
    prompt += f"If your answer to the first task is 1, you can answer null for this task. \n \n"
    prompt += f"The instruction is: {instruction} \n"
    prompt += f"The digital twin information is: \n"
    for key, info in digital_twin.items():
        prompt += f"Object {key}: \"semantic label\": {info['semantic label']}, \"path of mask\": {info['path of mask']}, \"depth\": {info['depth']}\n"

    prompt += f"After you finish all the three task above, you should give all the answer in the form: \n"
    prompt += f"The answer for task one is: <an integer from 1 to 3>; "
    prompt += f"The answer for task two is: <[number1, number 2, ...]>"
    prompt += f"The answer for task three is: <the object that should be draw>;"
    answer = call_llm(prompt=prompt, api_key='1', model_name='1')
    return answer

def edit_image(edit_type, image_path, mask_path, save_path, text_prompt, device):
    text_prompt = remove_draw_prefix(text_prompt)
    save_path_return_list = []
    if edit_type == '1':
        for i in range(3):
            edited_save_path = remove_main(input_img_path=image_path, mask_path=mask_path, output_dir=save_path[i], device=device)
            save_path_return_list.append(edited_save_path)
        for i in range(3):
            edited_save_path = inpaint_pipeline(image_path=image_path, mask_path=mask_path, save_path=save_path[i+3], text_prompt='white wall', device=device)
            save_path_return_list.append(edited_save_path)
    elif edit_type == '2':
        for i in range(3):
            edited_save_path = replace_main(input_img_path=image_path, mask_path=mask_path, text_prompt=text_prompt, output_dir=save_path[i], device=device)
            save_path_return_list.append(edited_save_path)
        for i in range(3):
            edited_save_path = inpaint_pipeline(image_path=image_path, mask_path=mask_path, save_path=save_path[i+3], text_prompt=text_prompt, device=device)
            save_path_return_list.append(edited_save_path)
    else:
        # print("this is an add type")
        for i in range(3):
            edited_save_path = replace_main(input_img_path=image_path, mask_path=mask_path, text_prompt=text_prompt, output_dir=save_path[i], device=device)
            save_path_return_list.append(edited_save_path)
        for i in range(3):
            edited_save_path = inpaint_pipeline(image_path=image_path, mask_path=mask_path, save_path=save_path[i+3], text_prompt=text_prompt, device=device)
            save_path_return_list.append(edited_save_path)
    return save_path_return_list

def parse_answers(input_string):

    parts = [part.strip().rstrip(';') for part in input_string.split(';') if part.strip()]
    
    task_one_part = parts[0].split(':')[-1].strip()
    task_one = int(task_one_part)
    
    task_two_part = parts[1].split(':')[-1].strip().lstrip('[').rstrip(']')
    task_two = [int(num.strip()) for num in task_two_part.split(',')]
    
    task_three_part = parts[2].split(':')[-1].strip()
    task_three = task_three_part
    
    return task_one, task_two, task_three

def single_reasoning_chain(image_path, text_prompt, instruction, api_key,
                           sam_result_path, mask_path, dt_path, edited_path,
                           result_edit_mask_path,
                           model_name,
                           device):

    time_all = 0.0

    json_path = generate_json(image_path=image_path, text_prompt=text_prompt, 
                                sam_result_path=sam_result_path, mask_output_path=mask_path, dt_path=dt_path,
                                device=device)
    digital_twin = load_digital_twin(json_path)
    start_time = timer()
    answer = call_llm_result(instruction=instruction, digital_twin=digital_twin)
    end_time = timer()
    time_all = end_time - start_time
    print(answer)
    edit_type, mask_list_to_merge, edit_prompt = parse_answers(answer)
    result = merge_masks_by_label(file_path=json_path, input_numbers=mask_list_to_merge, output_dir=str(mask_path))
    edit_mask_path=list(result.values())[0]
                
    copy_image(source_path=edit_mask_path, destination_path=result_edit_mask_path)      
    save_path = edit_image(edit_type=edit_type, 
               image_path=image_path, 
               mask_path=edit_mask_path, 
               save_path=edited_path, 
               text_prompt=edit_prompt, 
               device=device)

    
    return time_all

def my_edit_lack_cot(image_path, save_path, instruction, middle_dir, device):
    chatmoss_api_key = ""
    device = device

    out_dir = middle_dir
    out_dir = Path(out_dir)
    
    sam_result_path = out_dir / 'sam_result.png'
    result_edit_mask_path = out_dir / 'result_edit_mask_path.png'
    
    pic_description = descri_img_by_qianwen(image_path=image_path)
    pic_obj_list = ast.literal_eval(pic_description)
    print(pic_obj_list)
    
    time = single_reasoning_chain(image_path=image_path, text_prompt=pic_obj_list, 
                                               instruction=instruction,
                                               api_key=chatmoss_api_key,
                                               sam_result_path=sam_result_path,
                                               mask_path=out_dir,
                                               dt_path=out_dir,
                                               result_edit_mask_path=result_edit_mask_path,
                                               edited_path=save_path,
                                               model_name='qwen-max',
                                               device=device)
    return time

   
