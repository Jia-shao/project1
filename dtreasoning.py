from dtreasoning_prompt import split_instruction, judge_instruction_one_step, extract_bracket_content, find_question, find_semantic_label, find_position, need_which_infor, simplified_editing_prompt, call_llm, judge_dt_answer, check_first_word_is_no, check_first_word_is_yes, classify_instruction, trans_prompt, mask_list

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


def merge_masks_by_label(file_path, input_numbers, output_dir):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
        
    for number in input_numbers:
        if str(number) not in json_data:
            raise ValueError(f"Error")

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

            output_path = os.path.join(output_dir, f"merged_{label}.png")
            merged_mask_image = Image.fromarray(merged_mask.astype(np.uint8))
            merged_mask_image.save(output_path)
            
            result[label] = output_path

    return result

def load_digital_twin(json_path):
    with open(json_path, 'r') as f:
        digital_twin = json.load(f)
    return digital_twin


def edit_image(edit_type, image_path, mask_path, save_path, text_prompt, device):
    if edit_type == '1':
        save_path = remove_main(input_img_path=image_path, mask_path=mask_path, output_dir=save_path, device=device)
    elif edit_type == '2':
        save_path = replace_main(input_img_path=image_path, mask_path=mask_path, text_prompt=text_prompt, output_dir=save_path, device=device)
    else:
        print("this is an add type")
        save_path = ''
        pass
    return save_path

def single_reasoning_chain(image_path, text_prompt, instruction, api_key,
                           sam_result_path, mask_path, dt_path, edited_path,
                           model_name,
                           device):

    answered = False
    text_prompt_local = text_prompt
    edit_type = classify_instruction(instr=instruction, api_key=api_key, model_name=model_name)
    while not answered:
        json_path = generate_json(image_path=image_path, text_prompt=text_prompt_local, 
                                  sam_result_path=sam_result_path, mask_output_path=mask_path, dt_path=dt_path,
                                  device=device)
        digital_twin = load_digital_twin(json_path)
        answer_for_judge_dt = judge_dt_answer(instruction=instruction, digital_twin=digital_twin, 
                                              api_key=api_key, model_name=model_name)
        if answer_for_judge_dt:
            print("The Digital Twin Information above is enough to answer the question.")
            answered = True
            mask_list_to_merge = mask_list(instrution=instruction, digital_twin=digital_twin, api_key=api_key, model_name=model_name)
            result = merge_masks_by_label(file_path=json_path, input_numbers=mask_list_to_merge, output_dir=str(mask_path))
            edit_prompt = simplified_editing_prompt(edit_type=edit_type, instruction=instruction, api_key=api_key, model_name=model_name)
            
        else:
            print("The Digital Twin Information above is not enough to answer the question.")
            need_info = need_which_infor(instruction=instruction, digital_twin=digital_twin, api_key=api_key, model_name=model_name)
            print(need_info)
            if need_info == '1':
                semantic_info = find_semantic_label(instruction=instruction, digital_twin=digital_twin, api_key=api_key, model_name=model_name)
                for item in extract_bracket_content(semantic_info):
                    print(item)
                    text_prompt_local.append(item)
            elif need_info == '2':
                position_code = find_position(instruction=instruction, digital_twin=digital_twin, api_key=api_key, model_name=model_name)
                print(position_code)
            else:
                question_to_be_answered = find_question(instruction=instruction, digital_twin=digital_twin, api_key=api_key, model_name=model_name)
                print(question_to_be_answered)        
    save_path = edit_image(edit_type=edit_type, 
               image_path=image_path, 
               mask_path=list(result.values())[0], 
               save_path=edited_path, 
               text_prompt=edit_prompt, 
               device=device)   
    
    return save_path

def reasoning_chain(image_path, text_prompt, question, api_key,
                    sam_result_path, mask_path, dt_path, edited_path,
                    model_name,
                    device):
    one_step = judge_instruction_one_step(question=question, api_key=api_key, model_name=model_name)
    if check_first_word_is_yes(one_step):
        save_path = single_reasoning_chain(image_path=image_path, text_prompt=text_prompt, instruction=question, api_key=api_key,
                                sam_result_path=sam_result_path, mask_path=mask_path, dt_path=dt_path, edited_path=edited_path,
                                model_name=model_name, device=device)
    elif check_first_word_is_no(one_step):
        instr_list = split_instruction(instruction=question, api_key=api_key, model_name=model_name)
        for instr in extract_bracket_content(instr_list):
            save_path = single_reasoning_chain(image_path=image_path, text_prompt=text_prompt, instruction=instr, api_key=api_key,
                                sam_result_path=sam_result_path, mask_path=mask_path, dt_path=dt_path, edited_path=edited_path,
                                model_name=model_name, device=device)
            image_path = save_path
    else:
        print("Something wrong when judge whether the instruction is a one-step instruction.")
    return save_path

def describe_image(image_path, api_key, 
                   model_name="gpt4o-mini"):
    prompt = f'''Your task is to list all the objects in the given image. \
Your answer should be in the form [\'object1\', \'object2\', ...]. \
Please note that if there are more than one objects with the same label, \
your should distinguish them by their positions (e.g., top-left, top-right, bottom-right, bottom-left).'''

    """将图片转换为Base64编码"""
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.aihao123.cn/luomacode-api/open-api/v1"  
    )
    response = client.chat.completions.create(
        messages=[
            {
                'role': 'user', 
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
            },
        ],
        model=model_name,
        stream=False,
        max_tokens=100
    )
    return response.choices[0].message.content
