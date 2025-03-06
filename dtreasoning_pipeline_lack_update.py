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
import base64
import numpy as np
from pathlib import Path
import ast
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
        print(f"Error")
    except Exception as e:
        print(f"Error: {e}")

import base64
import io

def descri_img_by_qianwen(image_path, api_key, model_name):
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
                print(f"{resized_size / (1024 * 1024):.2f} MB")
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

def run_code(generated_code):
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".py", delete=False) as temp_file:
        temp_file.write(generated_code)
        temp_file_path = temp_file.name

    result = subprocess.run(["python", temp_file_path], capture_output=True, text=True)

    print("Result:", result.stdout)

    os.remove(temp_file_path)
    
    return result.stdout

def merge_masks_by_label(file_path, input_numbers, output_dir):
    with open(file_path, 'r') as file:
        json_data = json.load(file)
        
    for number in input_numbers:
        if str(number) not in json_data:
            raise ValueError(f"Number {number} doesn't exist!!")

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

def single_reasoning_chain(image_path, text_prompt, instruction, api_key,
                           sam_result_path, mask_path, dt_path, edited_path,
                           result_edit_mask_path,
                           model_name,
                           device):

    answered = False
    time_all = 0.0
    text_prompt_local = text_prompt
    edit_type = classify_instruction(instr=instruction, api_key=api_key, model_name=model_name)
    json_path = generate_json(image_path=image_path, text_prompt=text_prompt_local, 
                                sam_result_path=sam_result_path, mask_output_path=mask_path, dt_path=dt_path,
                                device=device)

    digital_twin = load_digital_twin(json_path)
    start_time = timer()
    answer_for_judge_dt = judge_dt_answer(instruction=instruction, digital_twin=digital_twin, 
                                            api_key=api_key, model_name=model_name)

    end_time = timer()
    time1 = end_time - start_time
    time_all += time1

    start_time = timer()
    mask_list_to_merge = mask_list(instrution=instruction, digital_twin=digital_twin, api_key=api_key, model_name=model_name)
    end_time = timer()
    time2 = end_time - start_time
    time_all += time2
    print(mask_list)
    result = merge_masks_by_label(file_path=json_path, input_numbers=mask_list_to_merge, output_dir=str(mask_path))
    edit_mask_path=list(result.values())[0]
    start_time = timer()
    edit_prompt = simplified_editing_prompt(edit_type=edit_type, instruction=instruction, api_key=api_key, model_name=model_name)
    end_time = timer()
    time3 = end_time - start_time
    time_all += time3
    copy_image(source_path=edit_mask_path, destination_path=result_edit_mask_path)      
    save_path = edit_image(edit_type=edit_type, 
               image_path=image_path, 
               mask_path=edit_mask_path, 
               save_path=edited_path, 
               text_prompt=edit_prompt, 
               device=device)

    
    return time_all


def reasoning_chain(image_path, text_prompt, instruction, api_key,
                    sam_result_path, mask_path, dt_path, edited_path,
                    model_name,
                    device):
    one_step = judge_instruction_one_step(instruction=instruction, api_key=api_key, model_name=model_name)
    if one_step:
        save_path = single_reasoning_chain(image_path=image_path, text_prompt=text_prompt, instruction=instruction, api_key=api_key,
                                sam_result_path=sam_result_path, mask_path=mask_path, dt_path=dt_path, edited_path=edited_path,
                                model_name=model_name, device=device)
    elif not one_step:
        instr_list = split_instruction(instruction=instruction, api_key=api_key, model_name=model_name)
        for instr in extract_bracket_content(instr_list):
            save_path = single_reasoning_chain(image_path=image_path, text_prompt=text_prompt, instruction=instr, api_key=api_key,
                                sam_result_path=sam_result_path, mask_path=mask_path, dt_path=dt_path, edited_path=edited_path,
                                model_name=model_name, device=device)
            image_path = save_path
    else:
        print("Something wrong when judge whether the instruction is a one-step instruction.")
    return save_path




def describe_image(image_path, api_key, 
                   model_name):
    prompt = f'''Your task is to list all the objects in the given image. \
Your answer should be in the form [\'object1\', \'object2\', ...]. \
Please note that if there are more than one objects with the same label, \
your should distinguish them by their positions (e.g., top-left, top-right, bottom-right, bottom-left).'''

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    # print('begin api')
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


def my_edit_lack_update(image_path, save_path, instruction, middle_dir, device):
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

   
