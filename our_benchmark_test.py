import json
import os
from dtreasoning_pipeline import my_edit
from Magic_Brush.magic_brush_pipeline import magic_brush
from resize_test import resize_image_and_mask
from utils_my.metrics_evaluation import eval_single_image
import warnings
warnings.filterwarnings("ignore")
import json
import os

def update_json_data(file_path, key, psnr, ssim, lpips, idcs, clip):

    if not os.path.exists(file_path):
        return

    with open(file_path, "r") as file:
        data = json.load(file)

    if str(key) not in data:
        return

    data[str(key)]["PSNR"] = psnr
    data[str(key)]["SSIM"] = ssim
    data[str(key)]["LPIPS"] = lpips
    data[str(key)]["IDCS"] = idcs
    data[str(key)]["CLIP"] = clip
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def update_time_json_file(file_path, key, time):

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} doesn't exit")

    with open(file_path, "r") as file:
        data = json.load(file)

    if str(key) not in data:
        raise KeyError(f"KEY {key} doesn't exist!")

    data[str(key)]["time_all"] = float(time)

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)

def save_list_to_txt(data_list, file_path):

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as file:
            for item in data_list:
                file.write(str(item) + "\n")  

    except Exception as e:
        print(f"{e}")

def load_json_data(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def process_image_dataset(json_file_path, output_base_path):
    device='cuda:1'
    inpaint_anything_result_list = []
    
    failed_key_list = []
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)
    
    dataset = load_json_data(json_file_path)
    
    for key, value in dataset.items():
        if int(key) >= 0:
            input_image_path = value.get('image path')
            edit_instructions = value.get('instruction')
            
            mask_path_list = value.get('mask path')
            clip_text = value.get('CLIP')

            
            if not input_image_path or not edit_instructions:
                
                continue
            
            output_folder_path = os.path.join(output_base_path, key)
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
            
            output_image_filename1 = f"{key}_our_edited_image_inpaint_anything{1}.png"
            output_image_filename2 = f"{key}_our_edited_image_inpaint_anything{2}.png"
            output_image_filename3 = f"{key}_our_edited_image_inpaint_anything{3}.png"
            output_image_filename4 = f"{key}_our_edited_image_inpaint_pipeline{1}.png"
            output_image_filename5 = f"{key}_our_edited_image_inpaint_pipeline{2}.png"
            output_image_filename6 = f"{key}_our_edited_image_inpaint_pipeline{3}.png"
            output_image_path1 = os.path.join(output_folder_path, output_image_filename1)
            output_image_path2 = os.path.join(output_folder_path, output_image_filename2)
            output_image_path3 = os.path.join(output_folder_path, output_image_filename3)
            output_image_path4 = os.path.join(output_folder_path, output_image_filename4)
            output_image_path5 = os.path.join(output_folder_path, output_image_filename5)
            output_image_path6 = os.path.join(output_folder_path, output_image_filename6)
            output_image_path = [output_image_path1, output_image_path2, output_image_path3, output_image_path4, output_image_path5, output_image_path6]
            
            middle_dir = os.path.join("./lack_cot/our_result_middle", key)
            if not os.path.exists(middle_dir):
                os.makedirs(middle_dir)
            try:
                time_llm = my_edit(image_path=input_image_path, save_path=output_image_path, 
                                   instruction=edit_instructions, middle_dir=middle_dir, device=device)

                psnr1, ssim1, lpips1, llm_score1, clip1 = 0, 0, 0, 0, 0
                for i in range(3):
                    psnr, ssim, lpips, llm_score, clip = eval_single_image(original_image_path=input_image_path, 
                                                                           edited_image_path=output_image_path[i], 
                                                                           mask_path=mask_path_list[0], 
                                                                           instruction=edit_instructions, 
                                                                           clip_text=clip_text, device=device)
                    if i == 0:
                        update_json_data(file_path='./inpaint_anything_result_only1.json',
                                        key=key,
                                        psnr=psnr, ssim=ssim, lpips=lpips, idcs=llm_score, clip=clip)                
                    if llm_score > llm_score1:
                        llm_score1 = llm_score
                        psnr1 = psnr
                        ssim1 = ssim
                        lpips1 = lpips
                        clip1 = clip
                update_json_data(file_path='./inpaint_anything_result.json',
                                key=key,
                                psnr=psnr1, ssim=ssim1, lpips=lpips1, idcs=llm_score1, clip=clip1)
                psnr1, ssim1, lpips1, llm_score1, clip1 = 0, 0, 0, 0, 0
                
                for i in range(3):
                    psnr, ssim, lpips, llm_score, clip = eval_single_image(original_image_path=input_image_path, 
                                                                           edited_image_path=output_image_path[i+3], 
                                                                           mask_path=mask_path_list[0], 
                                                                           instruction=edit_instructions, 
                                                                           clip_text=clip_text,
                                                                           device=device)
                    if i==0:
                        update_json_data(file_path='./inpaint_pipeline_result_only1.json',
                                        key=key, 
                                        psnr=psnr, ssim=ssim, lpips=lpips, idcs=llm_score, clip=clip)
                    if llm_score > llm_score1:
                        llm_score1 = llm_score
                        psnr1 = psnr
                        ssim1 = ssim
                        lpips1 = lpips
                        clip1 = clip
                update_json_data(file_path="./inpaint_pipeline_result.json",
                                    key=key,
                                    psnr=psnr1, ssim=ssim1, lpips=lpips1, idcs=llm_score1, clip=clip1)
                update_time_json_file(file_path='./time.json',
                                key=key,
                                time=time_llm)
            except Exception as e:
                print(f"Error occurred while processing {key}: {e}")
                failed_key_list.append(key)
                save_list_to_txt(data_list=failed_key_list, file_path='./our_method_failed_case.txt')
                continue
