import glob
from PIL import Image
import PIL
import json
import os
from dtreasoning_pipeline import my_edit
from Magic_Brush.magic_brush_pipeline import magic_brush
from utils_my.metrics_evaluation import eval_single_image

import warnings
warnings.filterwarnings("ignore")

def update_json_data(file_path, key, psnr, ssim, lpips, idcs, clip):

    if not os.path.exists(file_path):
        print(f"Error: File {file_path} doesn't exist")
        return

    with open(file_path, "r") as file:
        data = json.load(file)

    if str(key) not in data:
        print(f"Error: KEY {key} doesn't exist")
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
        raise FileNotFoundError(f"File {file_path} doesn't exist")

    with open(file_path, "r") as file:
        data = json.load(file)

    if str(key) not in data:
        raise KeyError(f"Error: KEY {key} doesn't exist")

    data[str(key)]["time_all"] = float(time)

    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)


def main():
    device = 'cuda:2'
    test_dir = "./our_smart_dataset/6-Reasoning"
    output_base_path = "./our_smart_dataset/our_method"
    
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)   

    test_img_list = sorted(glob.glob(f'{test_dir}/*.png'))
    mask_image_list = sorted(glob.glob(f'{test_dir}/*_mask.jpg'))
    
    with open(test_dir + "/Reason_test.txt", 'r') as f:
        prompt = f.readlines()

    for idx, img_path in enumerate(test_img_list):
        if idx >= 21:
            text_prompt = prompt[idx]
            print(text_prompt)
            edit_instruction = text_prompt.split("CLIP: ")[0].rstrip()
            print(edit_instruction)
            CLIP_text = text_prompt.split("CLIP: ")[1].replace('\n', '')
            print(CLIP_text)
            
            output_folder_path = os.path.join(output_base_path, str(idx))
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)

            output_image_filename1 = f"{idx}_our_edited_image_inpaint_anything{1}.png"
            output_image_filename2 = f"{idx}_our_edited_image_inpaint_anything{2}.png"
            output_image_filename3 = f"{idx}_our_edited_image_inpaint_anything{3}.png"
            output_image_filename4 = f"{idx}_our_edited_image_inpaint_pipeline{1}.png"
            output_image_filename5 = f"{idx}_our_edited_image_inpaint_pipeline{2}.png"
            output_image_filename6 = f"{idx}_our_edited_image_inpaint_pipeline{3}.png"
            output_image_path1 = os.path.join(output_folder_path, output_image_filename1)
            output_image_path2 = os.path.join(output_folder_path, output_image_filename2)
            output_image_path3 = os.path.join(output_folder_path, output_image_filename3)
            output_image_path4 = os.path.join(output_folder_path, output_image_filename4)
            output_image_path5 = os.path.join(output_folder_path, output_image_filename5)
            output_image_path6 = os.path.join(output_folder_path, output_image_filename6)
            output_image_path = [output_image_path1, output_image_path2, output_image_path3, 
                                 output_image_path4, output_image_path5, output_image_path6]
                

            middle_dir = os.path.join("./our_benchmark_new/our_result_middle", str(idx))
            if not os.path.exists(middle_dir):
                os.makedirs(middle_dir)
            
            try:    
                time_llm = my_edit(image_path=img_path, save_path=output_image_path, instruction=edit_instruction, 
                                middle_dir=middle_dir, device=device)

                psnr1, ssim1, lpips1, llm_score1, clip1 = 0, 0, 0, 0, 0
                for i in range(3):
                    psnr, ssim, lpips, llm_score, clip = eval_single_image(original_image_path=img_path, 
                                                                    edited_image_path=output_image_path[i], 
                                                                    mask_path=mask_image_list[idx], instruction=edit_instruction, 
                                                                    clip_text=CLIP_text,
                                                                    device=device)
                    if i == 0:
                        update_json_data(file_path='./our_smart_dataset/inpaint_anything_result_only1.json',
                                        key=idx,
                                        psnr=psnr, ssim=ssim, lpips=lpips, idcs=llm_score, clip=clip)
                    if clip > clip1:
                        llm_score1 = llm_score
                        psnr1 = psnr
                        ssim1 = ssim
                        lpips1 = lpips
                        clip1 = clip
                update_json_data(file_path='./our_smart_dataset/inpaint_anything_result.json',
                                key=idx,
                                psnr=psnr1, ssim=ssim1, lpips=lpips1, idcs=llm_score1, clip=clip1)
                
                psnr1, ssim1, lpips1, llm_score1, clip1 = 0, 0, 0, 0, 0
                for i in range(3):
                    psnr, ssim, lpips, llm_score, clip = eval_single_image(original_image_path=img_path, 
                                                                    edited_image_path=output_image_path[i+3], 
                                                                    mask_path=mask_image_list[idx], 
                                                                    instruction=edit_instruction, 
                                                                    clip_text=CLIP_text,device=device)
                    if i==0:
                        update_json_data(file_path='./our_smart_dataset/inpaint_pipeline_result_only1.json',
                                        key=idx, 
                                        psnr=psnr, ssim=ssim, lpips=lpips, idcs=llm_score, clip=clip)
                    if clip > clip1:
                        llm_score1 = llm_score
                        psnr1 = psnr
                        ssim1 = ssim
                        lpips1 = lpips
                        clip1 = clip
                update_json_data(file_path="./our_smart_dataset/inpaint_pipeline_result.json",
                                    key=idx,
                                    psnr=psnr1, ssim=ssim1, lpips=lpips1, idcs=llm_score1, clip=clip1)
                update_time_json_file(file_path='./our_abaltion/all_module/smart_time.json',
                                key=idx,
                                time=time_llm)        
            except Exception as e:

                continue

if __name__ == '__main__':
    main()