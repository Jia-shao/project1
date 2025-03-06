import glob
from PIL import Image
import PIL
import json
import os
from ip2p_pipeline import ip2p_edit
import warnings
warnings.filterwarnings("ignore")



def main():
    test_dir = "./our_smart_dataset/6-Reasoning"
    output_base_path = "./our_smart_dataset/ip2p_result"
    
    if not os.path.exists(output_base_path):
        os.makedirs(output_base_path)   

    test_img_list = sorted(glob.glob(f'{test_dir}/*.png'))
    print(len(test_img_list))
    with open(test_dir + "/Reason_test.txt", 'r') as f:
        prompt = f.readlines()

    for idx, img_path in enumerate(test_img_list):
        if idx >= 0:
            text_prompt = prompt[idx]
            text_prompt = text_prompt.split("CLIP: ")[0].rstrip()
            
            output_folder_path = os.path.join(output_base_path, str(idx))
            if not os.path.exists(output_folder_path):
                os.makedirs(output_folder_path)
                
            output_image_filename = f"{idx}_ip2p_edited_image.png"
            output_image_path = os.path.join(output_folder_path, output_image_filename)
            print(img_path)
            print(output_image_path)
            print(text_prompt)
            ip2p_edit(image_path=img_path, save_path=output_image_path, instruction=text_prompt, device='cuda:1')
        



if __name__ == '__main__':
    main()