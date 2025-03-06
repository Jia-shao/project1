from diffusers import StableDiffusionInpaintPipeline
import torch
from PIL import Image

def inpaint_pipeline(image_path, mask_path, save_path, text_prompt, device):
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "/stable-diffusion/stable-diffusion-v1-5-inpainting",
        variant="fp16",  
        torch_dtype=torch.float32, 
    )
    pipe.to(device)
    original_image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")  
    image = pipe(prompt=text_prompt, image=original_image, mask_image=mask).images[0]
    image.save(save_path)
    return save_path
    
