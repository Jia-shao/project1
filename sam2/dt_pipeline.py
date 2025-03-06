from sam2.mask_pipeline import sam_mask

# 调用sam，生成mask图像，函数返回path的list
mask_list = sam_mask(image_path='/home/wangyj/digital_twin/000000001650.jpg',
             save_path='/home/wangyj/digital_twin/result.png')
print(mask_list)