import torch
import numpy as np
from segment_anything.build_sam import sam_model_registry
from segment_anything.modeling import sam
from segment_anything.utils.transforms import ResizeLongestSide 
from segment_anything import SamAutomaticMaskGenerator
from load_img import load_images_from_folder
from PIL import Image
import os
import pdb

class My_sam_model:
    def __init__(self, device='cpu') -> None:
        model_type = "vit_h"
        ckpt = f"./segment_anything/weights/sam_{model_type}.pth"
        self.sam_model = sam_model_registry[model_type](checkpoint=ckpt).to(device)
        self.device = device
        self.transform = ResizeLongestSide(1024)

        self.auto_predictor = SamAutomaticMaskGenerator(self.sam_model)

    def single_img(self, img0, *args: torch.Any, **kwds: torch.Any) -> torch.Any:

        if (len(img0.shape) == 2):
            rgb_image = np.expand_dims(img0, axis=2)
            rgb_image = np.repeat(rgb_image, 3, axis=2)
            img0 = rgb_image


        # 
        # img0 = torch.tensor(img0)

        # img0 = self.transform.apply_image_torch(img0.unsqueeze(0))

        # # pdb.set_trace()
        # input_img = self.sam_model.preprocess(img0)

        # input_img = input_img.squeeze().numpy()
        # input_img = np.transpose(input_img, (1, 2, 0))

        return img0            

    def segment(self, images):
        
        save_root = "data/results"
        if not os.path.exists(save_root):
            os.makedirs(save_root)

        for filename, img_array in images.items():
            # 保存预处理tensor

            img_tensor = self.single_img(img_array)
            # pdb.set_trace()

            masks = self.auto_predictor.generate(img_tensor)

            ###处理masks
            mask_array = []
            for mask in masks:
                mask_array.append(np.expand_dims(mask["segmentation"], axis=2))

            mask_array = np.concatenate(mask_array, axis=-1)


            # pdb.set_trace()

            ### 最后添加background
            ground_mask = ~mask_array.any(axis=2)
            mask_array = np.concatenate([mask_array, np.expand_dims(ground_mask, axis=2)], axis=-1)

            np.save(os.path.join(save_root, filename + ".npy"), mask_array)



def annote_data(root = "data/wechatAvatars"):
    
    images = load_images_from_folder(root)

    sam_model = My_sam_model()

    sam_model.segment(images)

annote_data()