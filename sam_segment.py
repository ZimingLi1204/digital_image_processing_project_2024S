import torch
import numpy as np
from segment_anything.build_sam import sam_model_registry
from segment_anything.modeling import sam
from segment_anything.utils.transforms import ResizeLongestSide 
from PIL import Image
import os
import pdb

class My_sam_model:
    def __init__(self) -> None:
        model_type = "vit_b"
        ckpt = "./segment_anything/sam_vit_b.pth"
        self.sam_model = sam_model_registry[model_type](checkpoint=ckpt)
        # self.transform = ResizeLongestSide(1024)
        self.load_images()

    def load_images(self, root = "data/tensors"):
        self.images = {}
        for filename in os.listdir(root):
            # if filename.endswith(".jpg") or filename.endswith(".png"):
                # img = Image.open(os.path.join(folder, filename))
                # if img is not None:
                    
            img_tensor = torch.load(os.path.join(root, filename))
            self.images[filename.replace(".pth", "")] = img_tensor
            # print(img_array.shape, filename)
            

    def segment(self):


