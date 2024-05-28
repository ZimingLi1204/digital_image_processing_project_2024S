import torch
import numpy as np
from segment_anything.build_sam import sam_model_registry
from segment_anything.modeling import sam
from segment_anything.utils.transforms import ResizeLongestSide 
from PIL import Image
import os
import pdb


def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = Image.open(os.path.join(folder, filename))
            if img is not None:
                
                img_array = np.array(img)

                print(img_array.shape, filename)
                images[filename.replace(".jpg", "")] = (img_array)
    return images

class TransposeImage:
    def __init__(self) -> None:
        model_type = "vit_b"
        ckpt = "./segment_anything/sam_vit_b.pth"
        self.sam_model = sam_model_registry[model_type](checkpoint=ckpt)
        self.transform = ResizeLongestSide(1024)

    def single_img(self, img0, *args: torch.Any, **kwds: torch.Any) -> torch.Any:

        if (len(img0.shape) == 2):
            rgb_image = np.expand_dims(img0, axis=2)
            rgb_image = np.repeat(rgb_image, 3, axis=2)
            img0 = rgb_image


        img0 = np.transpose(img0, (2, 0, 1))
        img0 = torch.tensor(img0)

        img0 = self.transform.apply_image_torch(img0.unsqueeze(0))

        # pdb.set_trace()
        input_img = self.sam_model.preprocess(img0)

        # input_img = input_img.squeeze().numpy()
        # input_img = np.transpose(input_img, (1, 2, 0))

        return input_img

    def list_img(self, images):

        if not os.path.exists("data/tensors"):
            os.makedirs("data/tensors")
        
        for filename, img_array in images.items():
            # 保存预处理tensor
            # pdb.set_trace()
            img_tensor = self.single_img(img_array)
            torch.save(img_tensor, os.path.join("data/tensors", filename+".pth"))

            print(img_tensor.shape, filename)

root = "data/wechatAvatars"
images = load_images_from_folder(root)
transform = TransposeImage()
pdb.set_trace()
transform.list_img(images)

# 打印加载的图片数量
print("Loaded", len(images), "images")

