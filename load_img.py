import torch
import numpy as np
from segment_anything.build_sam import sam_model_registry
from segment_anything.modeling import sam
from segment_anything.utils.transforms import ResizeLongestSide 
from PIL import Image
import os
import pdb
import cv2


def load_images_from_folder(folder):
    images = {}
    for filename in sorted(os.listdir(folder)):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image = cv2.imread(os.path.join(folder, filename))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            images[filename.replace(".jpg", "")] = (image)

            print(filename, image.shape)

    return images

def image_feature(image):

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    features = lab_image[:, :, 1:]

    return features
