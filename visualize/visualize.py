import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import math
import pdb
import os

def show_anns(masks:np.ndarray, save_path=''):

    if len(masks) == 0:
        return
    # sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    index = masks.sum(axis=(0, 1))
    sorted_index = np.argsort(index)
    sorted_anns = masks[:, :, sorted_index]

    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones((sorted_anns.shape[0], sorted_anns.shape[1], 4))
    img[:,:,3] = 0
    for i in range(sorted_anns.shape[-1]):
        m = masks[:, :, i]
        print(m.sum())
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        # print(color_mask.shape, m.shape, img.shape)
        img[m] = color_mask
    ax.imshow(img)
    
    return img

def visualize_masks(masks, pure_image, path="output.png"):

    index = masks.sum(axis=(0, 1))
    sorted_index = np.argsort(index)[::-1]
    masks = masks[:, :, sorted_index]

    mask_num = masks.shape[-1]
    nrows = max(math.ceil((mask_num + 1) / 3), 2)
    fig, axs = plt.subplots(nrows, 3,  figsize=(10, 4 * nrows)) 

    for i in range(nrows * 3):
            
        ix = i // 3; iy = i % 3
        ax = axs[ix, iy]
        if (i == 0):

            ax.imshow(pure_image)
            ax.set_title(f"image")
        
        elif (i < mask_num + 1):
            mask = masks[..., i-1]

            ax.imshow(mask, cmap='gray')
            ax.set_title(f'mask {i}')
        ax.axis('off')

    plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域
    fig.savefig(path)
    # plt.show()

def visualize_mask_colored(masks, pure_image, path="output.png"):

    fig = plt.figure(figsize=(20,20))
    plt.imshow(pure_image)
    show_anns(masks[:, :, :])
    plt.axis('off')
    # plt.show() 
    fig.savefig(path)


def main_visualize(root = "data/ground_truth"):
    pass

    for filename in sorted(os.listdir(root)):
        

        ###可视化
        save_path = f"data/gt_images/{filename}.png"
        
        image = cv2.imread(os.path.join("data", "wechatAvatars", f"{filename}.jpg"))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask_array = np.load(os.path.join(root, filename))

        visualize_masks(mask_array)