import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

def show_anns(anns:np.ndarray):
    if len(anns) == 0:
        return
    # sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    index = anns.sum(axis=(0, 1))
    sorted_index = np.argsort(index)
    sorted_anns = anns[sorted_index]
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0].shape[0], sorted_anns[0].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

