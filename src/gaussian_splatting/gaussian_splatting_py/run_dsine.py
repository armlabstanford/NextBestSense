"""
run DSINE inference on the GS data dir
"""
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch


FILENAME = 'dsine_test.png'
GS_DATA_DIR = '/home/user/NextBestSense/data/2024-08-22-00-56-23'

def get_normals_from_img(img, normal_predictor):
    """
    get normals from an image with DSINE
    """
    print('Running inference...')
    with torch.inference_mode():
        normal = normal_predictor.infer_cv2(img)[0]
        normal = (normal + 1) / 2
    return normal


if __name__ == '__main__':
    img_dir = GS_DATA_DIR + '/images'
    normal_predictor = torch.hub.load("hugoycj/DSINE-hub", "DSINE", trust_repo=True)
    
    # loop through all images and run DSINE
    imgs_list  = os.listdir(img_dir)
    for img in imgs_list:
        if 'depth' in img:
            continue
        if 'normal' in img:
            continue
        # Load the input image using OpenCV
        image = cv2.imread(img_dir + '/' + img, cv2.IMREAD_COLOR)
        normal = get_normals_from_img(image, normal_predictor)
        # Convert the normal map to a displayable format
        normal = (normal * 255).cpu().numpy().astype(np.uint8).transpose(1, 2, 0)
        plt.imshow(normal)
        plt.show()
        
        # save normal image in same dir
        cv2.imwrite(img_dir + '/' + img.replace('.png', '_normal.png'), normal)
        
