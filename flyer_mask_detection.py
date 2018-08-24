
import numpy as np
import glob, pdb, math

import os, io, libconf, copy
import cv2
from PIL import Image
import matplotlib.pyplot as plt

from utils import *
from cfg import *

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image outside of the polygon
    formed from `vertices`. The inside of the polygon is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def split_and_write_image(image_file_path):
    imgs_x4 = split_kite_vertical_images(image_file_path)
    img_name = image_file_path.split('/')[-1].split('.')[0]
    for i in range(4):
        out_image_path = '/home/jzhang/vo_data/SN86/debug_mask/out'
        out_image_path = out_image_path + '/'+ img_name + '_' + str(i) + '.jpg'
        cv2.imwrite(out_image_path, imgs_x4[i])
    return

# split_and_write_image(image_path)
# image_path = '/home/jzhang/vo_data/SN86/debug_mask/1443141.jpg'

cam0_img = '/home/jzhang/vo_data/SN86/debug_mask/out/1443141_0.jpg'

gray = cv2.imread(cam0_img, cv2.IMREAD_GRAYSCALE)
low_threshold = 10
high_threshold = 180
edges = canny(gray, low_threshold, high_threshold)

vertices1 = np.array([[ (0, 230), (130, 260), (0, 250),(130, 270)]], dtype=np.int32) 
vertices2 = np.array([[ (426, 286), (425, 290), (600, 0),(640, 115)]], dtype=np.int32) 

vertices = np.concatenate((vertices1, vertices2), axis=0)

roi = region_of_interest(edges, vertices1)


plt.imshow(edges)
# plt.imshow(roi)
plt.show()

