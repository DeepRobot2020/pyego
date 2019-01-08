import numpy as np
import cv2
from cfg import *
import matplotlib.pyplot as plt


IMG_FILE='/home/jzhang/pyego/661.jpg'
DIGIT_POS = (vertices_large_0[0][0], vertices_large_0[0][1] + 80)
MAX_TEXT_H = 10

image = cv2.imread(IMG_FILE, cv2.IMREAD_GRAYSCALE)
IMG_H, IMG_W = image.shape

image_roi = image[DIGIT_POS[1]-MAX_TEXT_H:DIGIT_POS[1]+MAX_TEXT_H, DIGIT_POS[0]:IMG_W]


# Build the digit dict
text_dict='0123456789'
image_dict = np.zeros_like(image_roi)
image_dict2 = cv2.putText(image_dict, text_dict, (0, image_dict.shape[0]/2), cv2.FONT_HERSHEY_DUPLEX, 0.3, (255,255,255), 1, cv2.LINE_8, False)

import pdb; pdb.set_trace()

_, contours_dict, _= cv2.findContours(image_dict2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

digits_bb = []
max_area = 48
for c in contours_dict:
    # compute the bounding box of the contour
    (x, y, w, h) = cv2.boundingRect(c)
    if w >= 2 and (h >= 3 and h <= 15):
        digits_bb.append((x, y, w, h))
        max_area = max(max_area, w * h)
        
digits_bb = sorted(digits_bb, key=lambda x: x[0])

digits_codebooks = []
for roi in digits_bb:
    # compute the bounding box of the contour
    (x, y, w, h) = roi
    roi = image_dict2[y:y + h, x:x + w]
    code = np.zeros(max_area)
    _, binary_roi = cv2.threshold(roi,250,1,cv2.THRESH_BINARY)
    binary_roi = binary_roi.ravel()
    code[0:len(binary_roi)] = binary_roi
    digits_codebooks.append(code)

assert(len(digits_codebooks) == 10)

ret,img_thresh = cv2.threshold(image_roi,200, 255,cv2.THRESH_BINARY)
img, contours,  hierarchy= cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

digit_cnts = []
digits = []
for c in contours:
    	# compute the bounding box of the contour
	(x, y, w, h) = cv2.boundingRect(c) 
	# if the contour is sufficiently large, it must be a digit
	if w >= 2 and (h >= 3 and h <= 15):
		digit_cnts.append(c)
        roi = img_thresh[y:y + h, x:x + w]
        _, binary_roi = cv2.threshold(roi,250,1,cv2.THRESH_BINARY)
        binary_roi = binary_roi.ravel()
        code = np.zeros(max_area)
        code[0:len(binary_roi)] = binary_roi

        ret = max_area
        pred = 0
        for i, dcb in enumerate(digits_codebooks):
            diff = cv2.bitwise_xor(dcb, code)
            diff2 = cv2.countNonZero(diff);
            if diff2 < ret:
                ret = diff2
                pred = i
        print('----------------->', pred)
        digits.append(roi)
        plt.imshow(roi, cmap='gray')
        plt.show()

        # import pdb; pdb.set_trace()

import pdb; pdb.set_trace()


