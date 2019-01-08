import numpy as np
import cv2
import glob

import matplotlib.pyplot as plt
from cfg import *


def detect_digit_contours(gray, threshold=200, max_area=48, roi=None):
    ''' Detect the digits from input grayscale image
    '''
    if roi is not None:
        try:
            (x, y, w, h) = roi
            gray = gray[y:y+h, x:x+w]
        except:
            print('extract roi failed', roi)
    ret, img_thresh = cv2.threshold(gray, 120, 1, cv2.THRESH_BINARY)
    _, contours, _= cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the boundbing boxes of each digit
    digits_bb = []
    for c in contours:
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        if w >= 2 and (h >= 3 and h <= 15):
            digits_bb.append((x, y, w, h))
            max_area = max(max_area, w * h)
    # Sort the digits bb based on the x location (left to right)
    digits_bb = sorted(digits_bb, key=lambda x: x[0])
    # Convert the image pixel inside each bb into a feature vector
    digits_codebooks = []
    for roi in digits_bb:
        # compute the bounding box of the contour
        (x, y, w, h) = roi
        roi = img_thresh[y:y + h, x:x + w]
        code = np.zeros(max_area, dtype=np.uint8)
        binary_roi = roi.ravel()
        code[0:len(binary_roi)] = binary_roi
        digits_codebooks.append(code)
    return max_area, digits_codebooks, digits_bb



class DigitsDetect:
    def __init__(self, max_area=64):
        self.digits_codebooks = None
        self.longest_code = max_area
        self.build_codebook()
        

    def build_codebook(self):
        text_dict='0123456789'
        gray = np.zeros([128, 128], dtype=np.uint8)
        gray = cv2.putText(
            gray, text_dict, (0, gray.shape[0]/2), cv2.FONT_HERSHEY_DUPLEX, 
            0.3, (255,255,255), 1, cv2.LINE_8, False)
        self.longest_code, self.digits_codebooks, _= detect_digit_contours(gray, max_area=self.longest_code)

    def decode_to_digit(self, code):
        digit = -1
        if len(code) != self.longest_code:
            code = code[0:self.longest_code]
        for i, dcb in enumerate(self.digits_codebooks):
            diff = cv2.countNonZero(cv2.bitwise_xor(dcb, code))
            if diff  < 2:
                return i
        return -1

    def detect(self, gray, roi=None):
        max_len, codes, bb = detect_digit_contours(gray, max_area=self.longest_code)
        # import pdb; pdb.set_trace()
        # assert(max_len <= self.longest_code)
        digits = []
        for c in codes:
            d = self.decode_to_digit(c)
            if d < 0:
                continue
            digits.append(self.decode_to_digit(c))
        # Convert to a interger 
        num = 0
        for d in digits:
            num = num * 10 + d
        return num






# IMG_FILE='/home/jzhang/vo_data/SR80_901020874/2018-11-01/Seq7/287.jpg'
# DIGIT_POS = (vertices_large_0[0][0], vertices_large_0[0][1] + 80)
# MAX_TEXT_H = 10

# image = cv2.imread(IMG_FILE, cv2.IMREAD_GRAYSCALE)
# IMG_H, IMG_W = image.shape
# image_roi = image[DIGIT_POS[1]-MAX_TEXT_H:DIGIT_POS[1]+MAX_TEXT_H, DIGIT_POS[0]:IMG_W]


# dd = DigitsDetect()
# num = dd.detect(image_roi)
# print('number on image -> {}'.format(num))

