import cv2, os
import numpy as np
from utils import *

IMG_PATH = '/tmp/kite/images'
IMG_OUT = '/tmp/pyego/images'

imgs = get_image_files(IMG_PATH)

clahe = cv2.createCLAHE(clipLimit=120.0, tileGridSize=(8, 8))

for n, img in enumerate(imgs):
    if n < 1000:
        continue
    im = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
    # import pdb; pdb.set_trace()
    cl1 = clahe.apply(im)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    cl1 = cv2.cvtColor(cl1, cv2.COLOR_GRAY2BGR)
    im = concat_images(im, cl1)
    fname = os.path.join(IMG_OUT, str(n) + '.jpg')
    cv2.imwrite(fname, im)




