import matplotlib.pyplot as plt

from utils import *
from cfg import *

camera_index = 3
cam0_img = '/home/jzhang/vo_data/SN86/debug_mask/out/1443141_{}.jpg'.format(camera_index)
gray = cv2.imread(cam0_img, cv2.IMREAD_GRAYSCALE)
roi_mask = region_of_interest_mask(gray.shape, KITE_MASK_VERTICES[camera_index])
roi = apply_mask_image(gray, roi_mask)
plt.imshow(cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR))
plt.show()

