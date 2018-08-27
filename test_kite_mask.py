import matplotlib.pyplot as plt

from utils import *
from cfg import *

camera_index = 3
cam0_img = '/home/jzhang/vo_data/SN86/debug_mask/out/1443141_{}.jpg'.format(camera_index)
frame_gray = cv2.imread(cam0_img, cv2.IMREAD_GRAYSCALE)

roi_mask1 = region_of_interest_mask(frame_gray.shape, KITE_MASK_VERTICES[camera_index])
roi_mask2 = region_of_interest_mask(frame_gray.shape, KITE_MASK_VERTICES[camera_index], filler = 1)

key_points = cv2.goodFeaturesToTrack(frame_gray, mask=roi_mask2, maxCorners=128, qualityLevel=0.3, minDistance=16, blockSize=7)


frame_gray = apply_mask_image(frame_gray, roi_mask1)
frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)


# for kpt in key_points:
#     x, y = (int(kpt[0][0]), int(kpt[0][1]))
#     color = tuple(np.random.randint(0,255,3).tolist())
#     cv2.circle(frame_bgr,(x, y), 6, color,2)

plt.imshow(frame_bgr)
plt.show()

