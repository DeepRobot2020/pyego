import matplotlib.pyplot as plt
import cv2
from utils import *
from cfg import *

cam0_file = '/home/jzhang/pyego/cam0/cam0_dot_test_2.jpg'
cam1_file = '/home/jzhang/pyego/cam0/cam1_dot_test_2.jpg'
CALIB_PATH = '/home/jzhang/vo_data/jzhang_R80/nav_calib.cfg'

mtx, dist, rot, trans, _, _ = load_kite_config(CALIB_PATH)

K0 = correct_kite_camera_matrix(mtx[0], dist[0][0:4])
K1 = correct_kite_camera_matrix(mtx[1], dist[1][0:4])

rot_01, trans_01 = rot[1], trans[1]
rot_10, trans_10 = invert_RT(rot_01, trans_01)


img0 = cv2.imread(cam0_file, 0)
img1 = cv2.imread(cam1_file, 0)

img0 = undistort_kite_image(img0, mtx[0], K0, dist[0][0:4])
img1 = undistort_kite_image(img1, mtx[1], K1, dist[1][0:4])

feature_params = dict( maxCorners = 16,
                    qualityLevel = 0.1,
                    minDistance = 12,
                    blockSize = 7)

key_points = cv2.goodFeaturesToTrack(img1, **feature_params)
frame_bgr = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

for kpt in key_points:
    x, y = (int(kpt[0][0]), int(kpt[0][1]))
    color = tuple(np.random.randint(0,255,3).tolist())
    cv2.circle(frame_bgr,(x, y), 6, color,2)

# plt.imshow(frame_bgr)
# plt.show()

sd_card_pos0   = np.array([206.833, 224.583, 1]).reshape(3, 1)
sd_card_pos1   = np.array([316.627, 220.328, 1]).reshape(3, 1)

# the actual measurement is 31.7, so there is around 0.8cm measurement error 
sd_card_depth1 = 0.325 # meter

point1 = np.dot(inv(K1), sd_card_pos1) * sd_card_depth1
point0 = np.dot(rot_10, point1) + trans_10 
projection0 = np.dot(K0, point0)
projection0 /= projection0[2]
print('projection:', projection0[0], projection0[1])
print('cam0:', sd_card_pos0[0], sd_card_pos0[1])

# proj1 = 







