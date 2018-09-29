import matplotlib.pyplot as plt
import cv2
from utils import *
from cfg import *


def split_image_x4(img_file):
    num_cams = 2
    imgx4 = split_kite_vertical_images(img_file, num_cams)
    img_name = img_file.split('/')[-1]
    out_path = os.path.dirname(img_file)
    # import pdb ; pdb.set_trace()
    for camera_index in range(num_cams):
        fname = 'cam' + str(camera_index) + '_' + img_name
        f_path = os.path.join(out_path, fname)
        cv2.imwrite(f_path, imgx4[camera_index])


def rectify_fisheye_camera_pairs(cam0_img, cam1_img, K0, K1, D0, D1, R, T, img_size = (640, 480)):    
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(K0, D0, K1, D1, img_size, R, T, 0)
    left_maps = cv2.fisheye.initUndistortRectifyMap(K0, D0, R1, P1, img_size, cv2.CV_16SC2)
    right_maps = cv2.fisheye.initUndistortRectifyMap(K1, D1, R2, P2, img_size, cv2.CV_16SC2)
    left_img_remap = cv2.remap(cam0_img, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4)
    right_img_remap = cv2.remap(cam1_img, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4)
    return left_img_remap, right_img_remap, P1, P2

# split_image_x4('/home/jzhang/pyego/data2/p0.jpg')
# split_image_x4('/home/jzhang/pyego/data2/p1.jpg')

# import pdb; pdb.set_trace()

img0_file = '/home/jzhang/pyego/data2/cam0_p1.jpg'
img1_file = '/home/jzhang/pyego/data2/cam1_p1.jpg'
img3_file = '/home/jzhang/pyego/data2/cam0_p0.jpg'

img0 = cv2.imread(img0_file, 0)
img1 = cv2.imread(img1_file, 0)
img3 = cv2.imread(img3_file, 0)

CALIB_PATH = '/home/jzhang/vo_data/jzhang_R80/nav_calib.cfg'

mtx, dist, rot, trans, _, _ = load_kite_config(CALIB_PATH)
rot_01, trans_01 = rot[1], trans[1]
rot_10, trans_10 = invert_RT(rot_01, trans_01)

K1 = mtx[0]
D1 = dist[0][0:4]
K2 = mtx[1]
D2 = dist[1][0:4]
R = rot_01
t = trans_01



# img0_rec, img1_rec, P1, P2 = rectify_fisheye_camera_pairs(img0, img1, K1, K2, D1, D2, R, t)
# feature_params = dict( maxCorners = 16,
#                     qualityLevel = 0.1,
#                     minDistance = 12,
#                     blockSize = 7)

# key_points = cv2.goodFeaturesToTrack(img1_rec, **feature_params)
# frame_bgr = cv2.cvtColor(img1_rec, cv2.COLOR_GRAY2BGR)

# for kpt in key_points:
#     x, y = (int(kpt[0][0]), int(kpt[0][1]))
#     color = tuple(np.random.randint(0,255,3).tolist())
#     cv2.circle(frame_bgr,(x, y), 6, color,2)

# plt.imshow(frame_bgr)
# plt.show()
# import pdb ; pdb.set_trace()


# left_kpts=np.array([202.0, 219.0])
# right_kpts=np.array([335.0, 219.0])
# scene_pts = cv2.triangulatePoints(P1, P2, left_kpts.T, right_kpts.T).T
# pts0 = scene_pts[:,0:3] / scene_pts[:,3][:,np.newaxis]
# pts0 = pts0.reshape(3, 1)
# pts1 = np.dot(rot_01, pts0) + trans_01


# img0_rec = cv2.cvtColor(img0_rec, cv2.COLOR_GRAY2BGR)
# img1_rec = cv2.cvtColor(img1_rec, cv2.COLOR_GRAY2BGR)
# plt.imshow(concat_images(img0_rec, img1_rec))
# plt.show()
# import pdb; pdb.set_trace()

# pts0
# array([[-0.14065892],
#        [-0.02625641],
#        [ 0.32809965]])
# pts1
# array([[ 0.0241382 ],
#        [-0.02964257],
#        [ 0.33361715]])

K0 = correct_kite_camera_matrix(mtx[0], dist[0][0:4])
K1 = correct_kite_camera_matrix(mtx[1], dist[1][0:4])
construct_projection_mtx(K0, K1, R, t)

img0 = undistort_kite_image(img0, mtx[0], K0, dist[0][0:4])
img1 = undistort_kite_image(img1, mtx[1], K1, dist[1][0:4])
img3 = undistort_kite_image(img3, mtx[0], K0, dist[0][0:4])

feature_params = dict( maxCorners = 32,
                    qualityLevel = 0.05,
                    minDistance = 8,
                    blockSize = 7)

key_points = cv2.goodFeaturesToTrack(img3, **feature_params)
frame_bgr = cv2.cvtColor(img3, cv2.COLOR_GRAY2BGR)

for kpt in key_points:
    x, y = (int(kpt[0][0]), int(kpt[0][1]))
    color = tuple(np.random.randint(0,255,3).tolist())
    cv2.circle(frame_bgr,(x, y), 6, color,2)

# print(key_points)
# plt.imshow(frame_bgr)
# plt.show()

kpt0 = np.array([304.0, 229.0, 1]).reshape(3, 1)
kpt1 = np.array([224.0, 180.0, 1]).reshape(3, 1)
kpt3 = np.array([412.0, 227.0, 1]).reshape(3, 1)

sd_card_cam0   = np.array([304.0, 229.0])
sd_card_cam1   = np.array([224.0, 180.0])
sd_card_cam3   = np.array([412.0, 227.0])

pt0, _, _ = triangulate_3d_points(sd_card_cam0, sd_card_cam3, K0, K1, rot_01, trans_01)
pt2, _, _ = triangulate_3d_points(sd_card_cam0, sd_card_cam1, K0, K1, rot_01, trans_01)

pt0 = pt0.reshape(3,1)
pt1 = np.dot(rot_01, pt0) + trans_01


import pdb; pdb.set_trace()
# project the point to cam0 plane
cam0_projection = np.dot(K0, pt0)
cam0_projection /= cam0_projection[2]

# the actual measurement is 31.7, so there is around 0.8cm measurement error 
sd_card_depth1 = 0.325 # meter

point1 = np.dot(inv(K1), sd_card_pos1) * sd_card_depth1
point0 = np.dot(rot_10, point1) + trans_10 
projection0 = np.dot(K0, point0)
projection0 /= projection0[2]
print('projection:', projection0[0], projection0[1])
print('cam0:', sd_card_pos0[0], sd_card_pos0[1])

# proj1 = 







