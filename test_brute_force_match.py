from utils import *
from cfg import *
import cv2
import matplotlib.pyplot as plt
roi_masks = []
roi_frame = []
num_cams = 4
image_shape = (480, 640)

pose0 = '/home/jzhang/pyego/data/pose0.jpg'
pose1 = '/home/jzhang/pyego/data/pose1.jpg'

CALIB_PATH = '/home/jzhang/vo_data/jzhang_R80/nav_calib.cfg'
DEBUG_PATH = '/home/jzhang/pyego/output'

mtx, dist, rot, trans, _, _ = load_kite_config(CALIB_PATH)

if not os.path.exists(DEBUG_PATH):
    os.makedirs(DEBUG_PATH)

imgs = []
imgs.append(pose0)
imgs.append(pose1)

K = []
D = []

# some camera level of pre-processing 
for index in range(num_cams):
    mask_kpts = region_of_interest_mask(image_shape, KITE_MASK_VERTICES[index], filler = 1)
    mask_intensy = region_of_interest_mask(image_shape, KITE_MASK_VERTICES[index])
    roi_masks.append(mask_kpts)
    roi_frame.append(mask_intensy)
    D.append(dist[index][0:4])
    K.append(correct_kite_camera_matrix(mtx[index], D[index]))


# split and undistort images
imgs_x4_raw = []
imgs_x4_undist = []

for pose in imgs:
    x4_imgs = split_kite_vertical_images(pose)
    imgs_x4_raw.append(x4_imgs)
    x4_imgs_undist = []
    for i in range(4):
        img_undist = undistort_kite_image(x4_imgs[i], mtx[i], K[i], D[i])
        x4_imgs_undist.append(img_undist)
        # import pdb ; pdb.set_trace()
        frame_gray = apply_mask_image(img_undist, roi_frame[i])
        frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
        fname = pose.split('/')[-1].split('.')[0] + '_mask_cam_' + str(i) + '.jpg'
        fname = os.path.join(DEBUG_PATH, fname)
        cv2.imwrite(fname, frame_bgr)
    imgs_x4_undist.append(x4_imgs_undist)


# write raw images into the debug folder

img0 = imgs_x4_undist[1][0]
img1 = imgs_x4_undist[0][0]
img3 = imgs_x4_undist[1][1]


sift = cv2.xfeatures2d.SIFT_create()
feature_params = dict( maxCorners = 256,
                    qualityLevel = 0.01,
                    minDistance = 8,
                    blockSize = 7)

k0 = cv2.goodFeaturesToTrack(img0, mask=roi_masks[0], **feature_params)
kp0 = []
for i in range(k0.shape[0]):
    kp = cv2.KeyPoint(float(k0[i][0][0]), float(k0[i][0][1]), 9.0)
    kp0.append(kp)


k1 = cv2.goodFeaturesToTrack(img1, mask=roi_masks[0], **feature_params)
kp1 = []
for i in range(k1.shape[0]):
    kp = cv2.KeyPoint(float(k1[i][0][0]), float(k1[i][0][1]), 9.0)
    kp1.append(kp)


k3 = cv2.goodFeaturesToTrack(img3, mask=roi_masks[1], **feature_params)
kp3 = []
for i in range(k3.shape[0]):
    kp = cv2.KeyPoint(float(k3[i][0][0]), float(k3[i][0][1]), 9.0)
    kp3.append(kp)


kp0, des0 = sift.compute(img0, kp0)
kp1, des1 = sift.compute(img1, kp1)
kp3, des3 = sift.compute(img3, kp3)


# match keypoints between 0 and 1
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)


# BFMatcher with default params
bf = cv2.BFMatcher()
matches_01 = bf.knnMatch(des0,des1, k=2)
matches_03 = bf.knnMatch(des0,des3, k=2)

# matches_01 = flann.knnMatch(des0, des1, k=2)
# matches_03 = flann.knnMatch(des0, des3, k=2)

# Need to draw only good matches, so create a mask
matchesMask_01 = [[0,0] for i in range(len(matches_01))]
# Need to draw only good matches, so create a mask
matchesMask_03 = [[0,0] for i in range(len(matches_03))]


cnt_01 = 0
# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches_01):
    if m.distance < 0.6*n.distance:
        matchesMask_01[i]= [1,0]
        cnt_01 += 1

cnt_03 = 0
for i,(m,n) in enumerate(matches_03):
    if m.distance < 0.6*n.distance:
        matchesMask_03[i]= [1,0]
        cnt_03 += 1   

print('01_match',cnt_01, '03_match', cnt_03)


# draw 01 keypoints
img0_bgr = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
img1_bgr = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

for i, m in enumerate(matches_01):
    if matchesMask_01[i][0] == 0:
        continue
    # import pdb ; pdb.set_trace()
    pt0 = kp0[m[0].queryIdx]
    pt1 = kp1[m[0].trainIdx]
    x0, y0 = (int(pt0.pt[0]), int(pt0.pt[1]))
    x1, y1 = (int(pt1.pt[0]), int(pt1.pt[1]))   
    color = tuple(np.random.randint(0,255,3).tolist())
    cv2.circle(img0_bgr, (x0, y0), 6, color,2)
    cv2.circle(img1_bgr, (x1, y1), 6, color,2)


img = concat_images(img0_bgr, img1_bgr)
plt.imshow(img)
plt.show()

img0_bgr = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
img3_bgr = cv2.cvtColor(img3, cv2.COLOR_GRAY2BGR)

for i, m in enumerate(matches_03):
    if matchesMask_03[i][0] == 0:
        continue
    pt0 = kp0[m[0].queryIdx]
    pt1 = kp3[m[0].trainIdx]
    
    x0, y0 = (int(pt0.pt[0]), int(pt0.pt[1]))
    x1, y1 = (int(pt1.pt[0]), int(pt1.pt[1]))   
    color = tuple(np.random.randint(0,255,3).tolist())
    cv2.circle(img0_bgr, (x0, y0), 6, color,2)
    cv2.circle(img3_bgr, (x1, y1), 6, color,2)

img = concat_images(img0_bgr, img3_bgr)
plt.imshow(img)
plt.show()


import pdb ; pdb.set_trace()