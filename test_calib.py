import matplotlib.pyplot as plt
import cv2
from utils import *
from cfg import *


def split_image_x4(img_file):
    return split_kite_vertical_images(img_file)



def rectifyFisheyeCameraPairs(cam0_img, cam1_img, K0, K1, D0, D1, R, T, img_size = (640, 480)):    
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(K0, D0, K1, D1, img_size, R, T, 0, img_size, 0.0, 1.2)
    left_maps = cv2.fisheye.initUndistortRectifyMap(K0, D0, R1, P1, img_size, cv2.CV_16SC2)
    right_maps = cv2.fisheye.initUndistortRectifyMap(K1, D1, R2, P2, img_size, cv2.CV_16SC2)
    left_img_remap = cv2.remap(cam0_img, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4)
    right_img_remap = cv2.remap(cam1_img, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4)
    return left_img_remap, right_img_remap, P1, P2



INPUT_IMAGE_PATH = os.environ["VO_IMG"]
INPUT_CALIB_PATH ='/home/jzhang/vo_data/SR80_901020874/nav_calib.cfg'
image_files = sorted(glob.glob(INPUT_IMAGE_PATH + '/*.jpg'))

image_files.sort(key=lambda f: int(filter(str.isdigit, f)))

mtx, dist, rot, trans, _, _ = load_kite_config(INPUT_CALIB_PATH)

rot_01, trans_01 = rot[1], trans[1]
rot_10, trans_10 = invert_RT(rot_01, trans_01)

rot_23, trans_23 = rot[3], trans[3]
rot_32, trans_32 = invert_RT(rot[3], trans[3])


for i in range(len(image_files)):
    img_x4 = split_image_x4(image_files[i])
    img0, img1, _, _ = rectifyFisheyeCameraPairs(
        img_x4[0], 
        img_x4[1], 
        mtx[0], 
        mtx[1], 
        dist[0][0:4], 
        dist[1][0:4], 
        rot_01,
        trans_01)
    
    roi_mask = region_of_interest_mask(img0.shape, KITE_MASK_VERTICES[0])
    flow0 = shi_tomasi_corner_detection(img0, roi_mask = roi_mask, kpts_num = 16)

    roi_mask = region_of_interest_mask(img0.shape, KITE_MASK_VERTICES[1])
    flow1 = shi_tomasi_corner_detection(img1, roi_mask = roi_mask, kpts_num = 16)

    img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

    import pdb; pdb.set_trace()
    img = concat_images(img0, img1)
    cv2.imwrite('/tmp/test/' + str(i) + '.jpg', img)

# plt.imshow(img)

# import pdb; pdb.set_trace()








