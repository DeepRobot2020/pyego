import numpy as np
import cv2

import glob
from utils import *
from cfg import *




def undistort(img_path, K, D, balance=0.0):
    img = cv2.imread(img_path)
    dim = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = correct_kite_camera_matrix(K, D, dim, balance=balance)
    undistorted_img = undistort_kite_image(img, K, new_K, D)
    return undistorted_img



def fisheye_calib(image_path, CHECKERBOARD = (9,7)):
    images = glob.glob(image_path  + '/*.jpg')

    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

    objp = np.zeros((1, CHECKERBOARD[0]* CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

    _img_shape = None
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.


    for fname in images:
        img = cv2.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)

            cv2.drawChessboardCorners(img, (CHECKERBOARD[0], CHECKERBOARD[1]), corners, ret)
            fname2 = fname.split('/')[-1].split('.')[0]
            cv2.imwrite('/home/jzhang/pyego/data/cam0/debug/' + fname2 + '.jpg', img)
            
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    rms, _, _, _, _ = \
        cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        )
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
    return K, D

IMG_PATH = '/home/jzhang/pyego/data'
IMG_PATH2 = '/home/jzhang/pyego/data/cam0'
CALIB_PATH = '/home/jzhang/vo_data/jzhang_R80/nav_calib.cfg'

mtx, dist, rot, trans = load_kite_config(CALIB_PATH)

K = mtx[0]
D = dist[0][:4]

# split_image_x4(IMG_PATH)

CHESS_W = 9
CHESS_H = 7
CHECKERBOARD = (CHESS_W, CHESS_H)
# K, D = fisheye_calib(IMG_PATH2, CHECKERBOARD)

images = glob.glob(IMG_PATH2  + '/*.jpg')

for img in images:
    undistorted_img = undistort(img, K, D)
    cv2.imshow("undistorted", undistorted_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
