import cv2
import matplotlib.pyplot as plt
from utils import *
from cfg import *


IMG_PATH = '/home/jzhang/StereoCalib/undistroted/'
CALIB_PATH = '/home/jzhang/vo_data/SR80_201010761/nav_calib.cfg'



def load_image_points(board_shape, square_size, image_dir):
    img_files = glob.glob(image_dir + '/*.jpg')
    left_imgs = []
    right_imgs = []
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW

    for img_file in img_files:
        imgx4 = split_kite_vertical_images(img_file)
        left_imgs.append(imgx4[0])
        right_imgs.append(imgx4[1])
    
    leftImagePoints = []
    rightImagePoints = []
    objPoints = []
    j = 0
    for i in range(len(left_imgs)):
        left_found, corners_l = cv2.findChessboardCorners(left_imgs[i], board_shape, 
            cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

        right_found, corners_r = cv2.findChessboardCorners(right_imgs[i], board_shape, 
            cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)

        # import pdb; pdb.set_trace()
        if left_found and right_found:
            cv2.cornerSubPix(left_imgs[i], corners_l, (3,3), (-1,-1), subpix_criteria)
            cv2.cornerSubPix(right_imgs[i], corners_r, (3,3), (-1,-1), subpix_criteria)

            leftImagePoints.append(corners_l)
            rightImagePoints.append(corners_r)

            # cv2.drawChessboardCorners(left_imgs[i], board_shape, corners_l, left_found)
            # cv2.drawChessboardCorners(right_imgs[i], board_shape, corners_r, right_found)


            img = concat_images(cv2.cvtColor(left_imgs[i], cv2.COLOR_GRAY2BGR),  
                                cv2.cvtColor(right_imgs[i], cv2.COLOR_GRAY2BGR))

            # cv2.imwrite('/home/jzhang/StereoCalib/output/'+str(j)+'.jpg', img)
    
            objp = np.zeros((1, board_shape[0]*board_shape[1], 3), np.float32)
            objp[0,:,:2] = np.mgrid[0:board_shape[0], 0:board_shape[1]].T.reshape(-1, 2) * square_size
            objPoints.append(objp)
            j+= 1

    return leftImagePoints, rightImagePoints, objPoints


leftp, rightp, objp = load_image_points((9, 7), 0.02425, IMG_PATH)



mtx, dist, rot, trans, imu_rot, imu_trans = load_camera_calib('kite', CALIB_PATH, 4)

cameraMatrix = []

for cnt, m in enumerate(mtx):
    nMtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(m, dist[cnt][0:4], (640, 480), np.eye(3), balance=0)
    cameraMatrix.append(nMtx)

retval_l, cameraMatrix_l, distCoeffs_l, rvecs_l, tvecs_l = cv2.calibrateCamera(objp, leftp, (640, 480), None, None)
retval_r, cameraMatrix_r, distCoeffs_r, rvecs_r, tvecs_r = cv2.calibrateCamera(objp, rightp, (640, 480), None, None)

w = 640
h = 480
img_size = (640, 480)
newcameramtx_l, roi_l = cv2.getOptimalNewCameraMatrix(cameraMatrix_l, distCoeffs_l, img_size, 1, img_size)
newcameramtx_r, roi_r = cv2.getOptimalNewCameraMatrix(cameraMatrix_r, distCoeffs_r, img_size, 1, img_size)


img_files = glob.glob(IMG_PATH + '/*.jpg')
for cnt, img_file in enumerate(img_files):
    imgx4 = split_kite_vertical_images(img_file)
    left_img = cv2.undistort(imgx4[0], cameraMatrix_l, distCoeffs_l, None, newcameramtx_l)
    right_img = cv2.undistort(imgx4[1], cameraMatrix_r, distCoeffs_r, None, newcameramtx_r)

    
    left_img = concat_images(cv2.cvtColor(imgx4[0], cv2.COLOR_GRAY2BGR),
                             cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR))



    right_img = concat_images(cv2.cvtColor(imgx4[1], cv2.COLOR_GRAY2BGR),
                              cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR))

    img = stack_images(left_img, right_img)

    cv2.imwrite('/home/jzhang/StereoCalib/output/'+str(cnt)+'.jpg', img)

    import pdb; pdb.set_trace()




try:
    ret  = cv2.stereoCalibrate(objp, leftp, rightp, cameraMatrix[0], None, cameraMatrix[1], None, (640, 480))
    retval, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R1, T1, E, F = ret
except:
    import pdb; pdb.set_trace()

try:
    ret2  = cv2.stereoCalibrate(objp, leftp, rightp, cameraMatrix_l, None, cameraMatrix_r, None, (640, 480))
    retval, camMatrix1, dist1, camMatrix2, dist2, R2, T2, E, F = ret2
except:
    import pdb; pdb.set_trace()

# try:
#     retval, K1, D1, K2, D2, R, T = cv2.fisheye.stereoCalibrate(	objp, leftp, rightp, mtx[0], dist[0][0:4], mtx[1], dist[1][0:4], (640, 480))
# except:
#     import pdb; pdb.set_trace()

import pdb; pdb.set_trace()