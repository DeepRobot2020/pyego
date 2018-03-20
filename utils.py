
import numpy as np
import glob, pdb, math

import os, io, libconf, copy
import cv2, Image

from numpy.linalg import inv, pinv, norm

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

import time


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

''' Util functions '''
def gaussian_blur(img, kernel=(5,5)):
     return cv2.GaussianBlur(img, kernel, 0)


def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width, 3), dtype=np.uint8)
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img

def load_kv_nav_config(cfg_file='nav_calib.cfg', num_cams=4):
    if not os.path.exists(cfg_file):
        print('Error: ' + cfg_file + ' does not exit')
        return None, None, None, None 
    cam_matrix = [None] * num_cams
    dist = [None] * num_cams
    cam_rot = [None] * num_cams
    cam_trans = [None] * num_cams
    with io.open(cfg_file) as f:
        config = libconf.load(f)
        cam_config = config['calib']['cam']
        for i in range(num_cams):
            cam_calib = cam_config[i]
            cam_id = int(cam_calib['cam_id'])
    
            mtx = np.array(cam_calib['camera_matrix']).reshape(3,3)
            dist_coeff = np.array(cam_calib['dist_coeff'])
            rot = np.array(cam_calib['cam_rot']).reshape(3,3)
            trans = np.array(cam_calib['cam_trans']).reshape(3,1)
            # Store the results 
            cam_matrix[cam_id] = mtx
            dist[cam_id] = dist_coeff
            cam_rot[cam_id] = rot
            cam_trans[cam_id] = trans
    return cam_matrix, dist, cam_rot, cam_trans


def load_kitti_config(cfg_file='calib.txt',  num_cams=4):
    lines = [line.rstrip('\n') for line in open(cfg_file)]
    cam_matrix = [None] * num_cams
    dist = [None] * num_cams
    cam_rot = [None] * num_cams
    cam_trans = [None] * num_cams

    for cam_id, line in enumerate(lines):
        P0 = line.split(' ')[1:]
        P0 = [float(i) for i in P0]
        P0 = np.asarray(P0).reshape(3, -1)
        mtx = P0[0:3, 0:3]
        rot = np.eye(3)
        # import pdb; pdb.set_trace()
        trans = P0[:,3].reshape(3,1)
        trans = np.dot(inv(mtx), trans)
        cam_matrix[cam_id] = mtx
        dist[cam_id] = None
        cam_rot[cam_id] = rot
        cam_trans[cam_id] = trans
    return cam_matrix, dist, cam_rot, cam_trans

def load_kitti_poses(cfg_file=None):
    lines = [line.rstrip('\n') for line in open(cfg_file)]
    rotations = []
    translations = []
    for line in lines:
        pose = line.split(' ')
        pose = [float(i) for i in pose]
        pose = np.asarray(pose)
        rot = pose[0:9].reshape(3,3)
        tr = pose[9:].reshape(3,-1)
        rotations.append(rot)
        translations.append(tr)
    em_rot = []
    em_trans = []
    em_rot.append(np.eye(3))
    em_trans.append(np.zeros([3,1]))
    for i in range(1, len(rotations)):
        R0 = rotations[i-1]
        t0 = translations[i-1]
        R1 = rotations[i]
        t1 = translations[i]
        rot = np.dot(inv(R0), R1)
        trans = t1 - np.dot(rot, t0)
        em_rot.append(cv2.Rodrigues(rot)[0])
        em_trans.append(trans)
    return em_rot, em_trans


def undistort(img_path, K, D):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h,w = img.shape[:2]
    DIM = (w, h)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def fast_keypoints_detection(undistorted_img):
    fast = cv2.FastFeatureDetector_create(threshold=70, nonmaxSuppression=True)
    kpts = fast.detect(undistorted_img)
    out = []
    for kp in kpts:
        x = kp.pt[0] + 20
        y = kp.pt[1] + 21
        kp.pt = (x, y)
        kpt0 = np.array(kp.pt).reshape(1,2)
        out.append(kpt0.astype(np.float32))
    return np.array(out)

def translateImage3D(img, K_mtx, t):
    tx, ty, tz = t
    T = np.eye(4)
    T[0][3] = tx
    T[1][3] = ty
    T[2][3] = tz
    inv_K = inv(K_mtx)
    A1 = np.zeros([4, 3])
    A1[0:3, 0:3] = inv_K
    A1[3][2] = 1
    A2 = np.eye(4)[:3]
    A2[:,0:3] = K_mtx
    A2[:,3] = A2[:,2]
    A2[:,2] = np.zeros([3])
    M = np.dot(A2, np.dot(T, A1))
    warp = cv2.warpPerspective(img, M, (640, 481))
    return warp, M
    
def shi_tomasi_corner_detection(img, kpts_num=64):
    feature_params = dict( maxCorners = kpts_num,
                       qualityLevel = 0.05,
                       minDistance = 8,
                       blockSize = 7 )
    return cv2.goodFeaturesToTrack(img, mask = None, **feature_params)





def epi_constraint(pts1, pts2, F):
    pts1 = pts1.reshape(pts1.shape[0], -1)
    pts2 = pts2.reshape(pts2.shape[0], -1)
    d = []
    for i in range(len(pts1)):
        u0 = pts1[i][0]
        v0 = pts1[i][1]
        u1 = pts2[i][0]
        v1 = pts2[i][1]
        p0 = np.array([u0, v0, 1])
        p1 = np.array([u1, v1, 1])
        p1f = np.dot(p1, F)
        epi_error = abs(float(np.dot(p1f, p0)))
        d.append(epi_error)
    return d

def epiline(pt, F):
    f = F.flatten()
    x = float(pt[0])
    y = float(pt[1])
    ax = f[0]*x + f[1]*y + f[2]
    bx = f[3]*x + f[4]*y + f[5]
    cx = f[6]*x + f[7]*y + f[8]
    nu = ax*ax + bx*bx
    if nu > 0.00000001:
        nu = 1.0 / sqrt(nu)
    else:
        nu = 1.0
    return ax*nu, bx*nu, cx*nu


def rectify_camera_pairs(cam0_img, cam1_img, K0, K1, D0, D1, R, T, img_size = (640, 480), rectify_scale=1):    
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K0, D0, K1, D1, img_size, R, T)
    left_maps = cv2.initUndistortRectifyMap(K0, D0, R1, P1, img_size, cv2.CV_16SC2)
    right_maps = cv2.initUndistortRectifyMap(K1, D1, R2, P2, img_size, cv2.CV_16SC2)
    left_img_remap = cv2.remap(cam0_img, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4)
    right_img_remap = cv2.remap(cam1_img, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4)
    return left_img_remap, right_img_remap


def skew_symmetric(T):
    T = np.array(T)
    T = T.tolist()
    T = sum(T, [])
    r = [0, -T[2], T[1], T[2], 0, -T[0], -T[1], T[0], 0]
    r = np.array(r).reshape(3, 3)
    return r

def essential(R01, T01):
    T01_cross = skew_symmetric(T01)
    E = np.dot(T01_cross, R01)
    return E

def fundmental_matrix(R01, T01, K0, K1):
    E = essential(R01, T01)
    K1_inv_transpose = np.linalg.inv(K1).T
    K0_inv = np.linalg.inv(K0)
    F01 = np.dot(K1_inv_transpose, E)
    F = np.dot(F01, K0_inv)
    return F


def invert_RT(R, T):
    ''' Invert Rotation (3x3) and Translation (3x1)
    '''
    R2 = np.array(R).T
    T2 = -np.dot(R2, T)
    return R2, T2

def pil_split_rotate_navimage_4(img_file):
    """Split recorded nav images to 4
    # 0 | 3
    # 1 | 2
        """
    im = Image.open(img_file)
    width, height     = im.size
    splited_images    = 4 * [None]
    splited_images[0] = np.asarray(im.crop((0, 0, width//2, height//2)).rotate(-90, expand=1))
    splited_images[1] = np.asarray(im.crop((0, height//2, width//2, height)).rotate(-90, expand=1))
    splited_images[2] = np.asarray(im.crop((width//2, height//2, width, height)).rotate(90, expand=1))
    splited_images[3] = np.asarray(im.crop((width//2, 0, width, height//2)).rotate(90, expand=1))
    for i in range(4):
        splited_images[i] =cv2.cvtColor(splited_images[i], cv2.COLOR_RGB2GRAY)
    return splited_images

def cv_split_navimage_4(img_file):
    """Split recorded nav images to 4
    # 0 | 3
    # 1 | 2
    """
    im = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
    height, width = im.shape
    splited_images    = 4 * [None]
    splited_images[0] = im[0: height//2,      0:width//2]     # cam0
    splited_images[1] = im[height//2: height, 0:width//2]     # cam1
    splited_images[2] = im[height//2: height, width//2:width] # cam2
    splited_images[3] = im[0: height//2,      width//2:width] # cam3
    return splited_images

def load_recorded_images(record_path='./', max_imgs=10):
    img_files = glob.glob(record_path + '*.jpg')
    img_files.sort(key=lambda f: int(filter(str.isdigit, f))) 
    cam_imgs = [[] for i in range(4)]
    for file in img_files:
        imgs_x4 = pil_split_rotate_navimage_4(file)
        for c in range(4):
            cam_imgs[c].append(imgs_x4[c])
    return cam_imgs


def sparse_optflow(curr_im, target_im, flow_kpt0, win_size  = (8, 8)):
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = win_size,
                    maxLevel = 4,
                    minEigThreshold=1e-4,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.01))
    # perform a forward match
    flow_kpt1, st, err = cv2.calcOpticalFlowPyrLK(curr_im, target_im, flow_kpt0, None, **lk_params)
    # perform a reverse match
    flow_kpt2, st, err = cv2.calcOpticalFlowPyrLK(target_im, curr_im, flow_kpt1, None, **lk_params)
    return flow_kpt0, flow_kpt1, flow_kpt2 

def rotate(points, rot_vecs):
    """Rotate points by given rotation vectors.
    Rodrigues' rotation formula is used.
    """
    rot_vecs = rot_vecs.reshape(1, -1)
    theta = np.linalg.norm(rot_vecs, axis=1)[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        v = rot_vecs / theta
        v = np.nan_to_num(v)
    dot = np.sum(points * v, axis=1)[:, np.newaxis]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    return cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v


# Note: this sparse matrix only care bout keypoints which has both intra and inter matching
# Each motion has 6 unknows: 
# R = [wx, wy, wz]^T
# t = [tx, ty, tz]^T
def local_bundle_adjustment_sparsity(cam_obs, n_poses=1):
    n_obs_013 = cam_obs[0]
    n_obs_01  = cam_obs[1]
    n_obs = n_obs_013 + n_obs_01

    m = n_obs_013 * 6 + n_obs_01 * 4
    n = n_poses * 6 + n_obs * 3
    A = lil_matrix((m, n), dtype=int)
    # fill in the sparse struct of A
    i = np.arange(n_obs_013)
    j = np.arange(n_obs_01)

    n_offset_i = 0
    n_offset_j = 0
    m_offset = 0
    # fill the flow1 entries
    # fill the entries for egomotion
    for k in range(6):
        A[2 * i, k] = 1
        A[2 * i + 1, k] = 1

    for k in range(3):
        A[2 * i,     n_poses * 6 + i * 3 + k] = 1
        A[2 * i + 1, n_poses * 6 + i * 3 + k] = 1

    m_offset += n_obs_013 * 2
    n_offset_j += n_obs_013 * 3

    if n_obs_01 > 0:
        for k in range(6):
            A[m_offset + 2 * j, k] = 1
            A[m_offset + 2 * j + 1, k] = 1

        for k in range(3):
            A[m_offset + 2 * j,     n_offset_j + n_poses * 6 + j * 3 + k] = 1
            A[m_offset + 2 * j + 1, n_offset_j + n_poses * 6 + j * 3 + k] = 1

    m_offset += n_obs_01 * 2
    # fill the flow013_flow0 entries
    for k in range(3):
        A[m_offset + 2 * i,     n_offset_i + n_poses * 6 + i * 3 + k] = 1
        A[m_offset + 2 * i + 1, n_offset_i + n_poses * 6 + i * 3 + k] = 1

    m_offset += n_obs_013 * 2
    # fill the flow013_flow3 entries
    for k in range(3):
        A[m_offset + 2 * i,     n_offset_i + n_poses * 6 + i * 3 + k] = 1
        A[m_offset + 2 * i + 1, n_offset_i + n_poses * 6 + i * 3 + k] = 1


    m_offset += n_obs_013 * 2
    # fill the flow01_flow0 entries
    if n_obs_01 > 0:
        for k in range(3):
            A[m_offset + 2 * j,     n_offset_j + n_poses * 6 + j * 3 + k] = 1
            A[m_offset + 2 * j + 1, n_offset_j + n_poses * 6 + j * 3 + k] = 1

    return A

def global_bundle_adjustment_sparsity(cam_obs, n_cams=4, n_poses=1):
    n_obs_013 = cam_obs[:,0]
    n_obs_01  = cam_obs[:,1]

    n_obs_013_sum = np.sum(n_obs_013)
    n_obs_01_sum = np.sum(n_obs_01)

    n_obs = n_obs_013_sum + n_obs_01_sum

    m = (n_obs_013_sum * 3 + n_obs_01_sum * 2) * 2
    n = n_poses * 6 + n_obs * 3
    A = lil_matrix((m, n), dtype=int)
    # fill in the sparse struct of A
    m_offset_i = 0
    n_offset_i = 0
    m_offset_j = 0
    n_offset_j = 0

    for c in range(n_cams):
        n_obs_i = n_obs_013[c]
        n_obs_j = n_obs_01[c]

        i = np.arange(n_obs_i)
        j = np.arange(n_obs_j)
        
        m_offset_j = m_offset_i + 6*n_obs_i 
        n_offset_j = n_offset_i + 3*n_obs_i

        # fill flow0 for cam c
        for k in range(3):
            A[m_offset_i + 2 * i,     n_offset_i + n_poses * 6 + i * 3 + k] = 1
            A[m_offset_i + 2 * i + 1, n_offset_i + n_poses * 6 + i * 3 + k] = 1  
            A[m_offset_j + 2 * j,     n_offset_j + n_poses * 6 + j * 3 + k] = 1
            A[m_offset_j + 2 * j + 1, n_offset_j + n_poses * 6 + j * 3 + k] = 1  

        # fill the flow1 entries
        # fill the entries for egomotion
        for k in range(6):
            A[m_offset_i + n_obs_i * 2 + 2 * i, k] = 1
            A[m_offset_i + n_obs_i * 2 + 2 * i + 1, k] = 1
            A[m_offset_j + n_obs_j * 2 + 2 * j, k] = 1
            A[m_offset_j + n_obs_j * 2 + 2 * j + 1, k] = 1

        for k in range(3):
            A[m_offset_i + n_obs_i * 2 + 2 * i,     n_offset_i + n_poses * 6 + i * 3 + k] = 1
            A[m_offset_i + n_obs_i * 2 + 2 * i + 1, n_offset_i + n_poses * 6 + i * 3 + k] = 1

            A[m_offset_j + n_obs_j * 2 + 2 * j,     n_offset_j + n_poses * 6 + j * 3 + k] = 1
            A[m_offset_j + n_obs_j * 2 + 2 * j + 1, n_offset_j + n_poses * 6 + j * 3 + k] = 1

        # fill the flow3 entries 
        for k in range(3):
            A[m_offset_i + n_obs_i * 4 + 2 * i,     n_offset_i + n_poses * 6 + i * 3 + k] = 1
            A[m_offset_i + n_obs_i * 4 + 2 * i + 1, n_offset_i + n_poses * 6 + i * 3 + k] = 1

        m_offset_i += (3*n_obs_i + 2 * n_obs_j) * 2
        n_offset_i += (n_obs_i + n_obs_j) * 3

    return A


def global_bundle_adjustment_sparsity_opt(cam_obs, n_cams=4, n_poses=1):
    n_obs_013 = cam_obs[:,0]
    n_obs_01  = cam_obs[:,1]

    n_obs_013_sum = np.sum(n_obs_013)
    n_obs_01_sum = np.sum(n_obs_01)

    n_obs = n_obs_013_sum + n_obs_01_sum

    m = (n_obs_013_sum * 3 + n_obs_01_sum * 2) * 2
    n = n_poses * 6 + n_obs * 3
    A = lil_matrix((m, n), dtype=int)
    # fill in the sparse struct of A
    m_offset_i = 0
    n_offset_i = 0

    m_offset = 0
    n_offset_j = 0
    for c in range(n_cams):
        n_obs_i = n_obs_013[c]
        n_obs_j = n_obs_01[c]

        i = np.arange(n_obs_i)
        j = np.arange(n_obs_j)
        
        # m_offset = m_offset + 6*n_obs_i 
        # n_offset_j = n_offset_i + 3*n_obs_i

        # fill the flow013_1 entries
        # fill the entries for egomotion
        for k in range(6):
            A[m_offset + 2 * i,     k] = 1
            A[m_offset + 2 * i + 1, k] = 1
        # fill the entries for flow013_1
        for k in range(3):
            A[m_offset + 2 * i,     n_offset_i + n_poses * 6 + i * 3 + k] = 1
            A[m_offset + 2 * i + 1, n_offset_i + n_poses * 6 + i * 3 + k] = 1
            
        m_offset += n_obs_i * 2
        n_offset_j += n_obs_i * 3
        
        # fill the flow01_1 entries
        if n_obs_j > 0:
            # fill the entries for egomotion
            for k in range(6):
                A[m_offset + 2 * i,     k] = 1
                A[m_offset + 2 * i + 1, k] = 1

            for k in range(3):
                A[m_offset + 2 * j,     n_offset_j + n_poses * 6 + j * 3 + k] = 1
                A[m_offset + 2 * j + 1, n_offset_j + n_poses * 6 + j * 3 + k] = 1

        m_offset += n_obs_j * 2

        # fill the flow013_flow0 entries
        for k in range(3):
            A[m_offset + 2 * i,     n_offset_i + n_poses * 6 + i * 3 + k] = 1
            A[m_offset + 2 * i + 1, n_offset_i + n_poses * 6 + i * 3 + k] = 1

        m_offset += n_obs_i * 2
        # fill the flow013_flow3 entries
        for k in range(3):
            A[m_offset + 2 * i,     n_offset_i + n_poses * 6 + i * 3 + k] = 1
            A[m_offset + 2 * i + 1, n_offset_i + n_poses * 6 + i * 3 + k] = 1
        
        m_offset += n_obs_i * 2
        # fill the flow01_flow0 entries
        if n_obs_j > 0:
            for k in range(3):
                A[m_offset + 2 * j,     n_offset_j + n_poses * 6 + j * 3 + k] = 1
                A[m_offset + 2 * j + 1, n_offset_j + n_poses * 6 + j * 3 + k] = 1

        m_offset += n_obs_j * 2
        n_offset_i = (n_obs_i + n_obs_j) * 3
    return A
    