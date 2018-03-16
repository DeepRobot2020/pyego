import cv2
import numpy as np
import glob
import pdb
import math
import os
import io, libconf
import numpy 
import Image
import copy
from numpy.linalg import inv
from numpy.linalg import pinv
from numpy.linalg import multi_dot

from numpy.linalg import norm
from scipy.sparse import lil_matrix

import time
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
 
# from matplotlib import pyplot as plt
from math import hypot
from math import sqrt

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

class PinholeCamera:
    	def __init__(self, width, height, fx, fy, cx, cy, 
				k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
		self.width = width
		self.height = height
		self.fx = fx
		self.fy = fy
		self.cx = cx
		self.cy = cy
		self.distortion = (abs(k1) > 0.0000001)
		self.d = [k1, k2, p1, p2, k3]

kt_cam = PinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)

np.set_printoptions(suppress=True)

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
        nu = 1.0 / math.sqrt(nu)
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
    ''' Invert R and T
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

class navcam:
    def __init__(self,  index, stereo_pair_idx, F, intrinsic_mtx,  intrinsic_dist,  extrinsic_rot, extrinsic_trans, max_flow_kpts=64):
        self.calib_K    = intrinsic_mtx
        self.calib_d    = intrinsic_dist
        self.calib_R    = extrinsic_rot
        self.calib_R2   = cv2.Rodrigues(extrinsic_rot)[0]
        self.calib_t    = extrinsic_trans
        self.max_flow_kpts = max_flow_kpts
        self.flow_kpt0 = None
        self.flow_kpt1 = None
        self.flow_kpt2 = None
        self.flow_kpt3 = None
        self.flow_kpt4 = None
        self.stereo_pair_cam = None
        self.stereo_pair_idx = stereo_pair_idx 
        self.cam_obs = np.zeros([4,2], dtype=np.int)

        self.stereo_R = None # camera pose rotation in 
        self.stereo_t = None 
        self.mono_cur_R = None
        self.mono_cur_t = None
        self.ego_R = None
        self.ego_t = None
        self.prev_scale = None
        # keypoints which have both intra and inter matching
        self.flow_intra_inter0 = None  # original keypoints
        self.flow_intra_inter1 = None  # intra match
        self.flow_intra_inter3 = None  # inter match
        self.flow_intra_inter_P = None # world point in camera self.index's coordinate system

        self.flow_intra0  = None # original keypoints
        self.flow_intra1  = None # intra match
        self.flow_intra_P = None # world point in camera self.index's coordinate system

        self.flow_inter0 = None # original keypoints
        self.flow_inter3 = None # inter match

        self.intra0 = None # original keypoints
        self.intra1 = None # inter match

        self.curr_img  = None
        self.curr_stereo_img  = None
        self.prev_img  = None
        self.index   = index

        self.img_idx = None
        self.F = F
        self.proj_mtx = None
        self.focal = kt_cam.fx
        self.pp = (kt_cam.cx, kt_cam.cy)

    def mono_vo(self, abs_scale):
        if self.img_idx is None or self.img_idx == 0:
            return None, None
        if self.intra0 is None:
            return self.mono_cur_R, self.mono_cur_t
        if self.intra0.shape[0] < 5:
            return self.mono_cur_R, self.mono_cur_t
        try:
            E, mask = cv2.findEssentialMat(self.intra0, self.intra1, focal=self.focal, pp=self.pp, method=cv2.RANSAC, prob=0.999, threshold=1.0)
        except:
            import pdb; pdb.set_trace()
        try:
            _, R, t, mask = cv2.recoverPose(E, self.intra0, self.intra1, focal=self.focal, pp = self.pp)
        except:
            import pdb; pdb.set_trace()

        rot = cv2.Rodrigues(R)[0]
        tr = t

        # print('mon:', self.img_idx, 'est_rot', rot.ravel(), 'est_tras', abs_scale * tr.ravel())

        if self.img_idx == 1:
            self.mono_cur_t = t
            self.mono_cur_R = R
        else:
            if(abs_scale > 0.1):
                # import pdb; pdb.set_trace()
                self.mono_cur_t = self.mono_cur_t + abs_scale*self.mono_cur_R.dot(t) 
                self.mono_cur_R = R.dot(self.mono_cur_R)
        return rot, tr
    def set_its_pair(self, right_cam):
        self.stereo_pair_cam = right_cam

    def project_to_flow0(self, cam_points):
        if cam_points is None:
                return None
        image_points_flow0 = np.dot(self.calib_K, cam_points.T).T
        # convert the projected points to the P2 homogenous coordinates
        image_points_flow0 = image_points_flow0[:, :2] / image_points_flow0[:, 2, np.newaxis]
        return image_points_flow0

    def project_to_flow1(self, cam_points, ego_rot_vecs, ego_trans):
        if cam_points is None:
                return None
        ego_trans = ego_trans.reshape(1, -1)
        ego_rot_vecs = ego_rot_vecs.reshape(1, -1)
        points_proj_flow1 = rotate(cam_points, ego_rot_vecs) + ego_trans
        image_points_flow1 = np.dot(self.calib_K, points_proj_flow1.T).T
        # convert the projected points to the P2 homogenous coordinates
        image_points_flow1 = image_points_flow1[:, :2] / image_points_flow1[:, 2, np.newaxis]
        return image_points_flow1

    def project_to_flow3(self, cam_points, cam_rot_vecs, cam_trans):
        cam_rot_vecs = cam_rot_vecs.reshape(1,-1)
        cam_trans = cam_trans.reshape(1, -1)

        points_proj_flow3 = rotate(cam_points, cam_rot_vecs) + cam_trans
        image_points_flow3 = np.dot(self.stereo_pair_cam.calib_K, points_proj_flow3.T).T
        # convert the projected points to the P2 homogenous coordinates
        image_points_flow3 = image_points_flow3[:, :2] / image_points_flow3[:, 2, np.newaxis]
        return image_points_flow3

    def init_flow013_camera_points(self, est_depth=1.0):
        if len(self.flow_intra_inter0) == 0:
            return None, 1.0
        # est_depth = 
        flow0 = cv2.convertPointsToHomogeneous(self.flow_intra_inter0.astype(np.float32))
        flow0 = flow0.reshape(flow0.shape[0], flow0.shape[2]).T
        inv_K = inv(self.calib_K)
        camera_points = np.dot(inv_K, flow0).T * est_depth
        return camera_points

    def construct_projection_mtx(self, K1, K2, R, t):
        left_T = np.eye(4)[:3]
        left_mtx = np.dot(K1, left_T)
        right_T = np.zeros([3,4])
        right_T[0:3,0:3] = R
        right_T[:,3][:3] = t.ravel()
        right_mtx = np.dot(K2, right_T)
        return left_mtx, right_mtx

    def test_fake_points(self, gt_rot, gt_trans, npoints_013=10, npoints_01=30):
        
        true_points_013 = np.hstack([np.random.random(
            (npoints_013, 1)) * 3 - 1.5, np.random.random((npoints_013, 1)) - 0.5, np.random.random((npoints_013, 1)) + np.arange(npoints_013).reshape(-1, 1)])
        true_points_01 = np.hstack([np.random.random(
            (npoints_01, 1)) * 3 - 1.5, np.random.random((npoints_01, 1)) - 0.5, np.random.random((npoints_01, 1)) + 3])

        # project points to current camera
        points_013_0 = self.project_to_flow0(true_points_013)
        # project points to t-1
        points_013_1 = self.project_to_flow1(true_points_013, gt_rot, gt_trans) 
        # project points to stereo pair
        points_013_3 = self.project_to_flow3(true_points_013, self.calib_R2, self.calib_t) 

        # project points to current camera
        points_01_0 = self.project_to_flow0(true_points_01)
        # project points to t-1
        points_01_1 = self.project_to_flow1(true_points_01, gt_rot, gt_trans)

        self.flow_intra_inter0 = points_013_0 + np.random.randn(npoints_013, 2)
        self.flow_intra_inter1 = points_013_1 + np.random.randn(npoints_013, 2)
        self.flow_intra_inter3 = points_013_3 + np.random.randn(npoints_013, 2)

        self.flow_intra0 = points_01_0 + np.random.randn(npoints_01, 2)
        self.flow_intra1 = points_01_1 + np.random.randn(npoints_01, 2)
        
        return true_points_013, true_points_01

    def init_flow01_camera_points(self, est_depth=1.0):
        if self.flow_intra0 is None:
            return None
        if len(self.flow_intra0) == 0:
            return None
        flow0 = cv2.convertPointsToHomogeneous(self.flow_intra0.astype(np.float32))
        flow0 = flow0.reshape(flow0.shape[0], flow0.shape[2]).T
        inv_K = inv(self.calib_K)
        camera_points = est_depth * np.dot(inv_K, flow0).T
        return camera_points

    def triangulate_3d_points_intra(self, kpts1, kpts2, est_R, est_t):
        left_p, right_p = self.construct_projection_mtx(self.calib_K, self.calib_K, est_R, est_t)
        if kpts1 is None:
            return None
        if len(kpts1) < 1 or len(kpts2) < 1:
            return None
        try:
            scene_pts = cv2.triangulatePoints(left_p, right_p, kpts1.T, kpts2.T).T
        except:
            print('cv2.triangulatePoints() failed')
            return None
        # import pdb; pdb.set_trace()
        left_proj = np.dot(left_p, scene_pts.T).T
        left_proj = left_proj[:,0:2] / left_proj[:,2][:,np.newaxis]
        err_left = left_proj - kpts1

        right_proj = np.dot(right_p, scene_pts.T).T
        right_proj = right_proj[:,0:2] / right_proj[:,2][:,np.newaxis]
        err_right  = right_proj - kpts2
        points_cam_cur = scene_pts[:,0:3] / scene_pts[:,3][:,np.newaxis]
        return points_cam_cur, err_left, err_right


    def triangulate_3d_points(self, left_kpts, right_kpts):
        left_p, right_p = self.construct_projection_mtx(self.calib_K, self.stereo_pair_cam.calib_K, self.calib_R, self.calib_t)
        # left_kpts = self.flow_intra_inter0
        # right_kpts = self.flow_intra_inter3
        if right_kpts is None:
            return None
        if len(left_kpts) < 1 or len(right_kpts) < 1:
            return None
        try:
            scene_pts = cv2.triangulatePoints(left_p, right_p, left_kpts.T, right_kpts.T).T
        except:
            print('cv2.triangulatePoints() failed')
            return None
        # import pdb; pdb.set_trace()
        left_proj = np.dot(left_p, scene_pts.T).T
        left_proj = left_proj[:,0:2] / left_proj[:,2][:,np.newaxis]
        err_left = left_proj - left_kpts

        right_proj = np.dot(right_p, scene_pts.T).T
        right_proj = right_proj[:,0:2] / right_proj[:,2][:,np.newaxis]
        err_right  = right_proj - right_kpts
        points_cam_cur = scene_pts[:,0:3] / scene_pts[:,3][:,np.newaxis]
        return points_cam_cur, err_left, err_right

    def fun2(self, x0, cam_obs, y_meas, n_pose=0):
        n_obj_013, n_obj_01 = cam_obs
        points_013 = x0[6*n_pose: 6*n_pose + 3 * n_obj_013].reshape(-1, 3)

        flow013_0  = y_meas[0             : 1 * n_obj_013]
        flow013_3  = y_meas[2 * n_obj_013 : 3 * n_obj_013]

        flow0_err = flow013_0 - self.project_to_flow0(points_013)
        flow3_err = flow013_3 - self.project_to_flow3(points_013, self.calib_R2, self.calib_t)
        
        errs = np.vstack((flow0_err, flow3_err))
        return errs.ravel()

    def fun(self, x0, cam_obs, y_meas):
        n_obj_013, n_obj_01 = cam_obs
                
        rot_vecs   = x0[0:3]
        trans_vecs = x0[3:6]
        points_013 = x0[6: 6 + 3 * n_obj_013].reshape(-1, 3)
        flow013_0  = y_meas[0             : 1 * n_obj_013]
        flow013_1  = y_meas[1 * n_obj_013 : 2 * n_obj_013]
        flow013_3  = y_meas[2 * n_obj_013 : 3 * n_obj_013]
        flow0_err = self.project_to_flow0(points_013) - flow013_0
        flow1_err = self.project_to_flow1(points_013, rot_vecs, trans_vecs) - flow013_1
        flow3_err = self.project_to_flow3(points_013, self.calib_R2, self.calib_t) - flow013_3

        errs = flow1_err
        errs013_03 = np.vstack((flow0_err, flow3_err))
        flow01_err0 = None
        flow01_err1 = None

        if n_obj_01 > 0:
            points_01  = x0[6 + 3 * n_obj_013 : 6 + 3 * n_obj_013 + 3 * n_obj_01].reshape(-1, 3)
            flow01_0  = y_meas[3 * n_obj_013  : 3 * n_obj_013 + n_obj_01]
            flow01_1  = y_meas[3 * n_obj_013 + n_obj_01: 3 * n_obj_013 + 2*n_obj_01]

            flow01_err0 = self.project_to_flow0(points_01) - flow01_0
            flow01_err1 = self.project_to_flow1(points_01, rot_vecs, trans_vecs) - flow01_1
            errs = np.vstack((errs, flow01_err1))   
            errs013_03 = np.vstack((errs013_03, flow01_err0))

        errs = np.vstack((errs, errs013_03))
        return errs.ravel()

    def mono_ego_motion_estimation(self, abs_scale=None):
        mono_rot, mono_tr = self.mono_vo(1.0)
        if mono_rot is None:
            return None, None
        if abs_scale:
            mono_tr = abs_scale * mono_tr
        return mono_rot, mono_tr

    def update_camera_pose_egomotion(self, R, t):
        self.ego_R = R
        self.ego_t = t
        self.prev_scale = norm(t) if norm(t) > 0.01 else self.prev_scale
        if self.img_idx == 1:
            self.stereo_R = R
            self.stereo_t = t
        else:
            self.stereo_t = self.stereo_t + self.stereo_R.dot(t) 
            try:
                self.stereo_R = R.dot(self.stereo_R)
            except:
                import pdb; pdb.set_trace()
        return self.stereo_R, self.stereo_t

    def get_egomotion(self):
        return self.ego_R, self.ego_t

    def get_stereo_camera_pose(self):
        return self.stereo_R, self.stereo_t
    
    def local_ego_motion_solver(self, debug=False, fake_gt_rot=[0, 0., 0.], fake_gt_trans=[-0.2, 0.0, 0.0], n_points013=15, n_points01=30):
        if self.img_idx is None or self.img_idx == 0:
            return None, None

        # initial guess
        est_R = np.random.normal(0, 0.01, [3,1])
        est_t = np.random.normal(0, 0.01, [3,1])

        mono_R, mono_t = self.mono_ego_motion_estimation()
        if mono_R is not None:
            est_R = mono_R
            est_t = self.prev_scale * mono_t if self.prev_scale is not None else mono_t

        n_points013 = self.flow_intra_inter0.shape[0]
        n_points01 = self.flow_intra0.shape[0]
   
        n_obs_i = self.flow_intra_inter0.shape[0]
        n_obs_j = self.flow_intra0.shape[0]
   
        self.cam_obs = np.zeros([4,2], dtype=np.int)
        self.cam_obs[self.index][0] = n_obs_i
        self.cam_obs[self.index][1] = n_obs_j

        # sparse_A = local_bundle_adjustment_sparsity(self.cam_obs[self.index], 1)
        sparse_A = global_bundle_adjustment_sparsity_opt(self.cam_obs, n_cams=4)        

        y_meas = None
        x0 = np.vstack([est_R, est_t])

        if n_obs_i > 0:
            flow0 = self.flow_intra_inter0
            flow1 = self.flow_intra_inter1
            flow3 = self.flow_intra_inter3

            points013, terr013_0, terr013_1 = self.triangulate_3d_points(flow0, flow3)
            points013_flatten = points013.ravel().reshape(-1, 1)
            y_meas = np.vstack([flow0, flow1, flow3])
            x0 = np.vstack([x0, points013_flatten])
            if n_obs_j > 0:
                flow0 = self.flow_intra0
                flow1 = self.flow_intra1
                points01, terr01_0, terr01_1 = self.triangulate_3d_points_intra(flow0, flow1, cv2.Rodrigues(est_R)[0], est_t)
                points01_flatten = points01.ravel().reshape(-1, 1)
                x0 = np.vstack([x0, points01_flatten])
                y_meas = np.vstack([y_meas, flow0, flow1])
        else:
            print('cam_'+ str(self.index) + ' dont have inter match. Using mono results')
            # import pdb; pdb.set_trace()
            stereo_R, stereo_t = self.update_camera_pose_egomotion(cv2.Rodrigues(est_R)[0], est_t.reshape(3,))
            # import pdb; pdb.set_trace()
            return stereo_R, stereo_t

        x0 = x0.flatten()

        t0 = time.time()

        ls_pars = dict(jac_sparsity=sparse_A,
                    max_nfev=5, 
                    verbose=0,
                    x_scale='jac',
                    jac='2-point',
                    ftol=1e-3, 
                    xtol=1e-3,
                    gtol=1e-3,
                    method='trf')

        # err0 = self.fun(x0, self.cam_obs[self.index], y_meas)
        res = least_squares(self.fun, x0, args=(self.cam_obs[self.index], y_meas), **ls_pars)
        # err1 = self.fun(res.x, self.cam_obs[self.index], y_meas)
        n_obj_013, n_obj_01 = self.cam_obs[self.index] 
        t1 = time.time()

        print('cam_'+str(self.index), self.img_idx, n_obs_i, n_obs_j, 'est_rot', res.x[0:3], 'est_tras', res.x[3:6])
        # import pdb; pdb.set_trace()
        # plt.plot(err0); plt.plot(err1); plt.show();
        R = cv2.Rodrigues(res.x[0:3])[0]
        t = res.x[3:6]
        stereo_R, stereo_t = self.update_camera_pose_egomotion(R, t)
        return stereo_R, stereo_t


    def projection_mtx(self):
        if self.proj_mtx is not None:
            return self.proj_mtx
        T = np.zeros([3,4])
        if self.index == 0:
            T[0:3,0:3] = np.identity(3)
        else:
            T[0:3,0:3] = self.calib_R
            T[:,3][:3] = self.calib_t.ravel()
        self.proj_mtx = np.dot(self.calib_K, T)
        return self.proj_mtx
        
    def update_image(self, imgs_x4):
        self.prev_img = copy.deepcopy(self.curr_img)        
        self.curr_img = copy.deepcopy(imgs_x4[self.index])
        self.curr_stereo_img = copy.deepcopy(imgs_x4[self.stereo_pair_idx])
        if self.img_idx is None:
            self.img_idx = 0
        else:
            self.img_idx += 1


    def keypoint_detection(self):
        if self.curr_img is None:
            print('Warning: curr_img is None')
            return
        self.flow_kpt0 = shi_tomasi_corner_detection(self.curr_img, self.max_flow_kpts)
    
    def intra_sparse_optflow(self):
        if self.prev_img is not None:
            k0, k1, k2 = sparse_optflow(self.curr_img, self.prev_img, self.flow_kpt0)
            self.flow_kpt1 = k1
            self.flow_kpt2 = k2

    def inter_sparse_optflow(self):
        if self.prev_img is not None:
            k0, k3, k4 = sparse_optflow(self.curr_img, self.curr_stereo_img, self.flow_kpt0, win_size=(16, 16))
            self.flow_kpt3 = k3
            self.flow_kpt4 = k4

    def filter_intra_keypoints(self, debug=True, out_dir='/tmp'):
        img = None
        if self.prev_img is not None:
            for ct, (pt1, pt2, pt3) in enumerate(zip(self.flow_kpt0, self.flow_kpt1, self.flow_kpt2)):
                x1, y1 = (pt1[0][0], pt1[0][1])
                x2, y2 = (pt2[0][0], pt2[0][1])
                x3, y3 = (pt3[0][0], pt3[0][1])
                xe = abs(x3 - x1)
                ye = abs(y3 - y1)
                if x2 < 0.0 or y2 < 0.0:
                    self.flow_kpt1[ct][0][0] = -10.0
                    self.flow_kpt1[ct][0][1] = -10.0
                    continue
                if xe > 0.5 or ye > 0.5:
                    self.flow_kpt1[ct][0][0] = -20.0
                    self.flow_kpt1[ct][0][1] = -20.0
                    continue  


    def filter_inter_keypoints(self, debug=True, out_dir='/tmp'):
        img = None
        if self.prev_img is not None:
            ep_err = epi_constraint(self.flow_kpt0, self.flow_kpt3, self.F)
            for ct, (err, pt1, pt3, pt4) in enumerate(zip(ep_err, self.flow_kpt0, self.flow_kpt3, self.flow_kpt4)):
                x1, y1 = (pt1[0][0], pt1[0][1])
                x3, y3 = (pt3[0][0], pt3[0][1])
                x4, y4 = (pt4[0][0], pt4[0][1])
                xe = abs(x4 - x1)
                ye = abs(y4 - y1)
                if x3 < 0.0 or y3 < 0.0:
                    self.flow_kpt3[ct][0][0] = -10.0
                    self.flow_kpt3[ct][0][1] = -10.0
                    continue
                if xe > 0.5 or ye > 0.5:
                    self.flow_kpt3[ct][0][0] = -20.0
                    self.flow_kpt3[ct][0][1] = -20.0
                    continue  
                if err > 0.01:
                    self.flow_kpt3[ct][0][0] = -30.0
                    self.flow_kpt3[ct][0][1] = -30.0
                    continue


    def debug_inter_keypoints(self, out_dir='/tmp'):
        img = None
        if self.prev_img is not None:
            img1 = cv2.cvtColor(self.curr_img, cv2.COLOR_GRAY2BGR)
            img2 = cv2.cvtColor(self.curr_stereo_img, cv2.COLOR_GRAY2BGR)

            for pt1, pt3 in zip(self.flow_intra_inter0, self.flow_intra_inter3):
                x1, y1 = (pt1[0], pt1[1])
                x3, y3 = (pt3[0], pt3[1])
                color = tuple(np.random.randint(0,255,3).tolist())
                cv2.circle(img1,(x1, y1), 6, color,2)
                cv2.circle(img2,(x3, y3), 6, color,2)
            img = concat_images(img1, img2)
            # img_warp, M0 = translateImage3D(self.curr_img, self.calib_K, [0.2, 0.0, 0.0])
            if img is not None:
                out_img_name = os.path.join(out_dir, 'cam_' + str(self.index) + '_inter_' + str(self.img_idx)+'.jpg')
                cv2.imwrite(out_img_name, img)

    def debug_intra_keypoints(self, out_dir='/tmp'):
        img = None
        if self.prev_img is not None:
            img1 = cv2.cvtColor(self.curr_img, cv2.COLOR_GRAY2BGR)
            img2 = cv2.cvtColor(self.prev_img, cv2.COLOR_GRAY2BGR)
            for pt1, pt2 in zip(self.flow_intra_inter0, self.flow_intra_inter1):
                x1, y1 = (pt1[0], pt1[1])
                x3, y3 = (pt2[0], pt2[1])
                color = tuple(np.random.randint(0,255,3).tolist())
                cv2.circle(img1,(x1, y1), 6, color,2)
                cv2.circle(img2,(x3, y3), 6, color,2)

            for pt1, pt2 in zip(self.flow_intra0, self.flow_intra1):
                x1, y1 = (pt1[0], pt1[1])
                x3, y3 = (pt2[0], pt2[1])
                color = tuple(np.random.randint(0,255,3).tolist())
                cv2.circle(img1,(x1, y1), 6, color,2)
                cv2.circle(img2,(x3, y3), 6, color,2)

            img = concat_images(img1, img2)
            if img is not None:
                out_img_name = os.path.join(out_dir, 'cam_' + str(self.index) + '_intra_' + str(self.img_idx)+'.jpg')
                cv2.imwrite(out_img_name, img)


    def filter_keypoints(self, debug=False, out_dir='/home/jzhang/Pictures/tmp/', max_inter_pts = 100):
        if self.prev_img is not None:            
            self.filter_intra_keypoints(debug=debug, out_dir=out_dir)
            self.filter_inter_keypoints(debug=debug, out_dir=out_dir)

            flow_intra_inter0 = []
            flow_intra_inter1 = []
            flow_intra_inter3 = []

            flow_intra0 = []
            flow_intra1 = []

            flow_inter0 = []
            flow_inter3 = []
            # import pdb; pdb.set_trace()
            intra0 = []
            intra1 = []

            points013, terr0, terr1 = self.triangulate_3d_points(np.array(self.flow_kpt0).reshape(-1,2), np.array(self.flow_kpt3).reshape(-1,2))
            # import pdb; pdb.set_trace()
            num_flow_013 = 0
            for kp0, kp1, kp3, wp in zip(self.flow_kpt0, self.flow_kpt1, self.flow_kpt3, points013):
                if wp[2] < 0:
                    continue
                x0, y0 = kp0[0][0], kp0[0][1]
                x1, y1 = kp1[0][0], kp1[0][1]
                x3, y3 = kp3[0][0], kp3[0][1]
                if x1 > 0.0 and x3 > 0.0 and num_flow_013 < max_inter_pts: # intra and inter
                    num_flow_013 += 1
                    flow_intra_inter0.append(np.array([x0, y0]))
                    flow_intra_inter1.append(np.array([x1, y1]))
                    flow_intra_inter3.append(np.array([x3, y3]))
                    intra0.append(np.array([x0, y0]))
                    intra1.append(np.array([x1, y1]))
                    # n_res += 1
                elif x1 > 0.0 and x3 < 0.0: # intra only
                    flow_intra0.append(np.array([x0, y0]))
                    flow_intra1.append(np.array([x1, y1]))
                    intra0.append(np.array([x0, y0]))
                    intra1.append(np.array([x1, y1]))
                elif x1 < 0.0 and x3 > 0.0: # inter only
                    flow_inter0.append(np.array([x0, y0]))
                    flow_inter3.append(np.array([x3, y3]))

            if len(flow_intra_inter0) > 8:
                M, mask01 = cv2.findHomography(np.array(flow_intra_inter0), np.array(flow_intra_inter1), cv2.RANSAC, 0.5)
                M, mask03 = cv2.findHomography(np.array(flow_intra_inter0), np.array(flow_intra_inter3), cv2.RANSAC, 0.5)
                mask_flow013 = mask01 & mask03
                flow_intra_inter0 = [flow_intra_inter0[i] for i in range(len(flow_intra_inter0)) if mask_flow013[i] == 1]
                flow_intra_inter1 = [flow_intra_inter1[i] for i in range(len(flow_intra_inter1)) if mask_flow013[i] == 1]
                flow_intra_inter3 = [flow_intra_inter3[i] for i in range(len(flow_intra_inter3)) if mask_flow013[i] == 1]

            if len(flow_intra0) > 8:
                M, mask_flow01 = cv2.findHomography(np.array(flow_intra0), np.array(flow_intra1), cv2.RANSAC, 0.5)
                flow_intra0 = [flow_intra0[i] for i in range(len(flow_intra0)) if mask_flow01[i] == 1]
                flow_intra1 = [flow_intra1[i] for i in range(len(flow_intra1)) if mask_flow01[i] == 1]

            # import pdb; pdb.set_trace()


            self.flow_intra_inter0 = np.array(flow_intra_inter0)
            self.flow_intra_inter1 = np.array(flow_intra_inter1)
            self.flow_intra_inter3 = np.array(flow_intra_inter3)

            self.flow_intra0 = np.array(flow_intra0)
            self.flow_intra1 = np.array(flow_intra1)

            self.flow_inter0 = np.array(flow_inter0)
            self.flow_inter3 = np.array(flow_inter3)

            self.intra0 = np.array(intra0)
            self.intra1 = np.array(intra1)


            if debug:    
                self.debug_inter_keypoints(out_dir)
                self.debug_intra_keypoints(out_dir)
                print(self.img_idx, 'cam_'+ str(self.index ), 'intra_inter:' + str(len(self.flow_intra_inter0)), 'intra:' + str(len(self.flow_intra0)), 'inter:'+str(len(self.flow_inter0)))
                
class KiteVision:
    """Kite vision object"""
    def __init__(self, calib_file=None, num_cams=4, max_flow_kpts=64, is_kitti=True, kitti_path=None, kitti_seq=None):
        self.calib_file      = calib_file
        self.max_flow_kpts   = max_flow_kpts
        self.navcams         = []
        self.calib_file = calib_file
        self.cam_imgs   = None
        self.cam_setup  = [1, 0, 3, 2]
        self.num_imgs = 0
        self.focal = None
        self.pp = None
        self.annotations = None
        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.max_cams = 4
        self.camera_images = self.max_cams * [None]
        self.prev_scale = 1.0
        self.img_idx = -1
        self.least_square_conf = None
        if is_kitti:
            self.num_cams = 2
            kitti_pose_path = os.path.join(kitti_path, 'poses')
            self.annotations = os.path.join(kitti_pose_path, kitti_seq + '.txt')
            kitti_seq_path =  os.path.join(kitti_path, 'sequences')
            kitti_seq_path =  os.path.join(kitti_seq_path, kitti_seq)
            kitti_calib_file = os.path.join(kitti_seq_path, 'calib.txt')

            for c in range(self.num_cams):
                kitti_images_base= os.path.join(kitti_seq_path, 'image_' + str(c))
                img_files = glob.glob(kitti_images_base + '/*.png')
                img_files.sort(key=lambda f: int(filter(str.isdigit, f))) 
                self.camera_images[c] = img_files

            mtx, dist, rot, trans = load_kitti_config(kitti_calib_file)
            assert(len(self.camera_images[0]) == len(self.camera_images[1]))
            self.num_imgs  = len(self.camera_images[0])
            num_cams = 2
            with open(self.annotations) as f:
			    self.annotations = f.readlines()
            self.focal = kt_cam.fx
            self.pp = (kt_cam.cx, kt_cam.cy)
        else:
            mtx, dist, rot, trans = load_kv_nav_config(calib_file)

        self.num_cams        = num_cams
        self.cam_obs         = np.zeros([4, 2], dtype=np.int)

        rot_01, trans_01 = rot[1], trans[1]
        rot_10, trans_10 = invert_RT(rot[1], trans[1])

        rot_23, trans_23 = rot[3], trans[3]
        rot_32, trans_32 = invert_RT(rot[3], trans[3])
        
        rot[0], trans[0] = rot_01, trans_01
        rot[1], trans[1] = rot_10, trans_10
        rot[2], trans[2] = rot_23, trans_23
        rot[3], trans[3] = rot_32, trans_32

        self.calib_K = mtx
        self.calib_d = dist
        self.calib_R = rot
        self.calib_t = trans

        self.ego_R = cv2.Rodrigues(np.eye(3))[0]
        self.ego_t = np.zeros([3, 1])

        self.pose_R = cv2.Rodrigues(np.eye(3))[0]
        self.pose_t = np.zeros([3, 1])

        F = 4 * [None]

        F[0] = fundmental_matrix(rot[1], trans[1], mtx[0], mtx[1])
        F[1] = fundmental_matrix(rot[0], trans[0], mtx[1], mtx[0])
        F[2] = fundmental_matrix(rot[3], trans[3], mtx[2], mtx[3])
        F[3] = fundmental_matrix(rot[2], trans[2], mtx[3], mtx[2])


        for c in range(self.num_cams):
            self.navcams.append(navcam(c, self.cam_setup[c], F[c], mtx[c], dist[c], rot[c], trans[c], max_flow_kpts))
        for c in range(self.num_cams):
            self.navcams[c].set_its_pair(self.navcams[self.cam_setup[c]])

    def get_abs_scale(self, frame_id):  #specialized for KITTI odometry dataset
        ss = self.annotations[frame_id-1].strip().split()
        x_prev = float(ss[3])
        y_prev = float(ss[7])
        z_prev = float(ss[11])
        ss = self.annotations[frame_id].strip().split()
        x = float(ss[3])
        y = float(ss[7])
        z = float(ss[11])
        self.trueX, self.trueY, self.trueZ = x, y, z
        return np.sqrt((x - x_prev)*(x - x_prev) + (y - y_prev)*(y - y_prev) + (z - z_prev)*(z - z_prev))
       
    def update_keypoints(self):
        for i in range(self.num_cams):
            self.navcams[i].keypoint_detection()

    def update_sparse_flow(self):
        for i in range(self.num_cams):
            self.navcams[i].intra_sparse_optflow()
            self.navcams[i].inter_sparse_optflow()

    def filter_nav_keypoints(self, debug=False):
        for c in range(self.num_cams):
            self.navcams[c].filter_keypoints(debug=debug)

    def upload_images(self, imgs_x4):
        self.img_idx += 1
        for c in range(self.num_cams):        
            self.navcams[c].update_image(imgs_x4)

    def load_recorded_images(self, record_path='./', max_imgs=100):
        img_files = glob.glob(record_path + '*.jpg')
        img_files.sort(key=lambda f: int(filter(str.isdigit, f))) 
        cam_imgs = []
        for file in img_files:
            imgs_x4 = pil_split_rotate_navimage_4(file)
            cam_imgs.append(imgs_x4)
            self.num_imgs  += 1
            if self.num_imgs  > max_imgs:
                break
        self.cam_imgs = cam_imgs

    def read_kitti_image(self, img_idx=0):
        imgs_x4 = []
        for c in range(self.num_cams):
            im = Image.open(self.camera_images[c][img_idx])
            imgs_x4.append(np.asarray(im))
        return imgs_x4

    def load_kitti_images(self, img_path='/home/jzhang/vo_data/kitti/dataset/sequences/01/', max_imgs=100):
        cam_files = self.num_cams * [None]
        for c in range(self.num_cams):
            cam_path = img_path + 'image_' + str(c) + '/'
            img_files = glob.glob(cam_path + '*.png')
            img_files.sort(key=lambda f: int(filter(str.isdigit, f))) 
            cam_files[c] = img_files

        total_imgs = len(cam_files[0])
        max_imgs = min(total_imgs, max_imgs)
        cam_imgs = []
        for i in range(max_imgs):
            imgs_x4 = []
            for c in range(self.num_cams):
                im = Image.open(cam_files[c][i])
                imgs_x4.append(np.asarray(im))
            cam_imgs.append(imgs_x4)
            self.num_imgs += 1
        self.cam_imgs = cam_imgs

    def load_calib_images(self, calib_path='/home/jzhang/vo_data/SN40/calib_data/', max_imgs=100):
        cam_files = self.num_cams * [None]
        for c in range(self.num_cams):
            cam_path = calib_path + 'cam'+ str(c) + '/'
            img_files = glob.glob(cam_path + '*.jpg')
            img_files.sort(key=lambda f: int(filter(str.isdigit, f))) 
            cam_files[c] = img_files

        cam_imgs = []
        for i in range(max_imgs):
            imgs_x4 = []
            for c in range(self.num_cams):
                im = Image.open(cam_files[c][i])
                imgs_x4.append(np.asarray(im))
            cam_imgs.append(imgs_x4)
            self.num_imgs += 1
        self.cam_imgs = cam_imgs

    def local_ego_motion_solver(self, cam_list=[0], debug=False):
        for c in cam_list:
            if self.navcams[c].flow_intra_inter0 is None:
                continue
            if len(self.navcams[c].flow_intra_inter0) > 0:
                res = self.navcams[c].local_ego_motion_solver(debug=debug)

    def update_global_camera_pose_egomotion(self, R, t):
        self.ego_R = R
        self.ego_t = t
        self.prev_scale = norm(t) if norm(t) > 0.01 else self.prev_scale
        if self.img_idx <= 1:
            self.pose_R = R
            self.pose_t = t
        else:
            self.pose_t = self.pose_t + self.pose_R.dot(t) 
            self.pose_R = R.dot(self.pose_R)
        return self.pose_R, self.pose_t

    def get_egomotion(self):
        return self.ego_R, self.ego_t

    def get_global_camera_pose(self):
        return self.pose_R, self.pose_t

    def global_fun(self, x0, cam_obs, y_meas, cam_list=range(4)):
        if cam_list is None:
            return None
        num_cams = len(cam_list)
        rot_vecs   = x0[0:3]
        trans_vecs = x0[3:6]
        x0_offset = 6
        y_offset = 0
        cost_err = None
        for c in cam_list:
            n_obj_013, n_obj_01 = cam_obs[c]
            if n_obj_013 > 0:      
                points_013 = x0[x0_offset: x0_offset + 3 * n_obj_013].reshape(-1, 3)
                flow013_0  = y_meas[y_offset: y_offset+n_obj_013]
                flow013_1  = y_meas[y_offset+ n_obj_013: y_offset + 2 * n_obj_013]
                flow013_3  = y_meas[y_offset + 2 * n_obj_013: y_offset + 3 * n_obj_013]
                flow0_err = self.navcams[c].project_to_flow0(points_013) - flow013_0
                flow1_err = self.navcams[c].project_to_flow1(points_013, rot_vecs, trans_vecs) - flow013_1
                flow3_err = self.navcams[c].project_to_flow3(
                    points_013, self.navcams[c].calib_R2, self.navcams[c].calib_t) - flow013_3
                flow013_errs = np.vstack((flow0_err, flow1_err, flow3_err))
                x0_offset += 3 * n_obj_013
                y_offset += 3 * n_obj_013
                if cost_err is None:
                    cost_err = flow013_errs
                else:
                    cost_err = np.vstack([cost_err, flow013_errs])

            if n_obj_01 > 0:
                points_01  = x0[x0_offset: x0_offset+3*n_obj_01].reshape(-1, 3)
                flow01_0  = y_meas[y_offset: y_offset + n_obj_01]
                flow01_1  = y_meas[y_offset + n_obj_01: y_offset + 2 * n_obj_01]
            
                flow01_err0 = self.navcams[c].project_to_flow0(points_01) - flow01_0
                flow01_err1 = self.navcams[c].project_to_flow1(points_01, rot_vecs, trans_vecs) - flow01_1
                flow01_errs = np.vstack((flow01_err0, flow01_err1))
        
                x0_offset += 3 * n_obj_01
                y_offset += 2 * n_obj_01
                if cost_err is None:
                    cost_err = flow01_errs
                else:
                    cost_err = np.vstack([cost_err, flow01_errs])
            if cost_err is None:
                return None
        return cost_err.ravel()


    def global_ego_motion_solver(self, img_idx=None, cam_list=[0 , 1], est=None):
        if img_idx is None or img_idx == 0:
            return None, None

        # Initialize the initial egomotion estimation by using the monocamera 
        est_R = np.random.normal(0, 0.01, [3,1])
        est_t = np.random.normal(0, 0.01, [3,1])
        num_cams = len(cam_list)
        mono_R, mono_t = self.navcams[0].mono_ego_motion_estimation()
        if mono_R is not None:
            est_R = mono_R
            est_t = self.prev_scale * mono_t if self.prev_scale is not None else mono_t

        if est is not None:
            est_R = est[0]
            est_t = est[1]
        cam_obs = np.zeros([self.num_cams, 2], dtype=np.int)
        y_meas = None

        x0 = np.vstack([est_R, est_t])

        for c in cam_list:
            if self.navcams[c].flow_intra_inter0 is None:
                continue
            cur_cam = self.navcams[c]
            n_obs_i = cur_cam.flow_intra_inter0.shape[0]
            n_obs_j = cur_cam.flow_intra0.shape[0]

            if n_obs_i > 0:
                flow0 = cur_cam.flow_intra_inter0
                flow1 = cur_cam.flow_intra_inter1
                flow3 = cur_cam.flow_intra_inter3

                points013, terr013_0, terr013_1 = cur_cam.triangulate_3d_points(flow0, flow3)
                flow013_z = np.vstack([flow0, flow1, flow3])
                y_meas = flow013_z if y_meas is None else np.vstack([y_meas, flow013_z])

                x0 = np.vstack([x0, points013.ravel().reshape(-1, 1)])

            if n_obs_j > 0:
                flow0 = cur_cam.flow_intra0
                flow1 = cur_cam.flow_intra1
                points01, terr01_0, terr01_1 =  cur_cam.triangulate_3d_points_intra(flow0, flow1, cv2.Rodrigues(est_R)[0], est_t)
                x0 = np.vstack([x0, points01.ravel().reshape(-1, 1)])
                flow01_z = np.vstack([flow0, flow1])
                y_meas = flow01_z if y_meas is None else np.vstack([y_meas, flow01_z])

            cam_obs[c][0] = n_obs_i
            cam_obs[c][1] = n_obs_j

        if y_meas is None or y_meas.shape[0] < 9:
            R, t = self.update_global_camera_pose_egomotion(cv2.Rodrigues(est_R)[0], est_t.reshape(3,))
            return R, t

        x0 = x0.flatten()
        sparse_A = global_bundle_adjustment_sparsity_opt(cam_obs, n_cams=self.num_cams) 

        ls_pars = dict(jac_sparsity=sparse_A,
                    max_nfev=6, 
                    verbose=0,
                    x_scale='jac',
                    jac='2-point',
                    ftol=2e-3, 
                    xtol=2e-3,
                    gtol=2e-3,
                    method='trf')

        # err0 = self.global_fun(x0, cam_obs, y_meas, cam_list)
        try:
            res = least_squares(self.global_fun, x0, args=(cam_obs, y_meas, cam_list), **ls_pars)
        except:
            import pdb; pdb.set_trace()

        err1 = self.global_fun(res.x, cam_obs, y_meas, cam_list)
        err_level = norm(err1)

        if res is None:
            return self.ego_R, self.ego_t

        R = cv2.Rodrigues(res.x[0:3])[0]
        t = res.x[3:6]
        if self.least_square_conf is None:
            self.least_square_conf = err_level
        avg_least_square_conf = self.least_square_conf / (self.img_idx)


        print('gba:',self.navcams[0].img_idx, 'est_rot', res.x[0:3], 'est_tras', res.x[3:6], 'conf', norm(err1), avg_least_square_conf)

    
        if err_level > 5 * avg_least_square_conf:
            return self.pose_R, self.pose_t
        else:
            self.least_square_conf += err_level

        avg_least_square_conf = self.least_square_conf / self.img_idx
        if t[2] > 5:
            import pdb; pdb.set_trace()
        pose_R, pose_t = self.update_global_camera_pose_egomotion(R, t)
        # pdb.set_trace()
        # plt.plot(err0); plt.plot(err1); plt.show();
        return pose_R, pose_t 
        


# kv = KiteVision(calib_file='/home/jzhang/vo_data/SN40/nav_calib.cfg', max_flow_kpts=128, is_kitti=True)
# bad seq, 03, 05 

seq = '10'
kv = KiteVision(kitti_path='/home/jzhang/vo_data/kitti/dataset', kitti_seq=seq, calib_file=None, max_flow_kpts=64, is_kitti=True)
# kv.load_recorded_images('/home/jzhang/vo_data/SN40/videos/output01.TS.JPEGS/', 100)
# kv.load_calib_images()
# kv.load_kitti_images('/home/jzhang/vo_data/kitti/dataset/sequences/02/', 1500)

# gt_rot, gt_tr = load_kitti_poses()
pose = [0, 0, 0]
px = []
py = []
pz = []

# import pdb; pdb.set_trace()
cur_R = None
cur_t = None
traj = np.zeros((1000, 1000, 3), dtype=np.uint8)

for img_id in range(kv.num_imgs):
    camera_images = kv.read_kitti_image(img_id)
    kv.upload_images(camera_images)
    kv.update_keypoints()
    kv.update_sparse_flow()
    kv.filter_nav_keypoints(debug=False)
    abs_scale = kv.get_abs_scale(img_id)

    # stereo_rot, stereo_tr = kv.navcams[0].local_ego_motion_solver(debug=False)
    # stereo_rot2, stereo_tr2 = kv.navcams[0].local_ego_motion_solver(debug=False)
    # stereo_ego_rot, stereo_ego_tr = kv.navcams[0].get_egomotion()
    global_rot, global_tr = kv.global_ego_motion_solver(img_id, cam_list=[0, 1])

    if img_id >= 1:
        x, y, z = global_tr[0], global_tr[1], global_tr[2]
        # x1, y1, z1 = stereo_tr[0], stereo_tr[1], stereo_tr[2]
    else:
        x, y, z = 0., 0., 0.
        x1, y1, z1 = 0., 0., 0.


    print('===================')
    print('goundt', kv.trueX, kv.trueZ)
    print('global', x, z)
    print('stereo', x1, z1)
    print('===================')

    draw_ofs_x = 500
    draw_ofs_y = 500

    draw_x0, draw_y0 = int(x)+draw_ofs_x, int(z)+draw_ofs_y    
    # draw_x1, draw_y1 = int(x1)+draw_ofs_x, int(z1)+draw_ofs_y
    true_x, true_y = int(kv.trueX)+draw_ofs_x, int(kv.trueZ)+draw_ofs_y

    cv2.circle(traj, (draw_x0, draw_y0), 1, (255, 0,0), 1)
    # cv2.circle(traj, (draw_x1, draw_y1), 1, (0,255,0), 1)
    cv2.circle(traj, (true_x,true_y), 1, (255,255,255), 2)
    cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
    text = "Img:%3d, Coordinates: x=%.2fm y=%.2fm z=%.2fm"%(img_id, x1, y1, z1)
    cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
    # img1 = cv2.resize(cv2.cvtColor(camera_images[0], cv2.COLOR_GRAY2BGR), (640, 480))
    # img2 = cv2.resize(cv2.cvtColor(camera_images[1], cv2.COLOR_GRAY2BGR), (640, 480))
    # img = concat_images(img1, img2)        
    # cv2.imshow('Navigation cameras', img)
    cv2.imshow('Trajectory' + seq, traj)
    cv2.waitKey(1)
traj_name = 'seq_' + seq + '_' + datetime.now().strftime('%Y-%m-%d-%H:%M:%S') + '.png'
cv2.imwrite(traj_name, traj)

