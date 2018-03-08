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
from numpy.linalg import norm
from scipy.sparse import lil_matrix
import time
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
 
# from matplotlib import pyplot as plt
from math import hypot
from math import sqrt

np.set_printoptions(suppress=True)

def load_nav_config(cfg_file='nav_calib.cfg', num_cams=4):
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


def sparse_optflow(curr_im, target_im, flow_kpt0):
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (8, 8),
                    maxLevel = 4,
                    minEigThreshold=1e-4,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 5, 0.01))
    # perform a forward match
    flow_kpt1, st, err = cv2.calcOpticalFlowPyrLK(curr_im, target_im, flow_kpt0, None, **lk_params)
    # perform a reverse match
    flow_kpt2, st, err = cv2.calcOpticalFlowPyrLK(target_im, curr_im, flow_kpt1, None, **lk_params)
    return flow_kpt0, flow_kpt1, flow_kpt2 

def sparse_optflow2(curr_im, target_im, flow_kpt0):
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (16, 16),
                    maxLevel = 5,
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
def bundle_adjustment_sparsity(n_obs, n_poses=1):
    m = n_obs * 6
    n = n_poses * 6 + n_obs * 3
    A = lil_matrix((m, n), dtype=int)
    # fill in the sparse struct of A
    i = np.arange(n_obs)

    # fill the flow0 entries
    for k in range(3):
        A[2 * i,     n_poses * 6 + i * 3 + k] = 1
        A[2 * i + 1, n_poses * 6 + i * 3 + k] = 1  
        
    # fill the flow1 entries
    # fill the entries for egomotion
    for k in range(6):
        A[n_obs * 2 + 2 * i, k] = 1
        A[n_obs * 2 + 2 * i + 1, k] = 1

    for k in range(3):
        A[n_obs * 2 + 2 * i,     n_poses * 6 + i * 3 + k] = 1
        A[n_obs * 2 + 2 * i + 1, n_poses * 6 + i * 3 + k] = 1

    # fill the flow3 entries 
    for k in range(3):
        A[n_obs * 4 + 2 * i,     n_poses * 6 + i * 3 + k] = 1
        A[n_obs * 4 + 2 * i + 1, n_poses * 6 + i * 3 + k] = 1
  
    return A

def bundle_adjustment_sparsity2(n_obs, n_poses=0):
    m = n_obs * 4
    n = n_poses * 6 + n_obs * 3
    A = lil_matrix((m, n), dtype=int)
    # fill in the sparse struct of A
    i = np.arange(n_obs)

    # fill the flow0 entries 
    for k in range(3):
        A[2 * i,     n_poses * 6 + i * 3 + k] = 1
        A[2 * i + 1, n_poses * 6 + i * 3 + k] = 1      

    # fill the flow3 entries 
        A[n_obs * 2 + 2 * i,     n_poses * 6 + i * 3 + k] = 1
        A[n_obs * 2 + 2 * i + 1, n_poses * 6 + i * 3 + k] = 1
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

        m_offset_i = (3*n_obs_i + 2 * n_obs_j) * 2
        n_offset_i = (n_obs_i + n_obs_j) * 3

    return A


class navcam:
    def __init__(self,  index, stereo_pair_idx, E, F, intrinsic_mtx,  intrinsic_dist,  extrinsic_rot, extrinsic_trans, max_flow_kpts=64):
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

        self.ego_R = None # camera pose rotation in 
        self.ego_t = None 
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

        self.curr_img  = None
        self.curr_stereo_img  = None
        self.prev_img  = None
        self.index   = index

        self.img_idx = None
        self.E = E
        self.F = F
        self.proj_mtx = None

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

    def init_flow013_camera_points(self):
        if len(self.flow_intra_inter0) == 0:
            return None, 1.0
        disparity = (self.flow_intra_inter0 - self.flow_intra_inter3)[:,0]
        depth_est = 0.146 * self.calib_K[0][0] / abs(disparity)
        flow0 = cv2.convertPointsToHomogeneous(self.flow_intra_inter0)
        flow0 = flow0.reshape(flow0.shape[0], flow0.shape[2]).T
        inv_K = inv(self.calib_K)
        camera_points = np.dot(inv_K, flow0).T
        for i in range(len(depth_est)):
            camera_points[i] *=  depth_est[i]
        return camera_points, depth_est

    def init_flow01_camera_points(self, est_depth=1.0):
        if self.flow_intra0 is None:
            return None
        if len(self.flow_intra0) == 0:
            return None
        flow0 = cv2.convertPointsToHomogeneous(self.flow_intra0)
        flow0 = flow0.reshape(flow0.shape[0], flow0.shape[2]).T
        inv_K = inv(self.calib_K)
        camera_points = est_depth * np.dot(inv_K, flow0).T
        return camera_points

    def triangulate_3d_points(self):
        left_p = self.projection_mtx()
        right_p = self.stereo_pair_cam.projection_mtx()
        left_kpts = self.flow_intra_inter0
        right_kpts = self.flow_intra_inter3
        if right_kpts is None:
            return None
        if len(left_kpts) < 1 or len(right_kpts) < 1:
            return None
        disparity = (self.flow_intra_inter0 - self.flow_intra_inter3)[:,0]
        depth_est = 0.146 * self.calib_K[0][0] / abs(disparity)
        try:
            scene_pts = cv2.triangulatePoints(left_p, right_p, left_kpts.T, right_kpts.T).T
        except:
            return None
        # left_proj = np.dot(left_p, scene_pts).T
        # left_proj = left_proj[:,0:2] / left_proj[:,2][:,np.newaxis]

        # right_proj = np.dot(right_p, scene_pts.T).T
        # right_proj = right_proj[:,0:2] / right_proj[:,2][:,np.newaxis]

        points_cam_cur = scene_pts[:,0:3] / scene_pts[:,3][:,np.newaxis]
        # points_cam_cur = points_cam_cur / points_cam_cur[:,2][:,np.newaxis]
        # points_cam_cur =  points_cam_cur * depth_est[:,np.newaxis]
        return points_cam_cur

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

        flow0_err = flow013_0 - self.project_to_flow0(points_013)
        flow1_err = flow013_1 - self.project_to_flow1(points_013, rot_vecs, trans_vecs)
        flow3_err = flow013_3 - self.project_to_flow3(points_013, self.calib_R2, self.calib_t)
        
        errs = np.vstack((flow1_err, flow3_err, flow0_err))

        if n_obj_01 > 0:
            points_01  = x0[6 + 3 * n_obj_013 : 6 + 3 * n_obj_013 + 3 * n_obj_01].reshape(-1, 3)
            flow01_0  = y_meas[3 * n_obj_013  : 3 * n_obj_013 + n_obj_01]
            flow01_1  = y_meas[3 * n_obj_013 + n_obj_01: 3 * n_obj_013 + 2*n_obj_01]
        
            flow01_err0 = flow01_0 - self.project_to_flow0(points_01)
            flow01_err1 = flow01_1 - self.project_to_flow1(points_01, rot_vecs, trans_vecs)
            errs = np.vstack((errs, flow01_err0, flow01_err1))
        return errs.ravel()

    def local_ego_motion_solver(self):
        n_obs_i = self.flow_intra_inter0.shape[0]
        n_obs_j = self.flow_intra0.shape[0]
        # n_obs_j = 0
        self.cam_obs = np.zeros([4,2], dtype=np.int)
        self.cam_obs[self.index][0] = n_obs_i
        self.cam_obs[self.index][1] = n_obs_j

        # sparse_A = bundle_adjustment_sparsity2(n_obs_i, 0)

        sparse_A = global_bundle_adjustment_sparsity(self.cam_obs, n_cams=4)        

        self.ego_R = cv2.Rodrigues(np.eye(3))[0]
        self.ego_t = np.zeros([3, 1])
        y_meas = None
        x0 = np.vstack([self.ego_R, self.ego_t])
        # x0 = None
        if n_obs_i > 0:
            points013_good = self.triangulate_3d_points() 
            points013, est_depth = self.init_flow013_camera_points()
            points013_flatten = points013.ravel().reshape(-1, 1)
            y_meas = np.vstack([self.flow_intra_inter0, self.flow_intra_inter1, self.flow_intra_inter3])
            if x0 is None:
                x0 = points013_flatten
            else:
                x0 = np.vstack([x0, points013_flatten])

        if n_obs_j > 0:
            points01 = self.init_flow01_camera_points(np.average(est_depth))
            points01_flatten = points01.ravel().reshape(-1, 1)
            if x0 is None:
                x0 = points01_flatten
            else:
                x0 = np.vstack([x0, points01_flatten])
            y_meas = np.vstack([y_meas, self.flow_intra0, self.flow_intra1])

        x0 = x0.flatten()

        t0 = time.time()
        err0 = self.fun(x0, self.cam_obs[self.index], y_meas)
        res = least_squares(self.fun, x0, jac_sparsity=sparse_A, verbose=2, x_scale='jac', ftol=1e-4, method='trf', args=(self.cam_obs[self.index], y_meas))
        err1 = self.fun(res.x, self.cam_obs[self.index], y_meas)
        n_obj_013, n_obj_01 = self.cam_obs[self.index] 
        pdb.set_trace()
        # plt.plot(err0); plt.plot(err1); plt.show();
        t1 = time.time()
        return res


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
            k0, k3, k4 = sparse_optflow2(self.curr_img, self.curr_stereo_img, self.flow_kpt0)
            self.flow_kpt3 = k3
            self.flow_kpt4 = k4

    def filter_intra_keypoints(self, debug=True, out_dir='/tmp'):
        img = None
        if self.prev_img is not None:
            img = cv2.cvtColor(self.curr_img, cv2.COLOR_GRAY2BGR)
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
                color = tuple(np.random.randint(0, 255, 3).tolist())
                img = cv2.circle(img,(x1, y1), 3, color, 1)
            if debug and img is not None:
                out_img_name = os.path.join(out_dir, 'cam_' + str(self.index) + '_intra_' + str(self.img_idx)+'.jpg')
                # print('writing...' + out_img_name)
                # cv2.imwrite(out_img_name, img)

    def filter_inter_keypoints(self, debug=True, out_dir='/tmp'):
        img = None
        if self.prev_img is not None:
            img1 = cv2.cvtColor(self.curr_img, cv2.COLOR_GRAY2BGR)
            img2 = cv2.cvtColor(self.curr_stereo_img, cv2.COLOR_GRAY2BGR)
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
                color = tuple(np.random.randint(0,255,3).tolist())
                img1 = cv2.circle(img1,(x1, y1), 6, color,1)
                img2 = cv2.circle(img2,(x3, y3), 6, color,1)
                img = concat_images(img1, img2)
            if debug and img is not None:
                out_img_name = os.path.join(out_dir, 'cam_' + str(self.index) + '_inter_' + str(self.img_idx)+'.jpg')
                # print('writing...' + out_img_name)
                cv2.imwrite(out_img_name, img)

    def final_kpts_to_array(self):
        self.flow_intra_inter0 = np.asarray(self.flow_intra_inter0) # original keypoints
        self.flow_intra_inter1 = np.asarray(self.flow_intra_inter1) # intra match
        self.flow_intra_inter3 = np.asarray(self.flow_intra_inter3) # inter match

        self.flow_intra0 = np.asarray(self.flow_intra0) # original keypoints
        self.flow_intra1 = np.asarray(self.flow_intra1) # intra match

        self.flow_inter0 = np.asarray(self.flow_inter0) # original keypoints
        self.flow_inter3 = np.asarray(self.flow_inter3) # intra match

    def filter_keypoints(self, debug=True, out_dir='/tmp'):
        if self.prev_img is not None:            
            self.filter_intra_keypoints(debug=False, out_dir=out_dir)
            self.filter_inter_keypoints(debug=debug, out_dir=out_dir)
            flow_intra_inter0 = []
            flow_intra_inter1 = []
            flow_intra_inter3 = []

            flow_intra0 = []
            flow_intra1 = []

            flow_inter0 = []
            flow_inter3 = []

            n_res = 0
            for kp0, kp1, kp3 in zip(self.flow_kpt0, self.flow_kpt1, self.flow_kpt3):
                # if n_res > 10:
                #     break
                x0, y0 = kp0[0][0], kp0[0][1]
                x1, y1 = kp1[0][0], kp1[0][1]
                x3, y3 = kp3[0][0], kp3[0][1]
                if x1 > 0.0 and x3 > 0.0: # intra and inter
                    flow_intra_inter0.append(np.array([x0, y0]))
                    flow_intra_inter1.append(np.array([x1, y1]))
                    flow_intra_inter3.append(np.array([x3, y3]))
                    n_res += 1
                elif x1 > 0.0 and x3 < 0.0: # intra only
                    flow_intra0.append(np.array([x0, y0]))
                    flow_intra1.append(np.array([x1, y1]))
                elif x1 < 0.0 and x3 > 0.0: # inter only
                    flow_inter0.append(np.array([x0, y0]))
                    flow_inter3.append(np.array([x3, y3]))
                    
            self.flow_intra_inter0 = np.array(flow_intra_inter0)
            self.flow_intra_inter1 = np.array(flow_intra_inter1)
            self.flow_intra_inter3 = np.array(flow_intra_inter3)

            self.flow_intra0 = np.array(flow_intra0)
            self.flow_intra1 = np.array(flow_intra1)

            self.flow_inter0 = np.array(flow_inter0)
            self.flow_inter3 = np.array(flow_inter3)
            # if debug:
            # print(self.img_idx, 'cam_'+ str(self.index ), 'intra_inter:' + str(len(self.flow_intra_inter0)), 'intra:' + str(len(self.flow_intra0)), 'inter:'+str(len(self.flow_inter0)))


class KiteVision:
    """Kite vision object"""
    def __init__(self, calib_file='/home/jzhang/vo_data/SN40/nav_calib.cfg', max_flow_kpts=64):
        self.calib_file      = calib_file
        self.max_flow_kpts   = max_flow_kpts
        self.num_cams        = 4
        self.cam_obs = np.zeros([self.num_cams, 2], dtype=np.int)

        self.navcams         = []
        self.calib_file = calib_file
        self.cam_imgs   = None
        self.cam_setup  = [1, 0, 3, 2]
        self.num_imgs = 0
        mtx, dist, rot, trans = load_nav_config(calib_file)

        rot[0], trans[0] = invert_RT(rot[1], trans[1])
        rot[2], trans[2] = invert_RT(rot[3], trans[3])

        self.calib_K = mtx
        self.calib_d = dist
        self.calib_R = rot
        self.calib_t = trans
        self.ego_R = cv2.Rodrigues(np.eye(3))[0]
        self.ego_t = np.zeros([3, 1])

        self.essential_mtx = self.num_cams * [None]
        self.fundmental_matrix = self.num_cams * [None]


        self.fundmental_matrix[0] = fundmental_matrix(rot[1], trans[1], mtx[0], mtx[1])
        self.fundmental_matrix[1] = fundmental_matrix(rot[0], trans[0], mtx[1], mtx[0])

        self.fundmental_matrix[2] = fundmental_matrix(rot[3], trans[3], mtx[2], mtx[3])
        self.fundmental_matrix[3] = fundmental_matrix(rot[2], trans[2], mtx[3], mtx[2])

        self.essential_mtx[0] = essential(rot[1], trans[1])
        self.essential_mtx[1] = essential(rot[0], trans[0]) 
        self.essential_mtx[2] = essential(rot[3], trans[3])
        self.essential_mtx[3] = essential(rot[2], trans[2]) 

        for c in range(self.num_cams):
            self.navcams.append(navcam(c, self.cam_setup[c], self.essential_mtx[c], self.fundmental_matrix[c], mtx[c], dist[c], rot[c], trans[c], max_flow_kpts))
        for c in range(self.num_cams):
            self.navcams[c].set_its_pair(self.navcams[self.cam_setup[c]])

    def update_keypoints(self):
        for i in range(self.num_cams):
            self.navcams[i].keypoint_detection()

    def update_sparse_flow(self):
        for i in range(self.num_cams):
            self.navcams[i].intra_sparse_optflow()
            self.navcams[i].inter_sparse_optflow()

    def filter_nav_keypoints(self):
        for c in range(self.num_cams):
            self.navcams[c].filter_keypoints(debug=True)

    def upload_images(self, imgs_x4):
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

    def local_ego_motion_solver(self):
        for c in range(self.num_cams):
            if self.navcams[c].flow_intra_inter0 is None:
                continue
            if len(self.navcams[c].flow_intra_inter0) > 8:
                res = self.navcams[c].local_ego_motion_solver()
                wx, wy, wz, tx, ty, tz = res.x[0:6]
                egpmotion_est = ", ".join("%.4f" % f for f in res.x[0:6])
                print(self.navcams[c].img_idx, 'cam_'+ str(self.navcams[c].index ), egpmotion_est)

    def global_fun(self, x0, cam_obs, y_meas):
        rot_vecs   = x0[0:3]
        trans_vecs = x0[3:6]
        x0_offset = 6
        y_offset = 0
        cost_err = None
        for c in range(self.num_cams):  
            n_obj_013, n_obj_01 = cam_obs[c]
            if n_obj_013 > 0:      
                points_013 = x0[x0_offset: x0_offset + 3 * n_obj_013].reshape(-1, 3)
                flow013_0  = y_meas[y_offset: y_offset+n_obj_013]
                flow013_1  = y_meas[y_offset+ n_obj_013: y_offset + 2 * n_obj_013]
                flow013_3  = y_meas[y_offset + 2 * n_obj_013: y_offset + 3 * n_obj_013]

                flow0_err = flow013_0 - self.navcams[c].project_to_flow0(points_013)
                flow1_err = flow013_1 - self.navcams[c].project_to_flow1(points_013, rot_vecs, trans_vecs)
                flow3_err = flow013_3 - self.navcams[c].project_to_flow3(points_013, self.navcams[c].calib_R2, self.navcams[c].calib_t)
                flow013_errs = np.vstack((flow0_err, flow1_err, flow3_err))
                x0_offset += 3 * n_obj_013
                y_offset += 3*n_obj_013
                if cost_err is None:
                    cost_err = flow013_errs
                else:
                    cost_err = np.vstack([cost_err, flow013_errs])

            if n_obj_01 > 0:
                points_01  = x0[x0_offset: x0_offset+3*n_obj_01].reshape(-1, 3)
                flow01_0  = y_meas[y_offset: y_offset + n_obj_01]
                flow01_1  = y_meas[y_offset + n_obj_01: y_offset + 2 * n_obj_01]
            
                flow01_err0 = flow01_0 - self.navcams[c].project_to_flow0(points_01)
                flow01_err1 = flow01_1 - self.navcams[c].project_to_flow1(points_01, rot_vecs, trans_vecs)
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

    def global_ego_motion_solver(self):
        self.cam_obs = np.zeros([self.num_cams,2 ], dtype=np.int)
        y_meas = None
        # Initialize the initial egomotion estimation
        self.ego_R = cv2.Rodrigues(np.eye(3))[0]
        self.ego_t = np.zeros([3, 1])
        x0 = np.vstack([self.ego_R, self.ego_t])

        for c in range(self.num_cams):
            if self.navcams[c].flow_intra_inter0 is None:
                continue
            n_obs_i = self.navcams[c].flow_intra_inter0.shape[0]
            n_obs_j = self.navcams[c].flow_intra0.shape[0]
            
            est_depth = 1.0
            if n_obs_i > 0:
                points013, est_depth = self.navcams[c].init_flow013_camera_points()
                points013_flatten = points013.ravel().reshape(-1, 1)
                cam_flow013_measurement = np.vstack([self.navcams[c].flow_intra_inter0, self.navcams[c].flow_intra_inter1, self.navcams[c].flow_intra_inter3])
                if y_meas is None:
                    y_meas = cam_flow013_measurement
                else:
                    y_meas = np.vstack([y_meas, cam_flow013_measurement])
                x0 = np.vstack([x0, points013_flatten])

            if n_obs_j > 0:
                points01 = self.navcams[c].init_flow01_camera_points(est_depth)
                points01_flatten = points01.ravel().reshape(-1, 1)
                x0 = np.vstack([x0, points01_flatten])
                cam_flow01_measurement = np.vstack([self.navcams[c].flow_intra0, self.navcams[c].flow_intra1])
                if y_meas is None:
                    y_meas = cam_flow01_measurement
                else:
                    y_meas = np.vstack([y_meas, cam_flow01_measurement])

            self.cam_obs[c][0] = n_obs_i
            self.cam_obs[c][1] = n_obs_j

        if y_meas is None:
            return None
        x0 = x0.flatten()
        sparse_A = global_bundle_adjustment_sparsity(self.cam_obs, n_cams=self.num_cams) 
        err0 = self.global_fun(x0, self.cam_obs, y_meas)
        res = least_squares(self.global_fun, x0, jac_sparsity=sparse_A, verbose=2, x_scale='jac', ftol=1e-4, method='trf', args=(self.cam_obs, y_meas))
        err1 = self.global_fun(res.x, self.cam_obs, y_meas)
        pdb.set_trace()
        # plt.plot(err0); plt.plot(err1); plt.show();
        # self.num_cams = 4
        return res
        
    def triangulate_3d_points(self):
        for left_cam in range(self.num_cams):
            right_cam = self.cam_setup[left_cam]
            left_p = self.navcams[left_cam].projection_mtx()
            right_p = self.navcams[right_cam].projection_mtx()
            left_kpts = self.navcams[left_cam].flow_kpt0
            right_kpts = self.navcams[left_cam].flow_kpt3
            if right_kpts is None:
                return None
            left_kpts = left_kpts.reshape(left_kpts.shape[0],left_kpts.shape[2])
            right_kpts = right_kpts.reshape(right_kpts.shape[0],right_kpts.shape[2])

            kpts1 = left_kpts[right_kpts[:,0] > 0.0]
            kpts2 = right_kpts[right_kpts[:,0] > 0.0]
            if len(kpts1) < 1 or len(kpts2) < 1:
                return None
            scene_pts = cv2.triangulatePoints(left_p, right_p, kpts1.T, kpts2.T)
            left1 = np.dot(left_p, scene_pts)
            return scene_pts
        

kv = KiteVision(calib_file='/home/jzhang/vo_data/SN11/nav_calib.cfg', max_flow_kpts=64)
kv.load_recorded_images('/home/jzhang/vo_data/SN11.TS.JPEGS/', 100)

for i in range(kv.num_imgs):
    kv.upload_images(kv.cam_imgs[i])
    kv.update_keypoints()
    kv.update_sparse_flow()
    kv.filter_nav_keypoints()
    kv.local_ego_motion_solver()
    # res = kv.global_ego_motion_solver()
    # if res and i > 1:
    #     # pdb.set_trace()
    #     rot = res.x[0:3]
    #     trans = res.x[3:6]
    #     print(rot, trans)




        

