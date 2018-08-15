
import numpy as np
import glob, pdb, math

import os, io, libconf, copy
import cv2
from PIL import Image

from numpy.linalg import inv, pinv, norm

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

import time


import matplotlib.pyplot as plt

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


def concat_images_list(im_list):
    """
    Combines a list of images vertically. Those images must have the same shape
    """
    n_images = len(im_list)
    if n_images == 0:
        return None
    if n_images == 1:
        return im_list[0]

    im_height, im_width = im_list[0].shape[:2]
    concat_height = n_images * im_height
    concat_width = im_width
    # new_img = np.zeros(shape=(concat_height, concat_width, 3), dtype=np.uint8)
    new_img = np.vstack(im_list)
    return new_img

def load_kite_config(cfg_file='nav_calib.cfg', num_cams=4):
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
    scaling_mtx = np.zeros([3, 3])

    scaling_mtx[0][0] = 1280.0 / 1241.0
    scaling_mtx[1][1] = 480.0 / 376.0
    scaling_mtx[2][2] = 1.0

    for cam_id, line in enumerate(lines):
        P0 = line.split(' ')[1:]
        P0 = [float(i) for i in P0]
        P0 = np.asarray(P0).reshape(3, -1)
        mtx = P0[0:3, 0:3]
        rot = np.eye(3)
        # import pdb; pdb.set_trace()
        trans = P0[:,3].reshape(3,1)
        trans = np.dot(inv(mtx), trans)
        mtx = np.dot(scaling_mtx, mtx)
        cam_matrix[cam_id] = mtx
        # import pdb ; pdb.set_trace()
        dist[cam_id] = None
        cam_rot[cam_id] = rot
        cam_trans[cam_id] = trans
    return cam_matrix, dist, cam_rot, cam_trans

def load_camera_calib(dataset = 'kitti', calib_file=None, num_cams=4):
    if not os.path.exists(calib_file):
        raise ValueError(calib_file + ' does not exsit')
        return
    if dataset.lower() == 'kitti':
        return load_kitti_config(calib_file, num_cams)
    elif dataset.lower() == 'kite':
        return load_kite_config(calib_file, num_cams) 
    else:
        raise ValueError('unsupported camera calibration')

def get_kitti_calib_path(kitti_base=None, data_seq='01'):
    seq_path =  os.path.join(kitti_base, 'sequences')
    seq_path =  os.path.join(seq_path, data_seq)
    return os.path.join(seq_path, 'calib.txt')

def get_kitti_ground_truth(kitti_base=None, data_seq=None):
    assert(data_seq is not None)
    pose_path = os.path.join(kitti_base, 'poses')
    pose_path = os.path.join(pose_path, data_seq + '.txt')
    if not os.path.exists(pose_path):
        return None
    with open(pose_path) as f:
	    annotations = f.readlines()
    return annotations

def load_calib_images(self, calib_img_path='~/vo_data/SN40/calib_data/', num_cams=4, max_imgs=100):
    cam_files = num_cams * [None]
    for c in range(num_cams):
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
    return cam_imgs

def read_kitti_image(camera_images, num_cams, img_idx=0):
    imgs_x4 = []
    for c in range(num_cams):
        im = Image.open(camera_images[c][img_idx])
        resized_image = im
        resized_image = im.resize((1280, 480), Image.BICUBIC);
        imgs_x4.append(np.asarray(resized_image))
    return imgs_x4

def read_kite_image(camera_images, num_cams=None, img_idx=0):
    imgs_x4 = pil_split_rotate_navimage_4(camera_images[img_idx])
    return imgs_x4

def get_kitti_image_files(kitti_base=None, data_seq='01', max_cam=4):
    seq_path =  os.path.join(kitti_base, 'sequences')
    seq_path =  os.path.join(seq_path, data_seq)
    camera_images = []
    for c in range(max_cam):
        images_base= os.path.join(seq_path, 'image_' + str(c))
        if not os.path.exists(images_base):
            continue
        img_files = glob.glob(images_base + '/*.png')
        img_files = sorted(img_files)
        resize_folder = os.path.join(seq_path, 'resized_image_' + str(c))
        resize_images(img_files, resize_folder, (1280, 480))
        camera_images.append(img_files)
    return camera_images

def resize_images(image_list,  output_path, target_size = (1280, 480)):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    else:
        return
    for im_file in image_list:
        im_name = im_file.split('/')[-1]
        im_name = os.path.join(output_path, im_name)
        im = Image.open(im_file)
        resized_im = im.resize(target_size, Image.BICUBIC);
        resized_im.save(im_name)

def get_kite_image_files(kite_base=None, data_seq=None, num_cam=4):
    # import pdb ; pdb.set_trace()
    img_files = glob.glob(kite_base + '/*.jpg')
    img_files.sort(key=lambda f: int(filter(str.isdigit, f))) 
    return img_files

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

def fundamental_matrix(R01, T01, K0, K1):
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

def construct_projection_mtx(K1, K2, R, t):
    left_T = np.eye(4)[:3]
    left_mtx = np.dot(K1, left_T)
    right_T = np.zeros([3,4])
    right_T[0:3,0:3] = R
    right_T[:,3][:3] = t.ravel()
    right_mtx = np.dot(K2, right_T)
    return left_mtx, right_mtx

def triangulate_3d_points(left_kpts, right_kpts, left_K, right_K, rotation, translation):
    left_p, right_p = construct_projection_mtx(left_K, right_K, rotation, translation)

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

def rotate_and_translate(points, rotation_vector, trans_vector):
    """Rotate points by given rotation vectors.
    Rodrigues' rotation formula is used.
    """
    if rotation_vector is None and trans_vector is None:
        return points
    if rotation_vector is not None:
        rotation_vector = rotation_vector.reshape(1, -1)
        theta = np.linalg.norm(rotation_vector, axis=1)[:, np.newaxis]
        with np.errstate(invalid='ignore'):
            v = rotation_vector / theta
            v = np.nan_to_num(v)
        dot = np.sum(points * v, axis=1)[:, np.newaxis]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        points = cos_theta * points + sin_theta * np.cross(v, points) + dot * (1 - cos_theta) * v

    if trans_vector is not None:
        trans_vector = trans_vector.reshape(1, -1)
        points += trans_vector
    return points


def reprojection_error(points_3x1=None, observations_2x1=None, camera_matrix_3x3=None, rotation_vector_3x1=None, translation_vector_3x1=None):
    """ Project an 3D scene points to image plane and calculate the the residuals with respect to the measurement
    """
    if points_3x1 is None or observations_2x1 is None or camera_matrix_3x3 is None:
            return None
    projected_points = rotate_and_translate(points_3x1, rotation_vector_3x1, translation_vector_3x1)
    projected_points = np.dot(camera_matrix_3x3, projected_points.T).T
    # convert the projected points to the P2 homogenous coordinates
    projected_points = projected_points[:, :2] / projected_points[:, 2, np.newaxis]
    return projected_points - observations_2x1


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
    