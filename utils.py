
import glob, math
import numpy as np
import cv2
from PIL import Image
import os, io, libconf, copy
from scipy.sparse import lil_matrix
from numpy.linalg import inv, norm
from pyquaternion import Quaternion

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure, show


from cfg import *

''' Util functions '''

def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def split_and_write_image(image_file_path, out_image_path = '/tmp'):
    imgs_x4 = split_kite_vertical_images(image_file_path)
    img_name = image_file_path.split('/')[-1].split('.')[0]
    for i in range(4):
        out_image_path = out_image_path + '/'+ img_name + '_' + str(i) + '.jpg'
        cv2.imwrite(out_image_path, imgs_x4[i])

def kiteFishEyeUndistortPoints(points, K, D, R = np.eye(3)):
    dst = np.zeros_like(points)
    f = [K[0][0], K[1][1]]
    c = [K[0][2], K[1][2]]
    k = D[0:4]

    for i in range(len(points)):
        pi = np.array([points[i][0][0], points[i][0][1]])         # image point
        pw = np.array([(pi[0] - c[0])/f[0], (pi[1] - c[1]) / f[1]])  # world point
        scale = 1.0;
        theta_d = math.sqrt(pw[0]*pw[0] + pw[1]*pw[1])
        # // the current camera model is only valid up to 180 FOV
        # // for larger FOV the loop below does not converge
        # // clip values so we still get plausible results for super fisheye images > 180 grad
        theta_d = min(max(-np.pi/2., theta_d), np.pi/2.);
        EPS = 1e-8 # or std::numeric_limits<double>::epsilon();
        if theta_d > EPS:
            # compensate distortion iteratively
            theta = theta_d;
            for j in range(10):
                theta2 = theta*theta
                theta4 = theta2*theta2
                theta6 = theta4*theta2
                theta8 = theta6*theta2;
                k0_theta2 = k[0] * theta2
                k1_theta4 = k[1] * theta4
                k2_theta6 = k[2] * theta6
                k3_theta8 = k[3] * theta8
                # /* new_theta = theta - theta_fix, theta_fix = f0(theta) / f0'(theta) */
                part1 = (theta * (1 + k0_theta2 + k1_theta4 + k2_theta6 + k3_theta8) - theta_d) 
                part2 = (1 + 3*k0_theta2 + 5*k1_theta4 + 7*k2_theta6 + 9*k3_theta8)
                theta_fix = part1 / part2

                theta = theta - theta_fix;
                if abs(theta_fix) < EPS:
                    break;
            scale = math.tan(theta) / theta_d;
        
        pu = pw * scale #undistorted point
        #reproject
        pr = np.dot(R, np.array([pu[0], pu[1], 1.0]).reshape(3,1)) # rotated point optionally multiplied by new camera matrix
        # final
        dst[i] = np.array([pr[0]/pr[2], pr[1]/pr[2]]).reshape(1,2)
    return dst


def kiteEstimateNewCameraMatrixForUndistortRectify(K, D, image_shape = (640, 480), balance = 0.0):
    w = image_shape[0]
    h = image_shape[1]
    corners = []
    corners.append(np.array([w / 2.0, 0]))
    corners.append(np.array([w,  h / 2.0]))
    corners.append(np.array([w / 2.0,  h]))
    corners.append(np.array([0,  h / 2.0]))
    
    corners = np.array(corners, dtype=np.float32)
    corners = corners.reshape(len(corners), 1, 2)

    corners = cv2.fisheye.undistortPoints(corners, K, D, np.eye(3))
    # corners = kiteFishEyeUndistortPoints(corners, K, D, np.eye(3)) 

    center_mass = np.mean(corners)
    cn = np.array([center_mass, center_mass])

    aspect_ratio = K[0][0] / K [1][1]
    cn[0] *= aspect_ratio
    corners[:,0,1] *= aspect_ratio


    miny = np.min(corners[:,:,1])
    maxy = np.max(corners[:,:,1])
    minx = np.min(corners[:,:,0])
    maxx = np.max(corners[:,:,0])

    f1 = w * 0.5 / (cn[0] - minx)
    f2 = w * 0.5 / (maxx - cn[0])
    f3 = h * 0.5 * aspect_ratio/(cn[1] - miny)
    f4 = h * 0.5 * aspect_ratio/(maxy - cn[1])

    fmin = min(f1, min(f2, min(f3, f4)))
    fmax = max(f1, max(f2, max(f3, f4)))

    f = balance * fmin + (1.0 - balance) * fmax

    new_f = [f, f / aspect_ratio]
    new_c = -cn * f + np.array([w, h * aspect_ratio]) * 0.5
    # restore aspect ratio
    new_c[1] /= aspect_ratio

    new_K = np.array([new_f[0], 0.0, new_c[0], 
             0.0, new_f[1], new_c[1], 
             0.0, 0.0, 1.0]).reshape(3, 3)

    return new_K

def correct_kite_camera_matrix(K, D, dim = (640, 480), balance = 0.0):
    K0 = kiteEstimateNewCameraMatrixForUndistortRectify(K, D, image_shape = dim)
    # K1 = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, (640, 480), np.eye(3), balance=balance)
    # import pdb; pdb.set_trace()
    print(K0)
    return K0

def undistort_kite_image(img, K_org, K_new, D, dim = (640, 480), balance=0.0):
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K_org, D, np.eye(3), K_new, dim, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img


def kite_stereo_rectify(K1, D1, K2, D2, R, t, dim = (640, 480)):
    return cv.fisheye.stereoRectif(K1, D1, K2, D2, dim, R, t)


def gaussian_blur(img, kernel=(5,5)):
     return cv2.GaussianBlur(img, kernel, 0)

def region_of_interest_mask(image_shape, vertices, filler = None):
    """
    Applies an image mask.
    
    Only keeps the region of the image outside of the polygon
    formed from `vertices`. The inside of the polygon is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros(image_shape, dtype=np.uint8) 
    if not filler:  
        mask.fill(255)
    else:
        mask.fill(filler)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(image_shape) > 2:
        channel_count = image_shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (0,) * channel_count
    else:
        ignore_mask_color = 0
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return mask

def apply_mask_image(img, mask):
    """
    Applies an image mask.
    
    Only keeps the region of the image outside of the polygon
    formed from `vertices`. The inside of the polygon is set to black.
    """
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


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

def compare_descriptor(k0, k1, img0, img1, descriptor_threshold = 100):
    descriptor = cv2.ORB_create()
    # descriptor = cv2.xfeatures2d.BriefDescriptorExtractor_create(bytes=16)
    for i in range(k0.shape[0]):
        if k1[i][0][0] < 1.0 or k1[i][0][1] < 1.0:
            continue
        kp0 = cv2.KeyPoint(float(k0[i][0][0]), float(k0[i][0][1]), 9.0)
        kp0, des0 = descriptor.compute(img0, [kp0])

        kp1 = cv2.KeyPoint(float(k1[i][0][0]), float(k1[i][0][1]), 9.0)
        kp1, des1 = descriptor.compute(img1, [kp1])
        if des0 is None or des1 is None:
            k1[i][0][0] = -99.0
            k1[i][0][1] = -99.0
            continue
        # import pdb ; pdb.set_trace()
        des_distance = descriptor_hamming_distance(des0, des1)
        if des_distance > descriptor_threshold:
            k1[i][0][0] = -99.0
            k1[i][0][1] = -99.0

def descriptor_hamming_distance(des1, des2):
    dist = 0
    for d1, d2 in zip(des1[0].tolist(), des2[0].tolist()):
        res = bin(d1 ^ d2)
        # import pdb ; pdb.set_trace()
        cnt = 0
        for b in res[2:]:
            if b == '1':
                dist += 1
    return dist

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

def invert_RT(R, T):
    ''' Invert Rotation (3x3) and Translation (3x1)
    '''
    R2 = np.array(R).T
    T2 = -np.dot(R2, T)
    return R2, T2


def load_kite_config(cfg_file='nav_calib.cfg', num_cams=4):
    cam_matrix = [None] * num_cams
    dist = [None] * num_cams
    cam_rot = [None] * num_cams
    cam_trans = [None] * num_cams
    imu_rot = [None] * num_cams
    imu_trans = [None] * num_cams

    with io.open(cfg_file) as f:
        config = libconf.load(f)
        # import pdb ; pdb.set_trace()
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
            # IMU to each camera's rotation and translation
            imu_rot[cam_id]  = np.array(cam_calib['cam_imu_rot']).reshape(3,3)
            imu_trans[cam_id]  = np.array(cam_calib['cam_imu_trans']).reshape(3,1)
    return cam_matrix, dist, cam_rot, cam_trans, imu_rot, imu_trans


def load_kitti_config(cfg_file='calib.txt',  num_cams=4):
    lines = [line.rstrip('\n') for line in open(cfg_file)]
    cam_matrix = [None] * num_cams
    dist = [None] * num_cams
    cam_rot = [None] * num_cams
    cam_trans = [None] * num_cams
    imu_rot = [None] * num_cams
    imu_trans = [None] * num_cams
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
        # mtx = np.dot(scaling_mtx, mtx)
        cam_matrix[cam_id] = mtx
        # import pdb ; pdb.set_trace()
        dist[cam_id] = np.zeros(6)
        cam_rot[cam_id] = rot
        cam_trans[cam_id] = trans
        imu_rot[cam_id]  = np.eye(3)
        imu_trans[cam_id]  = np.zeros([3, 1])
    return cam_matrix, dist, cam_rot, cam_trans, imu_rot, imu_trans

def split_image_x4(image_path):
    img_files = glob.glob(image_path + '/*.jpg')
    img_files = sorted(img_files)

    for img_file in img_files:
        imgx4 = split_kite_vertical_images(img_file)
        img_name = img_file.split('/')[-1]
        for camera_index in range(4):
            out_path = os.path.join(image_path, 'cam' + str(camera_index))
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            out_path = os.path.join(out_path, img_name)
            cv2.imwrite(out_path, imgx4[camera_index])

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
    pose_path = os.path.join(kitti_base, 'poses')
    pose_path = os.path.join(pose_path, data_seq + '.txt')
    if not os.path.exists(pose_path):
        return None
    with open(pose_path) as f:
	    annotations = f.readlines()
    return annotations

def get_kite_ground_truth(file_path):
    acs_data = []
    with open(file_path) as f:
        reader = csv.reader(f)
        for row in reader:
            acs_data.append(np.array(row, dtype=np.float32))
    return np.array(acs_data)

def get_closest_acs_metadata(acsmeta, image_timestamp_ns, prev_acsmeta_index, max_range = 10000):

    img_acs_diff = np.absolute(acsmeta[:,0][prev_acsmeta_index: prev_acsmeta_index + max_range] - image_timestamp_ns)
    min_error_index = np.argmin(img_acs_diff)
    error = img_acs_diff[min_error_index]

    matched_acsmeta_index = prev_acsmeta_index + min_error_index

    prev_acsmeta_index = min_error_index + 1
    return matched_acsmeta_index, 

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
    imgs_x2 = []
    for c in range(num_cams):
        im = Image.open(camera_images[c][img_idx])
        resized_image = im
        imgs_x2.append(np.asarray(resized_image))
    return imgs_x2

def read_kite_image(camera_images, num_cams=None, video_format='2x2',img_idx=0):
    if video_format == '2x2':
        imgs_x4 = pil_split_rotate_kite_record_image(camera_images[img_idx])
    elif video_format == '4x1':
        imgs_x4 = split_kite_vertical_images(camera_images[img_idx])
    elif video_format == '1x1':
        imgs_x4 = kite_read_4x_images(camera_images[img_idx])
    else:
        assert(0)
    ts = int(camera_images[img_idx].split('/')[-1].split('.')[0])
    # import pdb; pdb.set_trace()
    return imgs_x4, ts

def get_kitti_image_files(kitti_base=None, data_seq='01', max_cam=2):
    seq_path =  os.path.join(kitti_base, 'sequences')
    seq_path =  os.path.join(seq_path, data_seq)
    camera_images = []
    for c in range(max_cam):
        images_base = os.path.join(seq_path, 'image_' + str(c))
        if not os.path.exists(images_base):
            continue
        img_files = glob.glob(images_base + '/*.png')
        img_files = sorted(img_files)
        resize_folder = os.path.join(seq_path, 'resized_image_' + str(c))
        resize_images(img_files, resize_folder, (1280, 480))
        camera_images.append(img_files)
    return camera_images

def resize_images(image_list,  output_path, target_size = (None, None)):
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

def get_kite_image_files(kite_base=None, video_format = '2x2', skip_images_factor = 0):
    if video_format == '1x1':
        x4_imgs = sync_navcam_collected_images(kite_base)
        return x4_imgs 
    img_files = sorted(glob.glob(kite_base + '/*.jpg'))
    
    if skip_images_factor > 0:
        img_files = img_files[::skip_images_factor]
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
    
def shi_tomasi_corner_detection(img, quality_level = 0.01, min_distance = 8, roi_mask = None, kpts_num=64):
    feature_params = dict( maxCorners = kpts_num,
                        qualityLevel = quality_level,
                        minDistance = min_distance,
                        blockSize = 7,
                        mask = roi_mask)
    return cv2.goodFeaturesToTrack(img, **feature_params)

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


def rectify_camera_pairs(cam0_img, cam1_img, K0, K1, D0, D1, R, T, img_size = (640, 480)):    
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(K0, D0, K1, D1, img_size, R, T)

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



def pil_split_rotate_kite_record_image(img_file):
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

def split_kite_vertical_images(img_file, num_cams=4):
    """Split recorded nav images to 4
    # 0 
    # 1 
    # 2
    # 3
    """
    im = Image.open(img_file)
    width, height = im.size
    im_width = width
    im_height = height // num_cams
    splited_images    = num_cams * [None]
    for i in range(num_cams):
        splited_images[i] = np.asarray(im.crop((0, im_height * i, im_width, im_height * (i + 1))))
        splited_images[i] =cv2.cvtColor(splited_images[i], cv2.COLOR_RGB2GRAY)
    return splited_images


def kite_read_4x_images(img_files, num_cams=4):
    assert(num_cams == len(img_files))
    images_4x = num_cams * [None]
    for i in range(len(img_files)):
        im = cv2.imread(img_files[i], cv2.IMREAD_GRAYSCALE)
        if im is None:
            return None
        images_4x[i] = im
    return images_4x

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
        imgs_x4 = pil_split_rotate_kite_record_image(file)
        for c in range(4):
            cam_imgs[c].append(imgs_x4[c])
    return cam_imgs


def sparse_optflow(curr_im, target_im, flow_kpt0, win_size  = (18, 18)):
    if flow_kpt0 is None or len(flow_kpt0) == 0:
        return None, None, None
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = win_size,
                    maxLevel = 4,
                    minEigThreshold=1e-4,
                    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 4, 0.01))
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
def global_bundle_adjustment_sparsity(cam_obs, n_cams=4, n_poses=1):
    n_obs_013 = cam_obs[:,0]
    n_obs_01  = cam_obs[:,1]

    n_obs_013_sum = np.sum(n_obs_013)
    n_obs_01_sum = np.sum(n_obs_01)

    n_obs = n_obs_013_sum + n_obs_01_sum

    m = (n_obs_013_sum * 3 + n_obs_01_sum * 2) * 2 # rows
    n = n_poses * 6 + n_obs * 3 # cols
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


def get_current_image_timestamp(img_path):
    ''' Get the timestamp from the image file name
    ''' 
    if not os.path.exists(img_path):
        return -1
    fname = os.path.basename(img_path)
    ts = os.path.splitext(fname)[0]
    if not ts.isdigit():
        return -1
    return long(ts)

def get_cam0_valid_images(base_path, start_index = -1, end_index = -1):
    ''' Get the valid cam0 image files
    ''' 
    cam0_path = os.path.join(base_path, 'cam0')
    img_files = sorted(glob.glob(cam0_path + '/*.ppm'))
    img_list = []
    for img in img_files:
        ts = get_current_image_timestamp(img)
        if ts == -1:
            print(img + ' do not have valid timestsamp!')
            continue
        if start_index > 0 and ts < start_index:
            continue
        if end_index > 0 and ts > end_index:
            continue
        img_list.append(img)
    return img_list

def sync_navcam_collected_images(base_path, start_index = -1, end_index = -1):
    cam0_list = get_cam0_valid_images(base_path, start_index)
    cam_123 = [] 
    for index in range(1, 4):
        cam_i_path = os.path.join(base_path, 'cam'+str(index))
        if not os.path.exists(cam_i_path):
            print(cam_i_path + ' does not exist')
            return None        
        current_images = sorted(glob.glob(cam_i_path + '/*.ppm'))
        # for each image find the frame closest to cam0 frames
        prev_cam0_ts = 0
        prev_cami_ts = 0
        synced_cami = []
        prev_index = 0
        for cam0 in cam0_list:
            # get cam0 timestamp
            cam0_ts = get_current_image_timestamp(cam0)
            assert(cam0_ts)
            synced_file = None
            for i in range(prev_index, len(current_images)):
                img_i = current_images[i]
                cur_ts = get_current_image_timestamp(img_i)
                if cur_ts == -1:
                    continue
                offset =  cur_ts - cam0_ts
                if offset > 80000:
                    break
                if abs(offset) < 80000:
                    synced_file = img_i
                    prev_index = i
                    break
            assert(synced_file)
            synced_cami.append(synced_file)
        cam_123.append(synced_cami)


    synced = []
    for i in range(len(cam0_list)):
        x4_frames = []
        x4_frames.append(cam0_list[i])
        for index in range(0, len(cam_123)):
            x4_frames.append(cam_123[index][i])
        synced.append(x4_frames)
    return synced


def compute_body_to_camera0_transformation(imu_to_body_rotation=np.eye(3), rotation_imu_to_cam0=np.eye(3), translation_imu_to_camera0=np.zeros([3, 1])):
    '''[Compute the rotation and translation from ACS (NED)frame to camera0 frame ]
    
    Keyword Arguments:
        acs_to_imu_rotation_angle {int} -- [description] (default: {0})
        imu_to_camera0_rotation {[type]} -- [description] (default: {np.eye(3)})
        imu_to_camera0_translation {[type]} -- [description] (default: {np.zeros([3, 1])})
    '''
    rotation_body_to_cam0 = np.dot(rotation_imu_to_cam0,  imu_to_body_rotation.T)
    translation_body_to_cam0 = translation_imu_to_camera0
    # import pdb; pdb.set_trace()
    return rotation_body_to_cam0, translation_body_to_cam0

def transform_egomotion_from_frame_a_to_b(egomotion_rotation_a, egomotion_translation_a, rotation_a_to_b, translation_a_to_b):
    '''Transform egomotion from frame a to b
    '''
    egomotion_rotation_b = rotation_a_to_b.dot(egomotion_rotation_a.dot(rotation_a_to_b.T))

    # Compute the translation
    egomotion_translation_b =  np.dot((np.eye(3) - egomotion_rotation_b), translation_a_to_b)
    egomotion_translation_b +=  np.dot(rotation_a_to_b, egomotion_translation_a)
    return egomotion_rotation_b, egomotion_translation_b
    

def angular_velocity_to_rotation_matrix(w = [0.0, 0.0, 0.0], dt = 0.0):
    wx, wy, wz = w[0], w[1], w[2]
    # construct a unit quaternion from an identity matrix
    # q = q + 0.5 * w * q * t
    # 0.5 * w * q converting a body angular velocity to quaternion velocity
    n0 = 1.0; n1 = 0.0; n2 = 0.0; n3 = 0.0;
    n0 = n0 - dt*((wx*n1)/2 + (wy*n2)/2 + (wz*n3)/2)
    n1 = n1 + dt*((wx*n0)/2 - (wy*n3)/2 + (wz*n2)/2)
    n2 = n2 + dt*((wx*n3)/2 + (wy*n0)/2 - (wz*n1)/2)
    n3 = n3 + dt*((wy*n1)/2 - (wx*n2)/2 + (wz*n0)/2)

    scale = 1.0 / math.sqrt(n0 * n0 + n1 * n1 + n2 * n2 + n3 * n3 )

    q0 = Quaternion(array=np.array([n0, n1, n2, n3]))

    return q0.rotation_matrix


def linear_velocity_to_translation(v = [0.0, 0.0, 0.0], dt = 0.0):
    return np.array([v[0]*dt, v[1]*dt, v[2]*dt]).reshape(3, 1)

# ref
# http://danceswithcode.net/engineeringnotes/rotations_in_3d/rotations_in_3d_part1.html
def eular_angle_to_rotation_matrix(eular):
    u = eular[0]; v = eular[1]; w = eular[2];
    cu = math.cos(u); su = math.sin(u);
    cv = math.cos(v); sv = math.sin(v);
    cw = math.cos(w); sw = math.sin(w);


    m00 = cu * cw
    m01 = su * sv * cw - cu * sw
    m02 = su * sw + cu * sv * cw;

    m10 = cv * sw
    m11 = cu * cw + su * sv * sw
    m12 = cu * sv * sw - su * cw 

    m20 = -sv
    m21 = su * cv 
    m22 = cu * cv 
    return np.array([m00, m01, m02, m10, m11, m12, m20, m21, m22]).reshape(3,3)


def eulerAnglesToRotationMatrix(theta) :
     
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])

    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])
                 
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
                     
                     
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def get_translation_from_acsmeta(msg):
    return np.array([msg[ACS_POSITION_X], msg[ACS_POSITION_Y], msg[ACS_POSITION_Z]], dtype=np.float32).reshape(3,1)

def get_eular_angle_from_acsmeta(msg):
    return np.array([msg[ACS_ORIENTATION_PHI], msg[ACS_ORIENTATION_THETA], msg[ACS_ORIENTATION_PSI]], dtype=np.float32)


def get_position_orientation_from_acsmeta(msg):
    translation = get_translation_from_acsmeta(msg)
    rotation_eular = get_eular_angle_from_acsmeta(msg)
    # rotation_3x3 = pr.matrix_from_euler_zyx(rotation_eular)
    rotation_3x3 = eulerAnglesToRotationMatrix(rotation_eular)
    # import pdb; pdb.set_trace()
    return [rotation_3x3, translation]

def get_angular_linear_velocity_from_acsmeta(msg):
    w = np.array([msg[ACS_ORIENTATION_PHI_DOT], msg[ACS_ORIENTATION_THETA_DOT], msg[ACS_ORIENTATION_PSI_DOT]], dtype=np.float32).reshape(3,1)
    v = np.array([msg[ACS_POSITION_X_DOT], msg[ACS_POSITION_Y_DOT], msg[ACS_POSITION_Z_DOT]], dtype=np.float32).reshape(3,1)
    return [w, v]
    
# Taken from python curve fit covariance estimation code
def covarinace_svd(jac):
    _, s, VT = np.linalg.svd(jac, full_matrices=False)
    threshold = np.finfo(float).eps * max(res.jac.shape) * s[0]
    s = s[s > threshold]
    VT = VT[:s.size]
    pcov = np.dot(VT.T / s**2, VT)

def covariance_mvg_A6_4(jac): 
    # Estimate the covariance of the motion paramters by MVG Algorithm A6.4.
    hessian = jac.T.dot(jac)
    # with shape (6 + num_points * 3) x (6 + num_points)
    U = hessian[0:6,0:6] # Top left U block 6x6 
    W = hessian[0:6,6:]  # Top right W block 6 x (num_points * 3)
    V = hessian[6:,6:]   # Bottom right block diagonal matrix (num_points * 3) x (num_points * 3)
    # Compute the 6x6 S matrix
    # S = U - sum_i(W_i * Vi ^ -1 * W_i')
    n_points = (hessian.shape[0] - 6) / 3

    sum_i = np.zeros_like(U)
    for i in range(n_points):
        W_i = W[:,i * 3 : (i + 1) * 3]
        V_i = V[i * 3 : i * 3 + 3,i * 3 : i * 3 + 3]
        V_i_inv = np.linalg.inv(V_i)
        sum_i += (W_i.dot(V_i_inv)).dot(W_i.T)
    S = U - sum_i
    S_inv = np.linalg.inv(S)
    return S_inv

def visualize_sparsity_jacobian(jac):
    fig = figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.set_title("Jacobian 4 Camera 1 pose")
    ax2.set_title("J' * J 4 Camera 1 pose")
    ax1.spy(jac, markersize=5)
    ax2.spy(jac.T.dot(jac), markersize=5)
    show()

