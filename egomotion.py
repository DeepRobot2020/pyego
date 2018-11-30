
import numpy as np
import glob, pdb, math, json
import warnings
import os, io, libconf, copy
import cv2
import csv

from PIL import Image

from numpy.linalg import inv, norm
from scipy.optimize import least_squares

import time
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
from ImageImuSyncer import ImageImuSyncer
from apc import ImageAPCSyncer

from utils import *
from cfg import *
import shutil
from datetime import datetime

from matplotlib.pyplot import figure, show

np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(
    description='Compute camera relative pose on input stereo or quad images')

parser.add_argument(
    '-dataset',
    '--dataset_type',
    help='dataset type: (Kite, Kitti)',
    default=DATASET)

parser.add_argument(
    '-seq',
    '--seq_num',
    type=int,
    help='sequence number (only fir Kitti)',
    default=2)

parser.add_argument(
    '-img',
    '--images_path',
    help='path to directory of input images',
    default=INPUT_IMAGE_PATH)

parser.add_argument(
    '-calib',
    '--calib_path',
    help='path to directory of calibriation file (Kite)',
    default=INPUT_CALIB_PATH)

parser.add_argument(
    '-out',
    '--output_path',
    help='path to output test images',
    default='output')

parser.add_argument(
    '-json',
    '--enable_json',
    type=bool,
    help='output data to json format',
    default=False)

parser.add_argument(
    '-feats',
    '--num_features',
    type=int,
    help='Max number of features',
    default=MAX_NUM_KEYPOINTS)

parser.add_argument(
    '-num_cam',
    '--num_cameras',
    type=int,
    help='Max number of cameras used for VO',
    default=2)

parser.add_argument(
    '-json_pose',
    '--external_json_pose',
    help='Debug a extimated pose',
    default=KITE_OUTPUT_POSE_PATH)

parser.add_argument(
    '-json_feature_path',
    '--json_feature_path',
    help='use image features from json files',
    type=str,
    default=None)

class KTPinholeCamera:
    def __init__(self, width, height, fx, fy, cx, cy, k1=0.0, k2=0.0, p1=0.0, p2=0.0, k3=0.0):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.distortion = (abs(k1) > 0.0000001)
        self.d = [k1, k2, p1, p2, k3]

kt_cam = KTPinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)


class navcam:
    def __init__(self, 
        index, stereo_pair_idx, 
        intrinsic_mtx,  intrinsic_dist,  
        extrinsic_rot, extrinsic_trans, 
        num_features=64,
        stereo_rectify_P = None,
        stereo_rectify_R = None):

        self.calib_d   = intrinsic_dist[0:4].astype(np.float32)
        self.calib_K0  = intrinsic_mtx.astype(np.float32)
        self.calib_R   = extrinsic_rot.astype(np.float32)
        self.calib_t   = extrinsic_trans.astype(np.float32)

        # stereo_rectify_P: Stereo rectified projection matrix 
        # stereo_rectify_R: Stereo rectified rotation matrix 
        self.stereo_rectify_P = stereo_rectify_P
        self.stereo_rectify_R = stereo_rectify_R
    
        # self.stereo_rectify_P = None

        if DATASET == 'kitti': 
            # focal lenght and princial point of kitti stereo cameras
            self.focal      = kt_cam.fx
            self.pp         = (kt_cam.cx, kt_cam.cy) 
            self.calib_K    = intrinsic_mtx

        else: #DATASET == 'kite':  # focal lenght and princial point of kite navcam
            if self.stereo_rectify_P is None:
                self.calib_K   = correctCameraMatrix(intrinsic_mtx, self.calib_d)
                self.focal     = (self.calib_K[0,0] + self.calib_K[1,1]) / 2.0
                self.pp         = (self.calib_K[:2,2][0], self.calib_K[:2,2][1])  
            else:
                self.calib_K = self.stereo_rectify_P[index][0:3, 0:3]
                self.focal = (self.calib_K[0,0] + self.calib_K[1,1]) / 2.0
                self.pp = (self.calib_K[:2,2][0], self.calib_K[:2,2][1])  
                self.calib_R = np.identity(3)
                if index % 2 == 0:
                    tvec = -self.stereo_rectify_P[stereo_pair_idx][:,3]
                else:
                    tvec = self.stereo_rectify_P[index][:,3]

                self.calib_t = (inv(self.calib_K).dot(tvec)).reshape(3,1)

        self.calib_R2  = cv2.Rodrigues(self.calib_R)[0].astype(np.float32)

        
        self.num_features = num_features
        self.flow_kpt0 = None
        self.flow_kpt1 = None
        self.flow_kpt2 = None
        self.flow_kpt3 = None
        self.flow_kpt4 = None
        self.stereo_pair_cam = None
        self.stereo_pair_idx = stereo_pair_idx 

        self.stereo_R = None # camera pose rotation in 
        self.stereo_t = None 
        self.mono_cur_R = None
        self.mono_cur_t = None
        self.ego_R = None
        self.ego_t = None
        self.prev_scale = 1.0
        # keypoints which have both intra and inter matching
        self.flow_intra_inter0 = None  # original keypoints
        self.flow_intra_inter1 = None  # intra match
        self.flow_intra_inter3 = None  # inter match

        self.flow_intra0  = None # original keypoints
        self.flow_intra1  = None # intra match

        self.flow_inter0 = None # original keypoints
        self.flow_inter3 = None # inter match

        self.intra0 = None # original keypoints
        self.intra1 = None # inter match

        self.curr_img  = None
        self.curr_stereo_img  = None
        self.prev_img  = None
        self.index   = index

        self.img_idx = None
        self.F = None 
    
        self.proj_mtx = None


    def mono_vo(self, abs_scale = 1.0):
        if self.img_idx is None or self.img_idx == 0:
            return None, None
        if self.intra0 is None:
            return self.mono_cur_R, self.mono_cur_t
        if self.intra0.shape[0] < 5:
            return self.mono_cur_R, self.mono_cur_t
        try:
            K0 = self.calib_K
            E, e_mask = cv2.findEssentialMat(self.intra0, self.intra1, focal=self.focal, pp=self.pp, method=cv2.LMEDS, prob=0.99, threshold=1e-2)
            nin, R, t, mask = cv2.recoverPose(E, self.intra0, self.intra1, mask = e_mask, focal=self.focal, pp = self.pp)
        except:
            return None, None

        if self.img_idx == 1:
            self.mono_cur_R = R
            self.mono_cur_t = t
            return R, t

        if abs_scale > 0.1 and abs_scale != 1.0:
            self.mono_cur_t = self.mono_cur_t + abs_scale * self.mono_cur_R.dot(t) 
            self.mono_cur_R = R.dot(self.mono_cur_R)
        return R, t

    def set_stereo_pair(self, right_cam):
        '''Set the stereo pair of current camera  '''
        self.stereo_pair_cam = right_cam

    def fun(self, x0, cam_obs, y_meas):
        n_kpts_013, n_kpts_01 = cam_obs
                
        rot_vecs   = x0[0:3]
        trans_vecs = x0[3:6]
        points_013 = x0[6: 6 + 3 * n_kpts_013].reshape(-1, 3)

        flow013_0  = y_meas[0: 1 * n_kpts_013]
        flow013_1  = y_meas[1 * n_kpts_013 : 2 * n_kpts_013]
        flow013_3  = y_meas[2 * n_kpts_013 : 3 * n_kpts_013]

        flow0_err = reprojection_error(points_013, flow013_0, self.calib_K)
        flow1_err = reprojection_error(points_013, flow013_1, self.calib_K, rot_vecs, trans_vecs)
        flow3_err = reprojection_error(points_013, flow013_3, self.stereo_pair_cam.calib_K, self.calib_R2, self.calib_t)
    
        errs = flow1_err
        errs013_03 = np.vstack((flow0_err, flow3_err))
        
        flow01_err0 = None
        flow01_err1 = None

        if n_kpts_01 > 0:
            points_01  = x0[6 + 3 * n_kpts_013 : 6 + 3 * n_kpts_013 + 3 * n_kpts_01].reshape(-1, 3)
            flow01_0  = y_meas[3 * n_kpts_013  : 3 * n_kpts_013 + n_kpts_01]
            flow01_1  = y_meas[3 * n_kpts_013 + n_kpts_01: 3 * n_kpts_013 + 2*n_kpts_01]

            flow0_err = reprojection_error(points_01, flow01_0, self.calib_K)
            flow1_err = reprojection_error(points_01, flow01_1, self.calib_K, rot_vecs, trans_vecs)

            errs = np.vstack((errs, flow1_err))   
            errs013_03 = np.vstack((errs013_03, flow0_err))

        errs = np.vstack((errs, errs013_03))
        return errs.ravel()

    def reprojection_err(self, x0, cam_obs, y_meas):
        n_kpts_013, n_kpts_01 = cam_obs
        reprojection_errs = []     
        rot_vecs   = x0[0:3]
        trans_vecs = x0[3:6]
        points_013 = x0[6: 6 + 3 * n_kpts_013].reshape(-1, 3)
        flow013_0  = y_meas[0 : n_kpts_013]
        flow013_1  = y_meas[n_kpts_013 : 2 * n_kpts_013]
        flow013_3  = y_meas[2 * n_kpts_013 : 3 * n_kpts_013]

        flow0_err = reprojection_error(points_013, flow013_0, self.calib_K)
        flow1_err = reprojection_error(points_013, flow013_1, self.calib_K, rot_vecs, trans_vecs)
        flow3_err = reprojection_error(points_013, flow013_3, self.stereo_pair_cam.calib_K, self.calib_R2, self.calib_t)
    
        reprojection_errs.append(flow0_err)
        reprojection_errs.append(flow1_err)
        reprojection_errs.append(flow3_err)

        if n_kpts_01 > 0:
            points_01  = x0[6 + 3 * n_kpts_013 : 6 + 3 * n_kpts_013 + 3 * n_kpts_01].reshape(-1, 3)
            flow01_0  = y_meas[3 * n_kpts_013  : 3 * n_kpts_013 + n_kpts_01]
            flow01_1  = y_meas[3 * n_kpts_013 + n_kpts_01: 3 * n_kpts_013 + 2*n_kpts_01]
            flow0_err = reprojection_error(points_01, flow01_0, self.calib_K)
            flow1_err = reprojection_error(points_01, flow01_1, self.calib_K, rot_vecs, trans_vecs)
            reprojection_errs.append(flow0_err)
            reprojection_errs.append(flow1_err)
        return reprojection_errs

    def mono_ego_motion_estimation(self, abs_scale=None):
        mono_rotation_matrix, mono_translation_vector = self.mono_vo(1.0)
        if mono_rotation_matrix is None:
            return None, None
        if abs_scale:
            mono_translation_vector = abs_scale * mono_translation_vector
        mono_rotation_vector = cv2.Rodrigues(mono_rotation_matrix)[0]
        return mono_rotation_vector, mono_translation_vector

    def update_camera_pose_egomotion(self, R, t):
        self.ego_R = R
        self.ego_t = t
        # 
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
    
    def generate_initial_guess(self, init_with_mono=True):
        # initial guess
        est_R = np.random.normal(0, 0.01, [3,1])
        est_t = np.random.normal(0, 0.01, [3,1])
        if init_with_mono:
            mono_R, mono_t = self.mono_ego_motion_estimation()
            if mono_R is not None:
                est_R = mono_R
                est_t = self.prev_scale * mono_t if self.prev_scale is not None else mono_t
        return est_R, est_t

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
        roi_mask = None
        if DATASET == 'kite':
            roi_mask = region_of_interest_mask(self.curr_img.shape, KITE_MASK_VERTICES[self.index])

        self.flow_kpt0 = shi_tomasi_corner_detection(self.curr_img, 
                                                    quality_level = SHI_TOMASI_QUALITY_LEVEL,
                                                    min_distance = SHI_TOMASI_MIN_DISTANCE, 
                                                    roi_mask = roi_mask, 
                                                    kpts_num = self.num_features)
    def front_end_sift_detection_matching(self):
        if self.curr_img is None:
            print('Warning: curr_img is None')
            return
        roi_mask0 = None
        roi_mask1 = None
        if DATASET == 'kite':
            roi_mask0 = region_of_interest_mask(self.curr_img.shape, 
                                        KITE_MASK_VERTICES[self.index], 
                                        filler = 1)

            roi_mask1 = region_of_interest_mask(self.curr_img.shape, 
                                KITE_MASK_VERTICES[self.stereo_pair_cam.index], 
                                filler = 1)

        img0 = self.curr_img
        img1 = self.prev_img
        img3 = self.curr_stereo_img

        sift = cv2.xfeatures2d.SIFT_create()
        kp0, des0 = sift.detectAndCompute(img0, roi_mask0)
        kp1, des1 = sift.detectAndCompute(img1, roi_mask0)
        kp3, des3 = sift.detectAndCompute(img3, roi_mask1)

        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2,k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
        else:
            print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
            matchesMask = None
                    

    def intra_sparse_optflow(self):
        if self.prev_img is not None:
            k0, k1, k2 = sparse_optflow(self.curr_img, self.prev_img, self.flow_kpt0, win_size=INTRA_OPTFLOW_WIN_SIZE)
            self.flow_kpt1 = k1
            self.flow_kpt2 = k2
            compare_descriptor(k0, k1, self.curr_img, self.prev_img, descriptor_threshold=INTRA_OPT_FLOW_DESCRIPTOR_THRESHOLD)
            self.filter_intra_keypoints()

    def inter_sparse_optflow(self):
        k0, k3, k4 = sparse_optflow(self.curr_img, self.curr_stereo_img, self.flow_kpt0, win_size=INTER_OPTFLOW_WIN_SIZE)
        self.flow_kpt3 = k3
        self.flow_kpt4 = k4
        compare_descriptor(k0, k3, self.curr_img, self.curr_stereo_img, descriptor_threshold=INTER_OPT_FLOW_DESCRIPTOR_THRESHOLD)
        self.filter_inter_keypoints()

    def filter_intra_keypoints(self):
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
                if xe > INTRA_OPT_FLOW_FW_BW_ERROR_THRESHOLD or ye > INTRA_OPT_FLOW_FW_BW_ERROR_THRESHOLD:
                    self.flow_kpt1[ct][0][0] = -20.0
                    self.flow_kpt1[ct][0][1] = -20.0
                    continue  
        # import pdb ; pdb.set_trace()
    def filter_inter_keypoints(self):
        img = None
        if self.prev_img is not None:
            if self.F is None:
                K0 = self.calib_K
                K1 = self.stereo_pair_cam.calib_K
                rot = self.calib_R
                trans = self.calib_t
                self.F = fundamental_matrix(rot, trans, K0, K1)
            # import pdb; pdb.set_trace()
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
                if xe > INTER_OPT_FLOW_FW_BW_ERROR_THRESHOLD or ye > INTER_OPT_FLOW_FW_BW_ERROR_THRESHOLD:
                    self.flow_kpt3[ct][0][0] = -20.0
                    self.flow_kpt3[ct][0][1] = -20.0
                    continue  
                if abs(err) > INTER_OPT_FLOW_EPILINE_ERROR_THRESHOLD:
                    self.flow_kpt3[ct][0][0] = -30.0
                    self.flow_kpt3[ct][0][1] = -30.0
                    continue
            # import pdb; pdb.set_trace()
    def debug_inter_keypoints(self, out_dir='/tmp'):
        img = None
        if self.prev_img is not None:
            flow0 = self.flow_intra_inter0
            flow3 = self.flow_intra_inter3    

            if flow0 is None or flow3 is None:
                return 
            if len(flow0) == 0 or len(flow3) == 0:
                return 
            try:
                points013, _, _ = triangulate_3d_points(flow0, flow3, self.calib_K, self.stereo_pair_cam.calib_K, self.calib_R, self.calib_t)
            except:
                import pdb; pdb.set_trace()
            img1 = cv2.cvtColor(self.curr_img, cv2.COLOR_GRAY2BGR)
            img2 = cv2.cvtColor(self.curr_stereo_img, cv2.COLOR_GRAY2BGR)
            for pt1, pt3, wp in zip(self.flow_intra_inter0, self.flow_intra_inter3, points013):
                x1, y1 = (int(pt1[0]), int(pt1[1]))
                x3, y3 = (int(pt3[0]), int(pt3[1]))
                color = tuple(np.random.randint(0,255,3).tolist())
                depth = "{:.1f}".format(wp[2])
                cv2.putText(img1, depth, (x1 + 10, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, lineType=cv2.LINE_AA) 
                cv2.putText(img2, depth, (x3 + 10, y3 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, lineType=cv2.LINE_AA) 
                cv2.circle(img1,(x1, y1), 6, color,2)
                cv2.circle(img2,(x3, y3), 6, color,2)

            img = concat_images(img1, img2)
            if img is not None:
                out_img_name = os.path.join(out_dir, 'cam_' + str(self.index) + '_inter_' + str(self.img_idx)+'.jpg')
                cv2.imwrite(out_img_name, img)

    def debug_intra_keypoints(self, out_dir='/tmp'):
        img = None
        if self.prev_img is not None:
            img1 = cv2.cvtColor(self.curr_img, cv2.COLOR_GRAY2BGR)
            img2 = cv2.cvtColor(self.prev_img, cv2.COLOR_GRAY2BGR)
            for pt1, pt2 in zip(self.flow_intra_inter0, self.flow_intra_inter1):
                x1, y1 = (int(pt1[0]), int(pt1[1]))
                x3, y3 = (int(pt2[0]), int(pt2[1]))
                color = tuple(np.random.randint(0,255,3).tolist())
                cv2.circle(img1,(x1, y1), 6, color,2)
                cv2.circle(img2,(x3, y3), 6, color,2)

            for pt1, pt2 in zip(self.flow_intra0, self.flow_intra1):
                x1, y1 = (int(pt1[0]), int(pt1[1]))
                x3, y3 = (int(pt2[0]), int(pt2[1]))
                color = tuple(np.random.randint(0,255,3).tolist())
                cv2.circle(img1,(x1, y1), 6, color,2)
                cv2.circle(img2,(x3, y3), 6, color,2)

            img = concat_images(img1, img2)
            if img is not None:
                out_img_name = os.path.join(out_dir, 'cam_' + str(self.index) + '_intra_' + str(self.img_idx)+'.jpg')
                cv2.imwrite(out_img_name, img)


    def filter_keypoints(self, debug=False, out_dir='/home/jzhang/Pictures/tmp/', max_inter_pts = 100, max_depth = 50):
        if self.prev_img is not None:            
            flow_intra_inter0 = []
            flow_intra_inter1 = []
            flow_intra_inter3 = []

            flow_intra0 = []
            flow_intra1 = []

            flow_inter0 = []
            flow_inter3 = []
            intra0 = []
            intra1 = []
            self.flow_intra_inter0 = None
            self.flow_intra_inter1 = None
            self.flow_intra_inter3 = None

            self.flow_intra0 = None
            self.flow_intra1 = None

            self.flow_inter0 = None
            self.flow_inter3 = None

            self.intra0 = None
            self.intra1 = None
            
            K0 = self.calib_K
            K1 = self.stereo_pair_cam.calib_K
            rot_01 = self.calib_R
            trans_01 = self.calib_t
            
            points013, terr0, terr1 = triangulate_3d_points(np.array(self.flow_kpt0).reshape(-1,2), 
                                                            np.array(self.flow_kpt3).reshape(-1,2), 
                                                            K0, 
                                                            K1, 
                                                            rot_01, 
                                                            trans_01)

            dist01 = np.sum(np.abs(self.flow_kpt0 - self.flow_kpt1)**2,axis=-1)**(1./2)
            dist03 = np.sum(np.abs(self.flow_kpt0 - self.flow_kpt3)**2,axis=-1)**(1./2)


            num_flow_013 = 0
            for kp0, kp1, kp3, d01, d03 in zip(self.flow_kpt0, self.flow_kpt1, self.flow_kpt3, dist01, dist03):
                x0, y0 = kp0[0][0], kp0[0][1]
                x1, y1 = kp1[0][0], kp1[0][1]
                x3, y3 = kp3[0][0], kp3[0][1]
                # import pdb; pdb.set_trace()
                if d03 > 1.0 and d01 > 1.0 and x1 > 1.0 and x3 > 1.0 and num_flow_013 < max_inter_pts: # intra and inter
                    wp, _, _ = triangulate_3d_points(np.array([x0, y0]).reshape(-1,2), 
                                                     np.array([x3, y3]).reshape(-1,2), 
                                                     K0, 
                                                     K1, 
                                                     rot_01, 
                                                     trans_01)
                    
                    # import pdb ; pdb.set_trace()
                    world_pt_depth = wp.reshape(-1,)[2]

                    if world_pt_depth <= 0.1 or world_pt_depth > max_depth:
                        # print('noisy point', x0,y0, x3,y3, world_pt_depth)
                        continue
                    num_flow_013 += 1
                    flow_intra_inter0.append(np.array([x0, y0]))
                    flow_intra_inter1.append(np.array([x1, y1]))
                    flow_intra_inter3.append(np.array([x3, y3]))
                    intra0.append(np.array([x0, y0]))
                    intra1.append(np.array([x1, y1]))
                    # n_res += 1
                elif d03 > 1.0 and d01 > 1.0 and x1 > 1.0 and x3 < 1.0: # intra only
                    flow_intra0.append(np.array([x0, y0]))
                    flow_intra1.append(np.array([x1, y1]))
                    intra0.append(np.array([x0, y0]))
                    intra1.append(np.array([x1, y1]))
                elif d03 > 1.0 and d01 > 1.0 and x1 < 0.0 and x3 > 0.0: # inter only
                    flow_inter0.append(np.array([x0, y0]))
                    flow_inter3.append(np.array([x3, y3]))


            self.flow_intra_inter0 = np.array(flow_intra_inter0, dtype=np.float64)
            self.flow_intra_inter1 = np.array(flow_intra_inter1, dtype=np.float64)
            self.flow_intra_inter3 = np.array(flow_intra_inter3, dtype=np.float64)

            self.flow_intra0 = np.array(flow_intra0, dtype=np.float64)
            self.flow_intra1 = np.array(flow_intra1, dtype=np.float64)

            self.flow_inter0 = np.array(flow_inter0, dtype=np.float64)
            self.flow_inter3 = np.array(flow_inter3, dtype=np.float64)

            self.intra0 = np.array(intra0, dtype=np.float64)
            self.intra1 = np.array(intra1, dtype=np.float64)

            # import pdb ; pdb.set_trace()
            if debug:    
                self.debug_inter_keypoints(out_dir)
                self.debug_intra_keypoints(out_dir)
                print('img', self.img_idx, 'cam_'+ str(self.index ), 'intra_inter:' + str(len(self.flow_intra_inter0)), 'intra:' + str(len(self.flow_intra0)), 'inter:'+str(len(self.flow_inter0)))


    def filter_keypoints_extra(self, rot, trans, debug=False, out_dir='/home/jzhang/Pictures/tmp/', max_depth = 50):
        ''' Further filter out the bad keypoints by checking with initial motions
        '''
        if self.prev_img is not None:            
            flow_intra_inter0 = self.flow_intra_inter0
            flow_intra_inter3 = self.flow_intra_inter3

            flow_intra0_new = []
            flow_intra1_new = [] 

            K0 = self.calib_K
            K1 = self.stereo_pair_cam.calib_K
            rot_01  = self.calib_R
            trans_01 = self.calib_t

            try:
                points013, terr0, terr1 = triangulate_3d_points(np.array(flow_intra_inter0).reshape(-1,2), 
                                                                np.array(flow_intra_inter3).reshape(-1,2), 
                                                                K0, 
                                                                K1, 
                                                                rot_01, 
                                                                trans_01)
                avg_depth = np.mean(points013[:,2])
                min_depth = np.min(points013[:,2]) / 4
                max_depth = np.max(points013[:,2]) + 2 * avg_depth
                
            except:
                min_depth = 3



            try:
                points01, terr01_0, terr01_1 = triangulate_3d_points(self.flow_intra0, self.flow_intra1, K0, K0, rot, trans)
            except:
                return 0
            n_bad01 = 0

            for idx, pt01_depth in enumerate(points01[:,2]):
                if pt01_depth > min_depth and pt01_depth < max_depth:
                    flow_intra0_new.append(self.flow_intra0[idx])
                    flow_intra1_new.append(self.flow_intra1[idx])
                elif debug:
                    print('noisy 01 point', 
                        self.flow_intra0[idx][0],self.flow_intra0[idx][1], 
                        self.flow_intra1[idx][0],self.flow_intra1[idx][1], 
                        pt01_depth)
                    n_bad01 += 1
            self.flow_intra0 = np.array(flow_intra0_new, dtype=np.float64)
            self.flow_intra1 = np.array(flow_intra1_new, dtype=np.float64)
            return n_bad01


class EgoMotion:
    """Kite vision object"""
    def __init__(self, calib_file=None, num_cams=4, num_features=64, dataset=DATASET, input_path=None, data_seq=None, json_output=True, ransac=False):
        self.num_features = num_features
        self.navcams      = []
        self.STEREOCONFG    = [1, 0, 3, 2]
        self.num_imgs = 0
        self.focal = None

        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.max_cams = 4
        # Kitti: images of each camera are stored s
        # Kite: images have been concatenated into one image
        self.camera_images = None
        self.prev_scale = 1.0
        self.img_idx = -1
        self.dataset = dataset.lower()
        self.dataset_seq = data_seq
        self.annotations = get_kitti_ground_truth(input_path, data_seq)

        self.rotation_cam0_to_cam2 = np.identity(3)
        self.translation_cam0_to_cam2 = np.zeros((3,1))

        self.rotation_cam2_to_cam0 = np.identity(3)
        self.translation_cam2_to_cam0 = np.zeros((3,1))

        self.prev_egomotion = np.zeros((1,6))
        self.prev_invalid = True
        self.syncer = None
        self.json_log = None
        self.csv_writer = None

        self.prev_ts = 0
        self.prev_acsmeta = None
        self.curr_acsmeta = None
        self.ned_vel_err = np.zeros([3,1])
        self.vel_covar = np.zeros([6,1])

        self.prev_rot = np.identity(3)
        self.prev_trans = np.zeros((3,1))

        self.csv_log_file = None
        self.json_log_file = None

        if self.dataset.lower() == 'kitti':
            self.camera_images = get_kitti_image_files(input_path, data_seq, num_cams)
            self.camera_images.sort(key=lambda f: int(filter(str.isdigit, f)))
            # import pdb; pdb.set_trace()
            self.num_imgs  = len(self.camera_images[0])
            self.num_cams = len(self.camera_images)
        elif self.dataset.lower() == 'kite':
            self.camera_images = get_kite_image_files(input_path, KITE_VIDEO_FORMAT, KITE_SKIP_IMAGE_FACTOR)
            self.camera_images.sort(key=lambda f: int(filter(str.isdigit, f)))
            self.num_imgs  = len(self.camera_images)
            self.num_cams = 4
            if ACS_META:
                self.syncer = ImageAPCSyncer(ACS_META)
                log_file_name = os.path.join(input_path, 'log.csv')
                if os.path.exists(log_file_name):
                    old_log_file_name = log_file_name + str(datetime.now().time())
                    print('backing up the old log ', log_file_name)
                    shutil.move(log_file_name, old_log_file_name)
                self.csv_log_file = open(log_file_name, 'w')
                self.csv_writer = csv.writer(self.csv_log_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL) 
                self.csv_writer.writerow(self.syncer.get_json_keys())

                log_file_name = os.path.join(input_path, 'log.json')
                if os.path.exists(log_file_name):
                    old_log_file_name = log_file_name + str(datetime.now().time())
                    print('backing up the old log ', log_file_name)
                    shutil.move(log_file_name, old_log_file_name)
                self.json_log_file = open(log_file_name, 'w')

        else:
            raise ValueError('Unsupported dataset')

        mtx, dist, rot, trans, imu_rot, imu_trans = load_camera_calib(dataset, calib_file, num_cams)

        if self.dataset.lower() == 'kite':
            self.rotation_body_to_cam0, self.translation_body_to_cam0 = compute_body_to_camera0_transformation(IMU_TO_BODY_ROT, imu_rot[0], imu_trans[0])
        else:
            self.rotation_body_to_cam0 = np.eye(3)
            self.translation_body_to_cam0 = np.zeros((3,1))

        self.rotation_cam0_to_body, self.translation_cam0_to_body = invert_RT(self.rotation_body_to_cam0, self.translation_body_to_cam0)

        # import pdb; pdb.set_trace()
        # From nav_calib.cfg
        # Camera trans/rot prev camera to camera. Camera 0 is the first in the chain.
        # IMU trans/rot is imu to camera.
        # Convert the chained calibration cam0 -> cam1 -> cam2 -> cam3
        # to two stereo pair calibration: cam0 -> cam1, cam2 -> cam3

        rot_01, trans_01 = rot[1], trans[1]
        rot_10, trans_10 = invert_RT(rot[1], trans[1])
    
        # For Kite there are two stereo pair (0, 1) and (2, 3), 
        # where camera 0 and camera 2 are their reference cameras, we need know the 
        # translation between those two stereo pairs
        rot_21 = np.identity(3)
        trans_21 = np.zeros((3,1))

        rot_12 = np.identity(3)
        trans_12 = np.zeros((3,1))
        
        rot_23, trans_23 = rot[3], trans[3]
        rot_32, trans_32 = invert_RT(rot[3], trans[3])
    
        if DATASET == 'kite':
            rot_12, trans_12 = rot[2], trans[2]
            
            rot_02 = np.dot(rot_12, rot_01) 
            trans_02 = np.dot(rot_12, trans_01) + trans_12

            rot_20, trans_20 = invert_RT(rot_02,trans_02 ) 
            self.rotation_cam0_to_cam2 = rot_02
            self.translation_cam0_to_cam2 = trans_02
            self.rotation_cam2_to_cam0 = rot_20
            self.translation_cam2_to_cam0 = trans_20

        rot[0], trans[0] = rot_01, trans_01
        rot[1], trans[1] = rot_10, trans_10
        rot[2], trans[2] = rot_23, trans_23
        rot[3], trans[3] = rot_32, trans_32

        R00, R01, P00, P01, Q = fisheyeStereoRectify(
            mtx[0], dist[0][0:4], 
            mtx[1], dist[1][0:4], 
            rot_10, trans_10, True, (640, 480))

        R10, R11, P10, P11, Q = fisheyeStereoRectify(
            mtx[2], dist[2][0:4], 
            mtx[3], dist[3][0:4], 
            rot_32, trans_32, True, (640, 480))

        self.stereo_rectified_P = [P00, P01, P10, P11]
        self.stereo_rectified_R = [R00, R01, R10, R11]
    
        self.rot_01 = rot_01
        self.trans_01 = trans_01

        self.ego_R = cv2.Rodrigues(np.eye(3))[0]
        self.ego_t = np.zeros([3, 1])
        self.initial_origin = np.zeros((3, 1))
        self.pose_t = np.zeros((3, 1))
        self.pose_R = np.eye(3)

        if DATASET == 'kite' and self.syncer:
            self.initial_origin, self.pose_R = self.syncer.get_initial_pose()
            print('*************', self.initial_origin, '*************')

        # Compute the fundamental matrix
        for left in range(self.num_cams):
            right = self.STEREOCONFG[left]
            self.navcams.append(navcam(left, 
                self.STEREOCONFG[left], 
                mtx[left], dist[left], rot[left], trans[left], 
                self.num_features,
                self.stereo_rectified_P,
                self.stereo_rectified_R))

        # Set each camera's stereo config
        for left in range(self.num_cams):
            self.navcams[left].set_stereo_pair(self.navcams[self.STEREOCONFG[left]])

    def close(self):
        self.json_log_file.close()
        self.json_log_file = None
        self.csv_log_file.close()
        self.csv_log_file = None

    def load_kitti_gt(self, frame_id):  #specialized for KITTI odometry dataset
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
       
    def update_keypoints(self, img_id):
        for i in range(self.num_cams):
            if i in CAMERA_LIST:
                self.navcams[i].keypoint_detection()

    def update_inter_optflow(self):
        for i in range(self.num_cams):
            if i in CAMERA_LIST:
                self.navcams[i].inter_sparse_optflow()

    def update_intra_optflow(self):
        if self.img_idx > 0:
            for i in range(self.num_cams):
                if i in CAMERA_LIST:
                    self.navcams[i].intra_sparse_optflow()
              
    def filter_keypoints_outliers(self, debug=False):
        for c in range(self.num_cams):
            if c in CAMERA_LIST:
                self.navcams[c].filter_keypoints(debug=debug)

    def load_features_from_json(self, json_feature_path, NUM_CAMS=4, PREFIX='frame'):
        json_name = PREFIX + str(self.img_idx) + '.json'
        json_file = os.path.join(json_feature_path, json_name)
        if not os.path.exists(json_file):
            print(json_file + ' not exist')
            return 
        print('loading', json_file)
        with open(json_file, 'r') as f:
            data = json.load(f)
            for i in range(NUM_CAMS):
                try:
                    flow01 = np.array(data['cam'+str(i)]['flow01'], dtype=np.float32).reshape(-1, 4)
                    flow013 = np.array(data['cam'+str(i)]['flow013'], dtype=np.float32).reshape(-1, 6)
                    self.navcams[i].flow_intra0 = np.array(flow01[:, 0:2], dtype=np.float64)
                    self.navcams[i].flow_intra1 = np.array(flow01[:, 2:4], dtype=np.float64)

                    self.navcams[i].flow_intra_inter0 = np.array(flow013[:, 0:2], dtype=np.float64)
                    self.navcams[i].flow_intra_inter1 = np.array(flow013[:, 2:4], dtype=np.float64)
                    self.navcams[i].flow_intra_inter3 = np.array(flow013[:, 4:6], dtype=np.float64)
                except:
                    print(json_file, 'error for loading feature for cam ', i)

    def upload_images_acsmeta(self, imgs_x4, ts):
        self.img_idx += 1
        for c in range(self.num_cams):        
            self.navcams[c].update_image(imgs_x4)        
        if ts and DATASET == 'kite' and self.syncer:
            self.prev_acsmeta = self.curr_acsmeta
            self.curr_acsmeta = self.syncer.get_closest_acs_metadata(self.img_idx)

    def compute_acsmeta_transformation_1_to_0(self):
        if self.curr_acsmeta is None or self.prev_acsmeta is None:
            return None
        # transformation at time t0 and t1 with respect to takeoff NED
        Rw0, tw0 = get_position_orientation_from_acsmeta(self.prev_acsmeta)
        Rw1, tw1 = get_position_orientation_from_acsmeta(self.curr_acsmeta)

        r1_inv, t1_inv = invert_RT(Rw1, tw1)

        rotation_0_to_1 = r1_inv.dot(Rw0)
        translation_0_to_1 = r1_inv.dot(tw0) + t1_inv
        
        time_diff = (self.curr_acsmeta[1] - self.prev_acsmeta[1]) / 1e3
        # Convert the transformation to '1 to 0'
        rotation, translation = invert_RT(rotation_0_to_1, translation_0_to_1)
    
        return rotation, translation, time_diff

    def get_acs_pose(self, index):
        return self.syncer.get_closest_position(index)
        
    def read_one_image(self, img_idx):
        ts = -1 
        if self.dataset == 'kitti':
            return read_kitti_image(self.camera_images, self.num_cams, img_idx), ts
        elif self.dataset == 'kite':
            img, ts = read_kite_image(self.camera_images, self.num_cams, KITE_VIDEO_FORMAT, img_idx)
            if img is None:
                return None, ts
            if KITE_UNDISTORION_NEEDED:
                img_bgr = []
                for i in range(4):
                    img[i] = undistortImage(img[i], self.navcams[i].calib_K0, self.navcams[i].calib_K, self.navcams[i].calib_d)
                    #tmp = cv2.cvtColor(img[i], cv2.COLOR_GRAY2BGR)
                    #img_bgr.append(tmp)
                #img_ = concat_images_list(img_bgr)
                #cv2.imwrite('/tmp/' + str(img_idx) + '.jpg', img_)
            # import pdb; pdb.set_trace()
            ts = self.syncer.get_image_timestamp(img_idx)
            return img, ts
        else:
            raise ValueError('read_one_image failed: unsupported dataset')

    def local_ego_motion_solver(self, cam_list=[0]):
        for c in cam_list:
            if self.navcams[c].flow_intra_inter0 is None:
                continue
            if len(self.navcams[c].flow_intra_inter0) > 0:
                res = self.navcams[c].local_ego_motion_solver()

    def update_global_camera_pose_egomotion(self, R, t):
        if R.shape != (3, 3):
            R = cv2.Rodrigues(R.reshape(3,1))[0]

        self.ego_R = R
        self.ego_t = t
        self.prev_scale = norm(t) if norm(t) > 0.01 else self.prev_scale
        if norm(t) > 10:
            return self.pose_R, self.pose_t
        # C(k) = C(k-1) * T(k, k-1)
        self.pose_t = self.pose_t + self.pose_R.dot(t) 
        self.pose_R = self.pose_R.dot(R)
        return self.pose_R, self.pose_t

    def get_egomotion(self):
        return self.ego_R, self.ego_t

    def get_global_camera_pose(self):
        return self.pose_R, self.pose_t
    
    def global_fun(self, x0, cam_obs, y_meas, cam_list=range(4)):
        num_cams = len(cam_list)
        rot_vecs   = x0[0:3]
        trans_vecs = x0[3:6]
        x0_offset = 6
        y_offset = 0
        cost_err = None
        
        rotation_list = []
        translation_list = []

        rotation_camera0_frame, translation_camera0_frame = cv2.Rodrigues(x0[0:3])[0], x0[3:6].reshape(3, 1)

        rotation_camera1_frame, translation_camera1_frame = transform_egomotion_from_frame_a_to_b(rotation_camera0_frame, 
                                                                                                  translation_camera0_frame, 
                                                                                                  self.navcams[0].calib_R, 
                                                                                                  self.navcams[0].calib_t)
        rotation_list.append(rotation_camera0_frame)
        translation_list.append(translation_camera0_frame)

        rotation_list.append(rotation_camera1_frame)
        translation_list.append(translation_camera1_frame)

        if DATASET == 'kite':
            rotation_camera2_frame, translation_camera2_frame = transform_egomotion_from_frame_a_to_b(rotation_camera0_frame, 
                                                                                                    translation_camera0_frame, 
                                                                                                    self.rotation_cam0_to_cam2, 
                                                                                                    self.translation_cam0_to_cam2)

            rotation_camera3_frame, translation_camera3_frame = transform_egomotion_from_frame_a_to_b(rotation_camera2_frame, 
                                                                                                    translation_camera2_frame, 
                                                                                                    self.navcams[2].calib_R, 
                                                                                                    self.navcams[2].calib_t)

            rotation_list.append(rotation_camera2_frame)
            translation_list.append(translation_camera2_frame)

            rotation_list.append(rotation_camera3_frame)
            translation_list.append(translation_camera3_frame)

        cv2.Rodrigues(rotation_camera1_frame)[0]

        for index, c in enumerate(cam_list):
            if c == 0:
                egomotion_rotation = rotation_camera0_frame
                egomotion_translation = translation_camera0_frame
            elif c == 1:
                egomotion_rotation = rotation_camera1_frame
                egomotion_translation = translation_camera1_frame
            elif c == 2:
                egomotion_rotation = rotation_camera2_frame
                egomotion_translation = translation_camera2_frame
            else:
                egomotion_rotation = rotation_camera3_frame
                egomotion_translation = translation_camera3_frame

            rot_vecs = cv2.Rodrigues(egomotion_rotation)[0]
            trans_vecs = egomotion_translation

            n_obj_013, n_obj_01 = cam_obs[index]

            cur_cam = self.navcams[c]
            camera_matrix = cur_cam.calib_K
            stereo_matrix = cur_cam.stereo_pair_cam.calib_K
            stereo_rotation = cur_cam.calib_R2
            stereo_translation = cur_cam.calib_t

            if n_obj_013 > 0:      
                points_013 = x0[x0_offset: x0_offset + 3 * n_obj_013].reshape(-1, 3)
                flow013_0  = y_meas[y_offset: y_offset + n_obj_013]
                flow013_1  = y_meas[y_offset+ n_obj_013: y_offset + 2 * n_obj_013]
                flow013_3  = y_meas[y_offset + 2 * n_obj_013: y_offset + 3 * n_obj_013]

                points_3x1 = points_013
                observations_2x1 = flow013_1
                camera_matrix_3x3 = camera_matrix
                rotation_vector_3x1 = rot_vecs
                translation_vector_3x1 = trans_vecs

                flow0_err = reprojection_error(points_013, flow013_0, camera_matrix)
                flow1_err = reprojection_error(points_013, flow013_1, camera_matrix, rot_vecs, trans_vecs)
                flow3_err = reprojection_error(points_013, flow013_3, stereo_matrix, stereo_rotation, stereo_translation)

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
                
                flow0_err = reprojection_error(points_01, flow01_0, camera_matrix)
                flow1_err = reprojection_error(points_01, flow01_1, camera_matrix, rot_vecs, trans_vecs)

                flow01_errs = TWO_VIEW_FEATURE_WEIGHT * np.vstack((flow0_err, flow1_err)) * 1.0
        
                x0_offset += 3 * n_obj_01
                y_offset += 2 * n_obj_01
                if cost_err is None:
                    cost_err = flow01_errs
                else:
                    cost_err = np.vstack([cost_err, flow01_errs])
                # import pdb ; pdb.set_trace()
            if cost_err is None:
                return None
        return cost_err.ravel()

    def mono_egomotion_estimation(self, cam_list):
        mono_rotation = {}
        mono_translation = {}
        for cam_idx in cam_list:
            try:
                # import pdb ; pdb.set_trace()
                rot, trans = self.navcams[cam_idx].mono_ego_motion_estimation()
                mono_rotation[cam_idx] = rot
                mono_translation[cam_idx] = trans
            except ValueError:
                print('estimate R, t from camera {} failed'.format(cam_idx))
        return mono_rotation, mono_translation

    def get_initial_egomotion_from_mono_estimation(self, cam_list):
        mono_rotation, mono_translation = self.mono_egomotion_estimation(cam_list)
        # import pdb; pdb.set_trace()
        for cam_idx in cam_list:
            rot, trans = mono_rotation[cam_idx], mono_translation[cam_idx]
            if all(v is not None for v in [rot, trans]):
                if cam_idx >= 2:
                    rotation_cam2_to_cam0 = self.rotation_cam2_to_cam0
                    translation_cam2_to_cam0 = self.translation_cam2_to_cam0
                    rot2, trans2 = transform_egomotion_from_frame_a_to_b(rot, trans, rotation_cam2_to_cam0, translation_cam2_to_cam0)
                    return rot2, trans2, cam_idx, mono_rotation, mono_translation
                return rot, trans, cam_idx, mono_rotation, mono_translation
        return None, None, -1, None, None

    def update_pose_by_prev_motion(self):
        return self.update_global_camera_pose_egomotion(self.prev_rot, self.prev_trans)

    def nonUniqueSolutionDetetion(self, x,  cam_obs, y_meas, cam_list, scale=2.5):
        x_scaled = x.copy()
        err0 = self.global_fun(x, cam_obs, y_meas, cam_list)
        x_scaled[3:] = scale * x_scaled[3:]
        err1 = self.global_fun(x_scaled, cam_obs, y_meas, cam_list)
        diff = np.absolute(err0 - err1)
        diff_norm = norm(diff)
        print(diff_norm)
        # import pdb; pdb.set_trace()
        

    def egomotion_solver(self, img_idx=None, ts = None, cam_list=[0, 1], est=None, debug_json_path=None):
        if img_idx is None or img_idx == 0:
            return self.update_pose_by_prev_motion()

        num_cams = len(cam_list)
        cam_obs = np.zeros([num_cams, 2], dtype=np.int)
        y_meas = None

        print(ts, self.prev_ts)
        time_diff_image = (ts - self.prev_ts) / 1e3
        # import pdb; pdb.set_trace()

        time_diff = min(0.067, time_diff_image)

        if DATASET == 'kite' and self.syncer:
            Rw0 = self.syncer.get_closest_orentation(self.img_idx)
            # Rw0, tw0 = get_position_orientation_from_acsmeta(self.prev_acsmeta)
            # Rw1, tw1 = get_position_orientation_from_acsmeta(self.curr_acsmeta)
            # angular_vel0, linear_vel_ned0 = get_angular_linear_velocity_from_acsmeta(self.curr_acsmeta)

        if EGOMOTION_SEED_OPTION == 0:
            # Estimate each camera's local egomotion by 5 point algorithm   
            rot0, trans0, camera_index, rot_list, trans_list = self.get_initial_egomotion_from_mono_estimation(cam_list)

            if camera_index == -1:
                print('error: no initial estimation from 5 point are avalaible')
                return self.update_pose_by_prev_motion()
            x0 = np.vstack([rot0, trans0])
        else:
            if not self.prev_invalid and 0:
                x0 = self.prev_egomotion.reshape(6,1)
            elif ts is not None and self.syncer is not None:                
                # angular_vel0, linear_vel0 = self.syncer.get_closest_velocity(ts)
                angular_vel0 = 0
                if angular_vel0 is not None:
      
                    # linear_vel0 = Rw0.T.dot(linear_vel_ned0)

                    # imu_rot_est = angular_velocity_to_rotation_matrix(angular_vel0, time_diff)
                    # imu_trans_est = linear_velocity_to_translation(linear_vel0, time_diff)  
                    # init_motion_angle,  imu_trans_est = self.syncer.get_init_motion(ts)
                  
                    # imu_rot_est = cv2.Rodrigues(init_motion_angle)[0]
                    # pose_rot, pose_trans_bd, time_diff_pose = self.compute_acsmeta_transformation_1_to_0()
                    # pose_ang_vel = cv2.Rodrigues(pose_rot)[0] / time_diff_pose

                    # pose_lin_vel = pose_trans_bd / time_diff_pose
                    # pose_lin_vel_ned = Rw0.dot(pose_lin_vel)

                    # pose_rot_est = angular_velocity_to_rotation_matrix(pose_ang_vel, time_diff)
                    # pose_trans_est = linear_velocity_to_translation(pose_lin_vel, time_diff)  

                    # Conver body transformation from body to camera0
                    if EGOMOTION_SEED_OPTION == 1:
                        x0 = self.syncer.get_init_motion(self.img_idx)
                        # import pdb; pdb.set_trace()
                        # import pdb; pdb.set_trace()
                        # rotation0, translation0 = transform_egomotion_from_frame_a_to_b(
                        #         imu_rot_est, 
                        #         imu_trans_est, 
                        #         self.rotation_body_to_cam0, 
                        #         self.translation_body_to_cam0)
                    else:
                        # rotation0, translation0 = transform_egomotion_from_frame_a_to_b(
                        #     pose_rot_est, 
                        #     pose_trans_est, 
                        #     self.rotation_body_to_cam0, 
                        #     self.translation_body_to_cam0)                    
                        # x0 = np.vstack([cv2.Rodrigues(rotation0)[0], translation0])
                        # assert(0)
                        import pdb; pdb.set_trace()
 
        # import pdb; pdb.set_trace()
        self.prev_ts = ts
    
        json_data = {}
        json_data['egomotion'] = {}
        json_data['egomotion']['initial'] = x0.ravel().tolist()
        json_data['image_timestamp'] = ts
        json_data['dsp_timestamp'] = self.syncer.get_image_dsp_timestamp(img_idx)
    
        rot0 = x0[0:3]
        trans0 = x0[3:6]

        rotation_list = []
        translation_list = []

        rotation_camera0_frame, translation_camera0_frame = cv2.Rodrigues(x0[0:3])[0], x0[3:6].reshape(3, 1)

        rotation_camera1_frame, translation_camera1_frame = transform_egomotion_from_frame_a_to_b(rotation_camera0_frame, 
                                                                                                  translation_camera0_frame, 
                                                                                                  self.navcams[0].calib_R, 
                                                                                                  self.navcams[0].calib_t)
        rotation_list.append(rotation_camera0_frame)
        translation_list.append(translation_camera0_frame)

        rotation_list.append(rotation_camera1_frame)
        translation_list.append(translation_camera1_frame)

        if DATASET == 'kite':
            rotation_camera2_frame, translation_camera2_frame = transform_egomotion_from_frame_a_to_b(rotation_camera0_frame, 
                                                                                                    translation_camera0_frame, 
                                                                                                    self.rotation_cam0_to_cam2, 
                                                                                                    self.translation_cam0_to_cam2)

            rotation_camera3_frame, translation_camera3_frame = transform_egomotion_from_frame_a_to_b(rotation_camera2_frame, 
                                                                                                    translation_camera2_frame, 
                                                                                                    self.navcams[2].calib_R, 
                                                                                                    self.navcams[2].calib_t)

            rotation_list.append(rotation_camera2_frame)
            translation_list.append(translation_camera2_frame)

            rotation_list.append(rotation_camera3_frame)
            translation_list.append(translation_camera3_frame)
            
        for index, c  in enumerate(cam_list):
            egomotion_rotation = rotation_list[c]
            egomotion_translation = translation_list[c]

            if self.navcams[c].flow_intra_inter0 is None:
                continue

            n_bad = self.navcams[c].filter_keypoints_extra(egomotion_rotation, egomotion_translation)

            cur_cam = self.navcams[c]            
            K0 = cur_cam.calib_K
            K1 = cur_cam.stereo_pair_cam.calib_K
            n_obs_i = cur_cam.flow_intra_inter0.shape[0]
            n_obs_j = cur_cam.flow_intra0.shape[0]
            # n_obs_i = 0

            # import pdb; pdb.set_trace()
            if not USE_01_FEATURE:
                n_obs_j = 0

            json_data['cam'+str(c)] = {}
            json_data['cam'+str(c)]['n_flow013'] = n_obs_i
            json_data['cam'+str(c)]['n_flow01'] = n_obs_j
            json_data['cam'+str(c)]['raw_kpts'] = self.navcams[c].flow_kpt0.ravel().tolist()
            json_data['cam'+str(c)]['n_raw_kpts'] = MAX_NUM_KEYPOINTS

            if n_obs_i > 0:
                flow0 = cur_cam.flow_intra_inter0
                flow1 = cur_cam.flow_intra_inter1
                flow3 = cur_cam.flow_intra_inter3    
                points013, terr013_0, terr013_1 = triangulate_3d_points(flow0, flow3, 
                                           K0, 
                                           K1, 
                                           cur_cam.calib_R, 
                                           cur_cam.calib_t)

                flow013_z = np.vstack([flow0, flow1, flow3])
                y_meas = flow013_z if y_meas is None else np.vstack([y_meas, flow013_z])

                x0 = np.vstack([x0, points013.ravel().reshape(-1, 1)])

                flow013_json = np.hstack([flow0, flow1, flow3])
                json_data['cam'+str(c)]['flow013'] = flow013_json.ravel().tolist()
                json_data['cam'+str(c)]['flow013_init'] = points013.ravel().tolist()

            if n_obs_j > 0:
                flow0 = cur_cam.flow_intra0
                flow1 = cur_cam.flow_intra1
                try:
                    # avg_dept = np.mean(points013[:,2])
                    # import pdb; pdb.set_trace()
                    points01, terr01_0, terr01_1 = triangulate_3d_points(flow0, flow1, K0, K0, egomotion_rotation, egomotion_translation)
                except:
                    print('triangulate_3d_points failed ' )
                    return self.update_pose_by_prev_motion()


                x0 = np.vstack([x0, points01.ravel().reshape(-1, 1)])
                flow01_z = np.vstack([flow0, flow1])
                y_meas = flow01_z if y_meas is None else np.vstack([y_meas, flow01_z])

                flow01_json = np.hstack([flow0, flow1])
                json_data['cam'+str(c)]['flow01'] = flow01_json.ravel().tolist()
                json_data['cam'+str(c)]['flow01_init'] = points01.ravel().tolist()

            cam_obs[index][0] = n_obs_i
            cam_obs[index][1] = n_obs_j

        if y_meas is None or y_meas.shape[0] < 9:
            print('y_meas is None' )
            return self.update_pose_by_prev_motion()

        x0 = x0.flatten()

        sparse_A = global_bundle_adjustment_sparsity(cam_obs, n_cams=num_cams) 

        t0 = datetime.now()
        res = None
        err0 = self.global_fun(x0, cam_obs, y_meas, cam_list)

        opt_failed = False

        try:
            res = least_squares(self.global_fun, x0, args=(cam_obs, y_meas, cam_list), jac_sparsity=sparse_A, **LS_PARMS)
        except:
            opt_failed = True
        
        # import pdb; pdb.set_trace()
        if res is None:
            opt_failed = True
            print("optimization failed")
            # return self.update_pose_by_prev_motion()
        else:
            t1 = datetime.now()
            ego_elapsed = t1 - t0

            err1 = self.global_fun(res.x, cam_obs, y_meas, cam_list)
            reprojection_err = norm(err1)

            R = cv2.Rodrigues(res.x[0:3])[0]
            t = res.x[3:6]       

            json_data['egomotion']['optimized'] = res.x[0:6].ravel().tolist()
    
            x_offset = 6
            for c in cam_list:
                n_obj_013, n_obj_01 = cam_obs[index]
                if n_obj_013 > 0:      
                    points013 = res.x[x_offset: x_offset + 3 * n_obj_013].flatten()
                    json_data['cam'+str(c)]['flow013_opt'] = points013.tolist()
                    x_offset += 3 * n_obj_013
                    # import pdb ; pdb.set_trace()
                    
                if n_obj_01 > 0:
                    points_01  = res.x[x_offset: x_offset + 3 *n_obj_01].flatten()
                    json_data['cam'+str(c)]['flow01_opt'] = points_01.tolist()       
                    x_offset += 3 * n_obj_01
                # import pdb ; pdb.set_trace()

        if os.path.exists(debug_json_path):
            outfile = os.path.join(debug_json_path, 'frame'+str(img_idx)+'.json')
            with open(outfile, 'w') as f:
                json.dump(json_data, f, sort_keys=True, indent=4)

        if opt_failed:
            print('optimization failed')
            return self.update_pose_by_prev_motion()
        t = t.reshape(3,1)

        acs_rot, acs_trans = transform_egomotion_from_frame_a_to_b(R, t, self.rotation_cam0_to_body, self.translation_cam0_to_body)
        acs_rot_aa = cv2.Rodrigues(acs_rot)[0]
        

        acs_rot_aa2, acs_trans2 = transform_velocity_from_frame_a_to_b(res.x[0:3], res.x[3:6], self.rotation_cam0_to_body, self.translation_cam0_to_body)
        

        # Note: the actual egomotion should be inverted as we estimate R,t from current feature to previous 
        avg_reprojection_err = reprojection_err / float(np.sum(cam_obs))

        print('img:' + str(self.navcams[0].img_idx), 
                'rot0 [%.3f, %.3f, %.3f]' % (rot0[0], rot0[1], rot0[2]),  
                'rot1 [%.3f, %.3f, %.3f]' % (res.x[0], res.x[1], res.x[2]),  
                'trans0 [%.3f, %.3f, %.3f]'  % (trans0[0], trans0[1], trans0[2]), 
                'trans1 [%.3f, %.3f, %.3f]'  % (t[0], t[1], t[2]), 
                'proj_err [%.3f, %.3f]' % (norm(err0), norm(err1)))
        
        gps_ned_vel = self.syncer.get_current_gps_vel(img_idx).reshape(3, 1)
        gps_body = Rw0.T.dot(gps_ned_vel)

        ang_vel_est = acs_rot_aa  / time_diff
        lin_vel_est = acs_trans  / time_diff

        large_vel_detected = False
        for i in range(3):
            if lin_vel_est[i] > MAX_BODY_VEL[i]:
                large_vel_detected = True
                lin_vel_est[i] = min(lin_vel_est[i], MAX_BODY_VEL[i])


        ned_lin_vel_est = Rw0.dot(lin_vel_est)


        print('======================================')

        print('ts: %d | bd-gps-mot: [%.3f, %.3f, %.3f], norm: %f' % (
            ts, 
            gps_body[0], 
            gps_body[1], 
            gps_body[2],
            norm(gps_body)))


        print('ts: %d | bd-est-mot: [%.3f, %.3f, %.3f], norm: %f' % (
            ts, 
            lin_vel_est[0], 
            lin_vel_est[1], 
            lin_vel_est[2], 
            norm(lin_vel_est)
            ))


        opt_pts = res.x[6:].reshape(-1,3)
        opt_pts_depth = opt_pts[:,2]
        opt_mask = np.logical_and(opt_pts_depth < MAX_DEPTH, opt_pts_depth > MIN_DEPTH)

        jac = res.jac.toarray()
        cov_cam = covariance_mvg_A6_4(jac, opt_mask)
        cov_cam2 = covarinace_svd(jac)

        
        aug_rot0 = np.zeros([6, 6])
        aug_rot0[0:3,0:3] = self.rotation_cam0_to_body
        aug_rot0[3:6,3:6] = self.rotation_cam0_to_body

        aug_rot1 = np.zeros([6, 6])
        aug_rot1[0:3,0:3] = Rw0
        aug_rot1[3:6,3:6] = Rw0

        cov_body = aug_rot0.dot(cov_cam).dot(aug_rot0.T)
        cov_ned = aug_rot1.dot(cov_body).dot(aug_rot1.T)

        cov_body_diag = cov_body.diagonal()
        cov_ned_diag = cov_ned.diagonal()


        dtdt = time_diff * time_diff
        lin_cov_body = cov_body_diag[3:6] / dtdt
        ang_cov_body = cov_body_diag[0:3] / dtdt
        lin_cov_ned = cov_ned_diag[3:6] / dtdt

        print('ts: %d | con-lin: [%.3f, %.3f, %.3f] | val: %d' % (
            ts, 
            lin_cov_body[0], 
            lin_cov_body[1], 
            lin_cov_body[2], 
            res.success))

        print('======================================')


        norm_diff_gps = abs(norm(gps_body) - norm(lin_vel_est))
        norm_diff_gps_ratio = norm_diff_gps / abs(norm(gps_body))

        norm_diff_prev = norm(self.prev_trans) - norm(lin_vel_est)
        max_acc = 6
        cur_acc = norm_diff_prev / time_diff

        # if norm_diff_gps_ratio > 0.5:
        #     import pdb; pdb.set_trace()

        valid = True
        if (cur_acc > max_acc and res.success == False) or avg_reprojection_err > 10 or large_vel_detected:
            valid = False
            for i in range(3):
                ang_cov_body[i] = 9.9
                lin_cov_body[i] = 9.9
                lin_cov_ned[i] = 9.9   

        
        self.nonUniqueSolutionDetetion(res.x, cam_obs, y_meas, cam_list)
        # import pdb; pdb.set_trace()

# ============================================
        jo = self.syncer.get_json_object(img_idx)
        jo['body_ang_vel'] = ang_vel_est.tolist()
        jo['body_lin_vel'] = lin_vel_est.tolist()
        jo['ned_lin_vel'] = ned_lin_vel_est.tolist()


        jo['body_ang_vel_conf'] = ang_cov_body.tolist()
        jo['body_lin_vel_conf'] = lin_cov_body.tolist()
        jo['ned_lin_vel_conf'] = lin_cov_ned.tolist()
    
        jo['body_lin_vel_status'] = valid
        jo['body_ang_vel_status'] = valid

        jo['init_motion'] = x0[0:6].tolist()
        jo['opt_motion'] = res.x[0:6].tolist()

        list_json = [jo[k] for k in self.syncer.get_json_keys()]
        if self.csv_writer:
            self.csv_writer.writerow(list_json)
        if self.json_log_file:
            json.dump(jo, self.json_log_file, sort_keys=True)
            self.json_log_file.write('\n')
# ============================================
        if avg_reprojection_err > AVG_REPROJECTION_ERROR:
            self.prev_invalid = True
            print('*****reject the results: reprojection error: ' + str(reprojection_err) + ' avg reprjection error: ' + str(avg_reprojection_err))
            return self.update_pose_by_prev_motion()
        
        self.prev_egomotion = res.x[0:6]
        self.prev_invalid = False

        if DATASET == 'kite':
            self.prev_rot = ang_vel_est.reshape(3,1)
            self.prev_trans = ned_lin_vel_est.reshape(3,1)
        
        pose_R, pose_t = self.update_global_camera_pose_egomotion(acs_rot, acs_trans)
        return pose_R, pose_t 
        
def _main(args):
    input_path = os.path.expanduser(args.images_path)
    output_path = os.path.expanduser(args.output_path)
    calib_file = os.path.expanduser(args.calib_path)
    json_enabled = args.enable_json
    num_features = args.num_features
    num_cam = args.num_cameras
    dataset = args.dataset_type.lower()
    seq = ("%02d" % (args.seq_num))  if (dataset == 'kitti') else str(args.seq_num)
    external_json_pose = args.external_json_pose
    json_feature_path = args.json_feature_path

    if not os.path.exists(calib_file):
        raise Exception('no valid calib fie')

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    em_pars = dict(input_path=input_path, 
                   data_seq=seq, 
                   calib_file=calib_file, 
                   num_features=num_features, 
                   dataset=dataset, 
                   json_output=json_enabled)

    kv = EgoMotion(**em_pars)

    traj = np.zeros((960, 1280, 3), dtype=np.uint8)

    # text = "Aircraft Moving E/W"
    # cv2.putText(traj, text, (700, 58), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
    
    text = "N/E Traj by Fight Control"
    cv2.line(traj,(700, 78),(750, 78),(255,255,255),2)
    cv2.putText(traj, text, (775, 85), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

    text = "N/E Traj by Vision with One IMU Sample Every Frame"
    cv2.line(traj,(700, 98),(750, 98),EGOMOTION_TRAJ_COLOR,2)
    cv2.putText(traj, text, (775, 105), cv2.FONT_HERSHEY_PLAIN, 1, EGOMOTION_TRAJ_COLOR, 1, 8)

    traj_index = 0
    json_pose = None
    n_json_pose = 0
    if external_json_pose:
        external_json_pose = os.path.expanduser(args.external_json_pose)
        with open(external_json_pose) as f:
            json_pose = json.load(f)
            n_json_pose = json_pose['num_poses']

    x, y, z = np.array([0., 0., 0.]).reshape(3,1)
    global_tr = [0, 0, 0]

    for img_id in range(kv.num_imgs):
        if img_id > NUM_FRAMES:
            break
        if img_id < START_INDEX:
            continue
        
        if dataset == 'kitti':
            kv.load_kitti_gt(img_id)

        camera_images, ts = kv.read_one_image(img_id)

        if camera_images == None:
            continue
        kv.upload_images_acsmeta(camera_images, ts)
        if external_json_pose:
            if img_id < 1:
                global_tr = [0, 0, 0]
                continue
            
            if img_id >= n_json_pose:
                break
            frame_name = 'frame' + str(img_id)
            try:
                estimated_rt = json_pose[frame_name]['optimized']
                R = np.array([estimated_rt[0:3]])
                R = cv2.Rodrigues(R)[0]
                t = np.array([estimated_rt[3:]]).reshape(3,1)
                # import pdb; pdb.set_trace()
                acs_rot, acs_trans = transform_egomotion_from_frame_a_to_b(
                        R, t, kv.rotation_cam0_to_body, kv.translation_cam0_to_body)

                print(frame_name, img_id, estimated_rt)
                _, global_tr = kv.update_global_camera_pose_egomotion(acs_rot, acs_trans.reshape(3,1))
            except:
                print('warning, no ' + frame_name + ' estimation from json')
                global_tr = kv.pose_t
        else:
            if json_feature_path is None:
                kv.update_keypoints(img_id)
                kv.update_inter_optflow()
                kv.update_intra_optflow()
                kv.filter_keypoints_outliers(debug=DEBUG_KEYPOINTS)
            else:
                kv.load_features_from_json(json_feature_path)

            global_rot, global_tr = kv.egomotion_solver(img_id, ts,
                                                        cam_list=CAMERA_LIST, 
                                                        debug_json_path=PYEGO_DEBUG_OUTPUT)


        if img_id == 0:
            x, y, z = 0, 0, 0
        else:
            x, y, z = global_tr[0], global_tr[1], global_tr[2]

        print('===================')
        print('img_id', img_id)
        print('Benchmark Position', kv.trueX, kv.trueY, kv.trueZ)
        print('Egomotion Position', x, y, z)
        print('===================')

        draw_ofs_x = 750
        draw_ofs_y = 450
        if DATASET == 'kitti':
            draw_x0, draw_y0 = int(x) + draw_ofs_x, int(z) + draw_ofs_y    
            true_x, true_y = int(kv.trueX) + draw_ofs_x, int(kv.trueZ) + draw_ofs_y
        else:
            draw_x0, draw_y0 = int(x) + draw_ofs_x, int(y) + draw_ofs_y  
            current_true_pose = np.array(kv.get_acs_pose(img_id))

            if current_true_pose is not None:
                global_positon = current_true_pose.reshape(1,3) - kv.initial_origin.reshape(1,3)
                kv.trueX, kv.trueY, kv.trueZ = global_positon.reshape(-1,)
            true_x, true_y = int(kv.trueX) + draw_ofs_x, int(kv.trueY) + draw_ofs_y

        cv2.circle(traj, (draw_x0, draw_y0), 1, EGOMOTION_TRAJ_COLOR, 1)
        cv2.circle(traj, (true_x,true_y), 1, GT_TRAJ_COLOR, 2)



        cv2.rectangle(traj, (700, 120), (1280, 200), (0,0,0), -1)

        if dataset == 'kitti':
            text = "Image: %d: X= %.2fm Y= %.2fm Z= %.2fm"%(img_id, x, y, z)
            cv2.putText(traj, text, (700,140), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
        else:
            text = "Img: %d N: %.2fm E: %.2fm D: %.2fm"%(img_id, x, y, z)

            text1 = "Trajectory Err:      [N: %8.1f%%, E: %8.1f%%, D: %8.1f%%]"%( 
                100 * abs(abs(kv.trueX) - abs(x)) / abs(x),
                100 * abs(abs(kv.trueY) - abs(y)) / abs(y),
                100 * abs(abs(kv.trueZ) - abs(z)) / abs(z))

            # text2 = "Linear Velocity Err: [N: %8.1f%%, E: % 8.1f%%, D: % 8.1f%%]"%(
            #     kv.ned_vel_err[0], 
            #     kv.ned_vel_err[1], 
            #     kv.ned_vel_err[2])

            cv2.putText(traj, text, (700,140), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
            cv2.putText(traj, text1, (700,160), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)
            # cv2.putText(traj, text2, (700,180), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

        img_bgr = []
        for i in CAMERA_LIST:
            img = cv2.resize(cv2.cvtColor(camera_images[i], cv2.COLOR_GRAY2BGR), (320, 240))
            img_bgr.append(img)
        img_ = concat_images_list(img_bgr)
        img_ = cv2.imread(kv.camera_images[img_id])
        img_ = cv2.resize(img_, (640, 960))


        h1, w1 = img_.shape[:2]
        traj[:h1, :w1,:3] = img_
        traj_name = str(traj_index) + '.jpg'
        traj_index += 1
        cv2.imwrite(os.path.join(output_path, traj_name), traj)
    # Close and release the log file object
    kv.close()

if __name__ == '__main__':
    _main(parser.parse_args())
