
import numpy as np
import glob, pdb, math, json
import warnings
import os, io, libconf, copy
import cv2, Image

from numpy.linalg import inv, pinv, norm

from scipy.sparse import lil_matrix
from scipy.optimize import least_squares

import time
from datetime import datetime

import matplotlib.pyplot as plt
from utils import *
from cfg import *

import argparse

parser = argparse.ArgumentParser(
    description='Compute camera relative pose on input stereo or quad images')

parser.add_argument(
    '-dataset',
    '--dataset_type',
    help='dataset type: (Kite, Kitti, )',
    default='kitti')

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
    default='~/vo_data/kitti/dataset')

parser.add_argument(
    '-calib',
    '--calib_path',
    help='path to directory of calibriation file (Kite)',
    default='~/vo_data/SN52/nav_calib.cfg')

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
    default=64)

parser.add_argument(
    '-num_cam',
    '--num_cameras',
    type=int,
    help='Max number of cameras used for VO',
    default=2)

parser.add_argument(
    '-ransac',
    '--enable_ransac',
    type=bool,
    help='Enable Ransac for Egomotion estimation',
    default=2)

class KTPinholeCamera:
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

kt_cam = KTPinholeCamera(1241.0, 376.0, 718.8560, 718.8560, 607.1928, 185.2157)


class navcam:
    def __init__(self,  index, stereo_pair_idx, F, intrinsic_mtx,  intrinsic_dist,  extrinsic_rot, extrinsic_trans, num_features=64):
        self.calib_K    = intrinsic_mtx
        self.calib_d    = intrinsic_dist
        self.calib_R    = extrinsic_rot
        self.calib_R2   = cv2.Rodrigues(extrinsic_rot)[0]
        self.calib_t    = extrinsic_trans

        if DATASET == 'kite':  # focal lenght and princial point of kite navcam
            self.focal      = (intrinsic_mtx[0,0] + intrinsic_mtx[1,1]) / 2.0
            self.pp         = (intrinsic_mtx[:2,2][0], intrinsic_mtx[:2,2][0])    
        else: # focal lenght and princial point of kitti stereo cameras
            self.focal      = kt_cam.fx
            self.pp         = (kt_cam.cx, kt_cam.cy) 

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
        self.F = F
        self.proj_mtx = None
        self.least_square_conf = None
        

    def mono_vo(self, abs_scale = 1.0):
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
            # import pdb; pdb.set_trace()
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

    def local_bundle_adjustment(self, init_with_mono=False, kpts013=None, kpts01=None):
        if self.img_idx is None or self.img_idx == 0:
            return
        init_est = self.generate_initial_guess(init_with_mono)
        if kpts013 is None or len(kpts013[0]) < 2:
            kpts013 = (self.flow_intra_inter0, self.flow_intra_inter1, self.flow_intra_inter3)

        if kpts01 is None or len(kpts01[0]) < 2:    
            kpts01 = (self.flow_intra0, self.flow_intra1)

        res = self.local_ego_motion_solver(init_est, kpts013, kpts01)
        
        if res[0] is None:
            est_R, est_t = init_est
            self.update_camera_pose_egomotion(cv2.Rodrigues(est_R)[0], est_t.reshape(3,))
            return 
        err_proj = res[1]
        err_level = norm(err_proj)
        if self.least_square_conf is None:
            self.least_square_conf = err_level
        avg_least_square_conf = self.least_square_conf / (self.img_idx)

        if err_level > 5 * avg_least_square_conf:
            return self.stereo_R, self.stereo_t
        else:
            self.least_square_conf += err_level

        ba_R, ba_t = res[0][0:3], res[0][3:6]
        R = cv2.Rodrigues(ba_R)[0]
        t = ba_t
        stereo_R, stereo_t = self.update_camera_pose_egomotion(R, t)

    def local_bundle_adjustment_ransac(self, init_with_mono=True):
        if self.img_idx is None or self.img_idx == 0:
            return
        proj_err_threshold = 1.0
        min_ransac_013 = 2
        min_ransac_01 = 2
        target_013_inliers_pct = 0.8
        target_01_inliers_pct = 0.8
        max_ransac_inters = 5
        min_ransac_inters = 2

        target_013_inliers_num = int(self.flow_intra_inter0.shape[0] * target_013_inliers_pct)
        target_01_inliers_num = int(self.flow_intra0.shape[0] * target_01_inliers_pct)

        inliers_013 =  [[] for i in range(3)]
        inliers_01 = [[] for i in range(2)]
    
        kpts013 = (self.flow_intra_inter0, self.flow_intra_inter1, self.flow_intra_inter3)
        kpts01 = (self.flow_intra0, self.flow_intra1)        
        init_est = self.generate_initial_guess(init_with_mono)

        R, t = init_est
        for it in range(max_ransac_inters):
            kpts013_maybe_inliers = []
            if self.flow_intra_inter0 is not None and self.flow_intra_inter0.shape[0] > min_ransac_013:
                flow013_ransac_idx = np.random.choice(range(self.flow_intra_inter0.shape[0]), min_ransac_013, replace=False)
                kpts013_maybe_inliers.append(np.take(kpts013[0], flow013_ransac_idx, axis=0))
                kpts013_maybe_inliers.append(np.take(kpts013[1], flow013_ransac_idx, axis=0))
                kpts013_maybe_inliers.append(np.take(kpts013[2], flow013_ransac_idx, axis=0))


            kpts01_maybe_inliers = []
            if self.flow_intra0 is not None and self.flow_intra0.shape[0] > min_ransac_01:
                flow01_ransac_idx = np.random.choice(range(self.flow_intra0.shape[0]), min_ransac_01, replace=False)
                kpts01_maybe_inliers.append(np.take(kpts01[0], flow01_ransac_idx, axis=0))
                kpts01_maybe_inliers.append(np.take(kpts01[1], flow01_ransac_idx, axis=0))
            # import pdb; pdb.set_trace()
            rt, proj_err0 = self.local_ego_motion_solver(init_est, kpts013_maybe_inliers, kpts01_maybe_inliers)
            if rt is None:
                continue
            R, t = rt[0:3], rt[3:6] 

            proj_err1 = self.local_ego_motion_solver((R, t), kpts013, kpts01, ba_enabled=False)[1]
            mb_in_013 = []
            mb_in_01 = []  
            for k in range(self.flow_intra_inter0.shape[0]):
                err013 = abs(proj_err1[0][k]) + abs(proj_err1[1][k]) + abs(proj_err1[2][k])
                err013 = norm(err013) / 3.0
                if err013 < proj_err_threshold:
                    mb_in_013.append(k)
            for k in range(self.flow_intra0.shape[0]):
                err01 = abs(proj_err1[3][k]) + abs(proj_err1[4][k])
                err01 = norm(err01) / 2.0
                if err01 < proj_err_threshold:
                    mb_in_01.append(k)

            in_013 = []
            in_01 = []
            # import pdb; pdb.set_trace()
            if len(mb_in_013) > len(inliers_013[0]) and len(mb_in_01) > len(inliers_01[0]):
                in_013.append(np.take(kpts013[0], mb_in_013, axis=0))
                in_013.append(np.take(kpts013[1], mb_in_013, axis=0))
                in_013.append(np.take(kpts013[2], mb_in_013, axis=0))
                in_01.append(np.take(kpts01[0],  mb_in_01,  axis=0))
                in_01.append(np.take(kpts01[1],  mb_in_01,  axis=0))
                inliers_013 = in_013
                inliers_01 = in_01

            # import pdb; pdb.set_trace()
            if len(mb_in_013) > target_013_inliers_num and it > min_ransac_inters:
                self.inliers_013 = inliers_013
                self.inliers_01 = inliers_01
                self.local_bundle_adjustment(True, inliers_013, inliers_01)
                return 

    def local_ego_motion_solver(self, init_est = (None, None), kpts013=None, kpts01=None, ba_enabled=True):
        est_R, est_t = init_est
        if kpts01 is not None and len(kpts013) == 3:
            flow_intra_inter0, flow_intra_inter1, flow_intra_inter3 = kpts013
        else:
            flow_intra_inter0, flow_intra_inter1, flow_intra_inter3 = None, None, None

        if kpts01 is not None and len(kpts01) == 2:
            flow_intra0, flow_intra1 = kpts01
        else:
            flow_intra0, flow_intra1 = None, None
    
        n_kpts_013 = flow_intra_inter0.shape[0] if flow_intra_inter0 is not None else 0
        n_kpts_01 = flow_intra0.shape[0] if flow_intra0 is not None else 0
        cam_obs = np.zeros([2,2], dtype=np.int)
        cam_obs[self.index][0] = n_kpts_013
        cam_obs[self.index][1] = n_kpts_01

        y_meas = None
        x0 = np.vstack([est_R.reshape(3,1), est_t.reshape(3,1)])

        if n_kpts_013 > 0:
            points013, terr013_0, terr013_1 = triangulate_3d_points(flow_intra_inter0, flow_intra_inter3, 
                                           self.calib_K, self.stereo_pair_cam.calib_K, 
                                           self.calib_R, self.calib_t)

            points013_flatten = points013.ravel().reshape(-1, 1)
            y_meas = np.vstack([flow_intra_inter0, flow_intra_inter1, flow_intra_inter3])
            x0 = np.vstack([x0, points013_flatten])
            if n_kpts_01 > 0:
                points01, terr01_0, terr01_1 = triangulate_3d_points(flow_intra0, flow_intra1, 
                                               self.calib_K, self.calib_K,
                                               cv2.Rodrigues(est_R)[0], est_t)

                points01_flatten = points01.ravel().reshape(-1, 1)
                x0 = np.vstack([x0, points01_flatten])
                y_meas = np.vstack([y_meas, flow_intra0, flow_intra1])
        else:
            print('WARNING: ######cam_'+ str(self.index) + ' dont have inter match. Using mono results')
            return (None, [-1.0, -1.0, -1.0])
        # import pdb; pdb.set_trace()

        # sparse_A = local_bundle_adjustment_sparsity(cam_obs[self.index], 1)
        sparse_A = global_bundle_adjustment_sparsity_opt(cam_obs, n_cams=2)   

        x0 = x0.flatten()
        ls_pars = dict(jac_sparsity=sparse_A,
                    max_nfev=5, 
                    verbose=0,
                    x_scale='jac',
                    jac='2-point',
                    ftol=0.01, 
                    xtol=0.01,
                    gtol=0.01,
                    method='trf')
                    
        x1 = x0
        t0 = datetime.now()
        if ba_enabled:
            res = least_squares(self.fun, x0, args=(cam_obs[self.index], y_meas), **ls_pars)
            x1 = res.x
            ego_elapsed = datetime.now() - t0
            print('cam_'+str(self.index), ego_elapsed.microseconds / 1000.0, self.img_idx, n_kpts_013, n_kpts_01, 'est_rot', res.x[0:3], 'est_tras', res.x[3:6])
        else:
            err_proj = self.reprojection_err(x1, cam_obs[self.index], y_meas)
            return (x1[0:6], err_proj)

        # import pdb; pdb.set_trace()
        err_proj = self.fun(x1, cam_obs[self.index], y_meas)
        return (x1[0:6], err_proj)


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
        self.flow_kpt0 = shi_tomasi_corner_detection(self.curr_img, self.num_features)
    
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
                if err > 0.05:
                    self.flow_kpt3[ct][0][0] = -30.0
                    self.flow_kpt3[ct][0][1] = -30.0
                    continue


    def debug_inter_keypoints(self, out_dir='/tmp'):
        img = None
        if self.prev_img is not None:
            img1 = cv2.cvtColor(self.curr_img, cv2.COLOR_GRAY2BGR)
            img2 = cv2.cvtColor(self.curr_stereo_img, cv2.COLOR_GRAY2BGR)
            # import pdb; pdb.set_trace()
            for pt1, pt3 in zip(self.flow_intra_inter0, self.flow_intra_inter3):
                x1, y1 = (int(pt1[0]), int(pt1[1]))
                x3, y3 = (int(pt3[0]), int(pt3[1]))
                color = tuple(np.random.randint(0,255,3).tolist())
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
            intra0 = []
            intra1 = []

            points013, terr0, terr1 = triangulate_3d_points(np.array(self.flow_kpt0).reshape(-1,2), 
                                                            np.array(self.flow_kpt3).reshape(-1,2), 
                                                            self.calib_K, self.stereo_pair_cam.calib_K, 
                                                            self.calib_R, self.calib_t)

    
            num_flow_013 = 0
            for kp0, kp1, kp3, wp in zip(self.flow_kpt0, self.flow_kpt1, self.flow_kpt3, points013):
                # for kite system, the cam0 is a logical leftcam not a physicall left cam, so
                # the feature points could behind the camera 
                if wp[2] < 0 and DATASET == 'kitti':
                    # print('warning: ignore world points behind camera')
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

            self.flow_intra_inter0 = np.array(flow_intra_inter0, dtype=np.float)
            self.flow_intra_inter1 = np.array(flow_intra_inter1, dtype=np.float)
            self.flow_intra_inter3 = np.array(flow_intra_inter3, dtype=np.float)

            self.flow_intra0 = np.array(flow_intra0, dtype=np.float)
            self.flow_intra1 = np.array(flow_intra1, dtype=np.float)

            self.flow_inter0 = np.array(flow_inter0, dtype=np.float)
            self.flow_inter3 = np.array(flow_inter3, dtype=np.float)

            self.intra0 = np.array(intra0, dtype=np.float)
            self.intra1 = np.array(intra1, dtype=np.float)

            if debug:    
                self.debug_inter_keypoints(out_dir)
                self.debug_intra_keypoints(out_dir)
                print(self.img_idx, 'cam_'+ str(self.index ), 'intra_inter:' + str(len(self.flow_intra_inter0)), 'intra:' + str(len(self.flow_intra0)), 'inter:'+str(len(self.flow_inter0)))
        
    def searlize_features_for_json(self):
        features = []
        feat013_dict = {}
        feat01_dict = {}
        # import pdb; pdb.set_trace()
        flow013_0 = [] if self.flow_intra_inter0 is None else self.flow_intra_inter0.ravel().tolist()
        flow013_1 = [] if self.flow_intra_inter1 is None else self.flow_intra_inter1.ravel().tolist()
        flow013_3 = [] if self.flow_intra_inter3 is None else self.flow_intra_inter3.ravel().tolist()

        flow01_0 = [] if self.flow_intra0 is None else self.flow_intra0.ravel().tolist()
        flow01_1 = [] if self.flow_intra1 is None else self.flow_intra1.ravel().tolist()
        # import pdb; pdb.set_trace()
        features.append([it for it in flow013_0])
        features.append([it for it in flow013_1])
        features.append([it for it in flow013_3])
        features.append([it for it in flow01_0])
        features.append([it for it in flow01_1])
        return features
        
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
        self.least_square_conf = None
        self.dataset = dataset.lower()
        self.dataset_seq = data_seq
        self.annotations = get_kitti_ground_truth(input_path, data_seq)

        if self.dataset.lower() == 'kitti':
            self.camera_images = get_kitti_image_files(input_path, data_seq, num_cams)
            self.num_imgs  = len(self.camera_images[0])
            self.num_cams = len(self.camera_images)
        elif self.dataset.lower() == 'kite':
            self.camera_images = get_kite_image_files(input_path, data_seq, num_cams)
            self.num_imgs  = len(self.camera_images)
            self.num_cams = 4
        else:
            raise ValueError('Unsupported dataset')

        mtx, dist, rot, trans = load_camera_calib(dataset, calib_file, num_cams)
        # From nav_calib.cfg
        # Camera trans/rot prev camera to camera. Camera 0 is the first in the chain.
        # IMU trans/rot is imu to camera.
        # Convert the chained calibration cam0 -> cam1 -> cam2 -> cam3
        # to two stereo pair calibration: cam0 -> cam1, cam2 -> cam3
        rot_01, trans_01 = rot[1], trans[1]
        rot_10, trans_10 = invert_RT(rot[1], trans[1])

        rot_23, trans_23 = rot[3], trans[3]
        rot_32, trans_32 = invert_RT(rot[3], trans[3])
        
        rot[0], trans[0] = rot_01, trans_01
        rot[1], trans[1] = rot_10, trans_10
        rot[2], trans[2] = rot_23, trans_23
        rot[3], trans[3] = rot_32, trans_32

        self.ego_R = cv2.Rodrigues(np.eye(3))[0]
        self.ego_t = np.zeros([3, 1])

        self.pose_R = cv2.Rodrigues(np.eye(3))[0]
        self.pose_t = np.zeros([3, 1])
        # Compute the fundamental matrix
        for left in range(self.num_cams):
            right = self.STEREOCONFG[left]
            F_mtx = fundamental_matrix(rot[left], trans[left], mtx[left], mtx[right])
            self.navcams.append(navcam(left, self.STEREOCONFG[left], F_mtx, mtx[left], dist[left], rot[left], trans[left], self.num_features))
        # Set each camera's stereo config
        for left in range(self.num_cams):
            self.navcams[left].set_stereo_pair(self.navcams[self.STEREOCONFG[left]])

    def write_header_to_json(self, file_name):
        # if not os.path.exists(file_name):
        #     raise AssertionError(file_name + ' does not exit')
        calib_data = {}
        for i in range(self.num_cams):
            cur_cam = self.navcams[i]
            cam_data = {}
            cam_data['camera_matrix'] = cur_cam.calib_K.ravel().tolist()
            cam_data['camera_rotation'] = cur_cam.calib_R.ravel().tolist()
            cam_data['camera_translation'] = cur_cam.calib_t.ravel().tolist()
            cam_data['camera_features'] = []

            calib_data['camera'+str(i)] = cam_data
            with open(file_name, 'w') as outfile:
                json.dump(calib_data, outfile, sort_keys=True, indent=4)

    def update_features_to_json(self, outfile, img_idx):        
        with open(outfile, 'r') as f:
            try:
                json_data = json.load(f)
            except:
                import pdb; pdb.set_trace()

        for i in range(self.num_cams):
            cur_cam = self.navcams[i]
            cam_data = {}
            cam_feats = cur_cam.searlize_features_for_json() 
            json_data['camera'+str(i)]['camera_features'].append(cam_feats)

        with open(outfile, 'w') as f:
            json.dump(json_data, f, sort_keys=True, indent=4)

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

    def read_one_image(self, img_idx):
        if self.dataset == 'kitti':
            return read_kitti_image(self.camera_images, self.num_cams, img_idx)
        elif self.dataset == 'kite':
            return read_kite_image(self.camera_images, self.num_cams, img_idx) 
        else:
            raise ValueError('read_one_image(): unsupported dataset')

    def local_ego_motion_solver(self, cam_list=[0]):
        for c in cam_list:
            if self.navcams[c].flow_intra_inter0 is None:
                continue
            if len(self.navcams[c].flow_intra_inter0) > 0:
                res = self.navcams[c].local_ego_motion_solver()

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
            cur_cam = self.navcams[c]
            camera_matrix = cur_cam.calib_K
            stereo_matrix = cur_cam.stereo_pair_cam.calib_K
            stereo_rotation = cur_cam.calib_R2
            stereo_translation = cur_cam.calib_t

            if n_obj_013 > 0:      
                points_013 = x0[x0_offset: x0_offset + 3 * n_obj_013].reshape(-1, 3)
                flow013_0  = y_meas[y_offset: y_offset+n_obj_013]
                flow013_1  = y_meas[y_offset+ n_obj_013: y_offset + 2 * n_obj_013]
                flow013_3  = y_meas[y_offset + 2 * n_obj_013: y_offset + 3 * n_obj_013]

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

                flow01_errs = np.vstack((flow0_err, flow1_err))
        
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

        est_R = None
        est_t = None
        # Estimate the initial egomotion by using a single camera 
        mono_R0, mono_t0 = self.navcams[0].mono_ego_motion_estimation()
        mono_R1, mono_t1 = self.navcams[1].mono_ego_motion_estimation()

        if DATASET == 'kite':
            mono_R2, mono_t2 = self.navcams[2].mono_ego_motion_estimation()
            mono_R3, mono_t3 = self.navcams[3].mono_ego_motion_estimation()

        mono_R, mono_t =  mono_R0, mono_t0
        if mono_R is None:
            mono_R, mono_t =  mono_R1, mono_t1

        # import pdb ; pdb.set_trace()

        # if mono_R:
        est_R = mono_R
        est_t = self.prev_scale * mono_t if self.prev_scale is not None else mono_t

        if est is not None:
            est_R = est[0]
            est_t = est[1]

        num_cams = len(cam_list)
        cam_obs = np.zeros([num_cams, 2], dtype=np.int)
        y_meas = None
        x0 = np.vstack([est_R, est_t])

        for k in range(num_cams):
            c = cam_list[k]
            if self.navcams[c].flow_intra_inter0 is None:
                continue
            cur_cam = self.navcams[c]
            n_obs_i = cur_cam.flow_intra_inter0.shape[0]
            n_obs_j = cur_cam.flow_intra0.shape[0]

            if n_obs_i > 0:
                flow0 = cur_cam.flow_intra_inter0
                flow1 = cur_cam.flow_intra_inter1
                flow3 = cur_cam.flow_intra_inter3

                points013, terr013_0, terr013_1 = triangulate_3d_points(flow0, flow3, 
                                           cur_cam.calib_K, cur_cam.stereo_pair_cam.calib_K, 
                                           cur_cam.calib_R, cur_cam.calib_t)

                flow013_z = np.vstack([flow0, flow1, flow3])
                y_meas = flow013_z if y_meas is None else np.vstack([y_meas, flow013_z])

                x0 = np.vstack([x0, points013.ravel().reshape(-1, 1)])

            if n_obs_j > 0:
                flow0 = cur_cam.flow_intra0
                flow1 = cur_cam.flow_intra1
                points01, terr01_0, terr01_1 = triangulate_3d_points(flow0, flow1, 
                                               cur_cam.calib_K, cur_cam.calib_K,
                                               cv2.Rodrigues(est_R)[0], est_t)

                x0 = np.vstack([x0, points01.ravel().reshape(-1, 1)])
                flow01_z = np.vstack([flow0, flow1])
                y_meas = flow01_z if y_meas is None else np.vstack([y_meas, flow01_z])

            cam_obs[k][0] = n_obs_i
            cam_obs[k][1] = n_obs_j

        if y_meas is None or y_meas.shape[0] < 9:
            R, t = self.update_global_camera_pose_egomotion(cv2.Rodrigues(est_R)[0], est_t.reshape(3,))
            return R, t

        x0 = x0.flatten()

        sparse_A = global_bundle_adjustment_sparsity_opt(cam_obs, n_cams=num_cams) 

        ls_pars = dict(jac_sparsity=sparse_A,
                    max_nfev=5, 
                    verbose=0,
                    x_scale='jac',
                    jac='2-point',
                    ftol=0.01, 
                    xtol=0.01,
                    gtol=0.01,
                    method='trf')

        t0 = datetime.now()
        res = None
        # err0 = self.global_fun(x0, cam_obs, y_meas, cam_list)
        try:
            res = least_squares(self.global_fun, x0, args=(cam_obs, y_meas, cam_list), **ls_pars)
        except:
            import pdb; pdb.set_trace()
        if res is None:
            return self.ego_R, self.ego_t

        t1 = datetime.now()
        ego_elapsed = t1 - t0

        err1 = self.global_fun(res.x, cam_obs, y_meas, cam_list)
        err_level = norm(err1)

        R = cv2.Rodrigues(res.x[0:3])[0]
        t = res.x[3:6]
        if self.least_square_conf is None:
            self.least_square_conf = err_level
        avg_least_square_conf = self.least_square_conf / (self.img_idx)


        print('gba:',self.navcams[0].img_idx, ego_elapsed.microseconds / 1000.0, 
              'est_rot', res.x[0:3], 'est_tras', res.x[3:6], 'conf', norm(err1), avg_least_square_conf)

        if err_level > 5 * avg_least_square_conf:
            return self.pose_R, self.pose_t
        else:
            self.least_square_conf += err_level

        avg_least_square_conf = self.least_square_conf / self.img_idx

        pose_R, pose_t = self.update_global_camera_pose_egomotion(R, t)
        return pose_R, pose_t 
        


def _main(args):
    input_path = os.path.expanduser(args.images_path)
    output_path = os.path.expanduser(args.output_path)
    calib_file = os.path.expanduser(args.calib_path)
    json_enabled = args.enable_json
    num_features = args.num_features
    ransac_enabled = args.enable_ransac
    num_cam = args.num_cameras
    dataset = args.dataset_type.lower()
    seq = ("%02d" % (args.seq_num))  if (dataset == 'kitti') else str(args.seq_num)

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if num_features < 64:
        num_features = 64
    if dataset == 'kite' and not os.path.exists(calib_file):
        raise Exception, 'Kite data but no valid calib fie'
    elif dataset == 'kitti':
        calib_file = get_kitti_calib_path(input_path, seq)
    
    if not os.path.exists(calib_file):
        raise ValueError, 'Unable to find  valid calib fie'

    em_pars = dict(input_path=input_path, 
                   data_seq=seq, 
                   calib_file=calib_file, 
                   num_features=num_features, 
                   dataset=dataset, 
                   ransac=ransac_enabled, 
                   json_output=json_enabled)

    kv = EgoMotion(**em_pars)

    traj = np.zeros((1000, 1000, 3), dtype=np.uint8)
    kv.write_header_to_json('/tmp/test2.json')

    # import pdb; pdb.set_trace()
    for img_id in range(kv.num_imgs):
        camera_images = kv.read_one_image(img_id)

        kv.upload_images(camera_images)
        kv.update_keypoints()
        kv.update_sparse_flow()
        kv.filter_nav_keypoints(debug=False)
        
        if dataset == 'kitti':
            abs_scale = kv.load_kitti_gt(img_id)

        global_tr = np.zeros([3, 1])
        stereo_tr = np.zeros([3, 1])
        kv.navcams[0].local_bundle_adjustment(True)
        stereo_rot, stereo_tr = kv.navcams[0].get_stereo_camera_pose()
        global_rot, global_tr = kv.global_ego_motion_solver(img_id, cam_list=CAMERA_LIST)
        # kv.update_features_to_json('/tmp/test.json', img_id)

        if img_id >= 1:
            x, y, z = global_tr[0], global_tr[1], global_tr[2]
            x1, y1, z1 = stereo_tr[0], stereo_tr[1], stereo_tr[2]
            # import pdb ; pdb.set_trace()
        else:
            x, y, z = 0., 0., 0.
            x1, y1, z1 = 0., 0., 0.


        print('===================')
        print('goundt', kv.trueX, kv.trueZ)
        print('global', x, z)
        print('stereo', x1, z1)
        print('===================')

        draw_ofs_x = 50
        draw_ofs_y = 500

        draw_x0, draw_y0 = int(x)+draw_ofs_x, int(z)+draw_ofs_y    
        draw_x1, draw_y1 = int(x1)+draw_ofs_x, int(z1)+draw_ofs_y
        true_x, true_y = int(kv.trueX)+draw_ofs_x, int(kv.trueZ)+draw_ofs_y

        cv2.circle(traj, (draw_x0, draw_y0), 1, (255, 0,0), 1)
        cv2.circle(traj, (draw_x1, draw_y1), 1, (0,255,0), 1)
        cv2.circle(traj, (true_x,true_y), 1, (255,255,255), 2)
        cv2.rectangle(traj, (10, 20), (600, 60), (0,0,0), -1)
        text = "Img:%3d, Coordinates: x=%.2fm y=%.2fm z=%.2fm"%(img_id, x1, y1, z1)
        cv2.putText(traj, text, (20,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 1, 8)

        img_bgr = []
        for i in range(len(CAMERA_LIST)):
            img = cv2.resize(cv2.cvtColor(camera_images[i], cv2.COLOR_GRAY2BGR), (320, 240))
            img_bgr.append(img)
        img_ = concat_images_list(img_bgr)
      
        cv2.imshow('Navigation cameras', img_)
        cv2.imshow('Trajectory' + seq, traj)
        cv2.waitKey(1)
    traj_name = 'seq_' + seq + '_' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.png'
    cv2.imwrite(os.path.join(output_path, traj_name), traj)

    

if __name__ == '__main__':
    _main(parser.parse_args())
