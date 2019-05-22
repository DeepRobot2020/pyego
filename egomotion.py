import cv2
import glob, pdb, math, json
import warnings
import os, io, libconf, copy
import csv
import pandas as pd
import numpy as np
from scipy.optimize import least_squares 

import time
import argparse
from cameras import navcam
from acc_flt import AccFilter

from utils import *
from cfg import *
from kiteJsonLog import kiteJsonLog

from datetime import datetime
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

parser = argparse.ArgumentParser(
    description='Compute camera relative pose on input stereo or quad images')


parser.add_argument(
    '-img',
    '--images_path',
    help='path to directory of input images',
    default=INPUT_IMAGE_PATH)

parser.add_argument(
    '-undist',
    '--undistortion_required',
    help='Whether undistortion required or not',
    default=True)

parser.add_argument(
    '-calib',
    '--calib_path',
    help='path to directory of calibriation file (Kite)',
    default=INPUT_CALIB_PATH)


parser.add_argument(
    '-feats',
    '--num_features',
    type=int,
    help='Max number of features',
    default= MAX_NUM_KEYPOINTS)

parser.add_argument(
    '-num_cam',
    '--num_cameras',
    type=int,
    help='Max number of cameras used for VO',
    default=4)

parser.add_argument(
    '-json_init',
    '--json_init_path',
    help='Json file including the initial egomotion',
    type=str,
    default=VO_INIT_JSON)

parser.add_argument(
    '-linear_seed',
    '--linear_seed_enabled',
    help='Whether to use the linear motion as seed',
    type=int,
    default=0)

parser.add_argument(
    '-output_path',
    '--output_results_path',
    help='Whether to write the motion log',
    type=str,
    default=None)

parser.add_argument(
    '-json_feature_path',
    '--json_feature_path',
    help='use image features from json files',
    type=str,
    default=None)

parser.add_argument(
    '-temportal_stereo',
    '--temportal_stereo_enable',
    help='Whether to use the temporal only stereo matching',
    type=bool,
    default=USE_01_FEATURE)

parser.add_argument(
    '-points_param',
    '--points_param_enable',
    help='Whether to put points into the parameter list',
    type=bool,
    default=False)


class EgoMotion:
    """EgoMotion object"""
    def __init__(self, image_path = None,
                       calib_file = None, 
                       num_cams = 4, 
                       num_features = 64, 
                       init_json_path = None, 
                       use_linear_seed = False, 
                       distorted_image = True,
                       output_path = None,
                       stereo_rectify = False,
                       temportal_stereo = True, 
                       points_params = False):

        self.ips_data = False
        self.num_features = num_features
        self.STEREOCONFG  = [1, 0, 3, 2]
        self.camera_images = get_image_files(image_path)
        self.distorted_image = distorted_image
        self.num_imgs  = len(self.camera_images)
        self.num_cams = num_cams
        self.stereo_rectify = stereo_rectify
        self.json_log_list = None
        self.use_linear_seed = use_linear_seed
        self.use_temportal_stereo = temportal_stereo
        self.points_params = True
        # self.kl = kiteJsonLog(VO_KITE_LOG)

        if init_json_path is not None:
            self.json_log_list = load_json_to_list(init_json_path)

        self.json_motion_log = None
        if output_path is not None:
            if not os.path.exists(output_path):
                os.mkdir(output_path)
            log_file_name = os.path.join(output_path, 'motion_debug2.json')
            self.json_motion_log = open(log_file_name, 'w')

        self.rotation_cam0_to_cam2 = None
        self.translation_cam0_to_cam2 = None

        self.rotation_cam2_to_cam0 = None
        self.translation_cam2_to_cam0 = None


        self.navcams = []
        self.create_cameras(calib_file, num_cams)

        # Kitti: images of each camera are stored s
        # Kite: images have been concatenated into one image
        self.img_idx = -1

        self.history_egomotion = np.zeros((1,6))

        self.json_log = None
        self.csv_writer = None

        self.prev_timestamp = 0
        self.curr_timestamp = 0
        self.ned_vel_err = np.zeros([3,1])
        self.vel_covar = np.zeros([6,1])

        self.prev_rot = np.identity(3)
        self.prev_trans = np.zeros((3,1))

        self.csv_log_file = None
        self.json_log_file = None


        self.motion_rotation = cv2.Rodrigues(np.eye(3))[0]
        self.motion_translation = np.zeros([3, 1])

        self.position = np.zeros((3, 1))
        self.orientation = np.eye(3)

        self.trueX, self.trueY, self.trueZ = 0, 0, 0
        self.acc_flt = AccFilter(9, 20, 300)

    def get_failed_kite_list(self):
        ahrs_rate_xyz = self.kl.get_ahrs_rate()
        vo_angular_xyz = self.kl.get_body_ang_vel()

        vo_ned_vel_cpp = self.kl.vo_dict['ned_lin_vel']
        gps_ned_vel = self.kl.vo_dict['gps_ned_vel']

        ned_x_err = np.abs(vo_ned_vel_cpp[:,0] - gps_ned_vel[:,0]) > 5
        ned_y_err = np.abs(vo_ned_vel_cpp[:,1] - gps_ned_vel[:,1]) > 5
        ned_z_err = np.abs(vo_ned_vel_cpp[:,2] - gps_ned_vel[:,2])
        linear_mask = ned_x_err | ned_y_err
        
        ang_mask_x = np.abs(ahrs_rate_xyz[:,0] - vo_angular_xyz[:,0]) > 0.1
        ang_mask_y = np.abs(ahrs_rate_xyz[:,1] - vo_angular_xyz[:,1]) > 0.1
        ang_mask_z = np.abs(ahrs_rate_xyz[:,2] - vo_angular_xyz[:,2]) > 0.1

        angular_mask = ang_mask_x | ang_mask_y | ang_mask_z
        indexs = np.array(range(1, 1 + len(angular_mask)))[linear_mask]
        # import pdb; pdb.set_trace()
        return indexs
        
    def create_cameras(self, calib_path, num_cams):
        # Load the calib file
        mtx, dist, rot, trans, imu_rot, imu_trans = load_camera_calib('kite', calib_path, num_cams)
        
        self.rotation_body_to_cam0, self.translation_body_to_cam0 = compute_body_to_camera0_transformation(IMU_TO_BODY_ROT, imu_rot[0], imu_trans[0])


        self.rotation_cam0_to_body, self.translation_cam0_to_body = invertRt(self.rotation_body_to_cam0, self.translation_body_to_cam0)

        # From nav_calib.cfg
        # Camera trans/rot prev camera to camera. Camera 0 is the first in the chain.
        # IMU trans/rot is imu to camera.
        # Convert the chained calibration cam0 -> cam1 -> cam2 -> cam3
        # to two stereo pair calibration: cam0 -> cam1, cam2 -> cam3
        rot_01, trans_01 = rot[1], trans[1]
        # import pdb; pdb.set_trace()
        rot_10, trans_10 = invertRt(rot[1], trans[1])
    
        # For Kite there are two stereo pair (0, 1) and (2, 3), 
        # where camera 0 and camera 2 are their reference cameras, we need know the 
        # translation between those two stereo pairs
        rot_23, trans_23 = rot[3], trans[3]
        rot_32, trans_32 = invertRt(rot[3], trans[3])
    

        rot_12, trans_12 = rot[2], trans[2]
        rot_02 = np.dot(rot_12, rot_01) 
        trans_02 = np.dot(rot_12, trans_01) + trans_12

        rot_20, trans_20 = invertRt(rot_02,trans_02 ) 
        self.rotation_cam0_to_cam2 = rot_02
        self.translation_cam0_to_cam2 = trans_02
        self.rotation_cam2_to_cam0 = rot_20
        self.translation_cam2_to_cam0 = trans_20


        if self.stereo_rectify:
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
        else:
            self.stereo_rectified_P = [None, None, None, None]
            self.stereo_rectified_R = [None, None, None, None]

        rot[0], trans[0] = rot_01, trans_01
        rot[1], trans[1] = rot_10, trans_10
        rot[2], trans[2] = rot_23, trans_23
        rot[3], trans[3] = rot_32, trans_32
        # Create cameras
        for left in range(self.num_cams):
            right = self.STEREOCONFG[left]
            self.navcams.append(navcam(left, self.STEREOCONFG[left], mtx[left], dist[left], rot[left], trans[left], 
                self.num_features, self.stereo_rectified_P, self.stereo_rectified_R))

        # Set each camera's stereo config
        for left in range(self.num_cams):
            self.navcams[left].set_stereo_pair(self.navcams[self.STEREOCONFG[left]])

    def close(self):
        if self.json_motion_log is not None:
            self.json_motion_log.close()
            self.json_motion_log = None
        if self.csv_log_file is not None:
            self.csv_log_file.close()
            self.csv_log_file = None

    def vis3(self, cam_id, img_idx, flow0, flow1, flow3, rot_params = None, trans_params = None, points3d_stereo = None, points3d_temporal = None):
        

        curr_img = self.navcams[cam_id].curr_img
        prev_img = self.navcams[cam_id].prev_img
        curr_stereo_img = self.navcams[cam_id].curr_stereo_img
        cam = self.navcams[cam_id]

        if points3d_stereo is None:
            try:
                points03, _, _ = triangulate_3d_points(flow0, flow3, cam.calib_K, cam.stereo_pair_cam.calib_K, cam.calib_R, cam.calib_t)
            except:
                pass
        else:
            points03 = points3d_stereo

        img1 = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(curr_stereo_img, cv2.COLOR_GRAY2BGR)

        # import pdb; pdb.set_trace()
        for pt1, pt3, wp in zip(flow0, flow3, points03):
            x1, y1 = (int(pt1[0]), int(pt1[1]))
            x3, y3 = (int(pt3[0]), int(pt3[1]))
            color = tuple(np.random.randint(0,255,3).tolist())
            depth = "{:.1f}".format(wp[2])
            cv2.putText(img1, depth, (x1 + 10, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, lineType=cv2.LINE_AA) 
            cv2.putText(img2, depth, (x3 + 10, y3 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, lineType=cv2.LINE_AA) 
            cv2.circle(img1,(x1, y1), 6, color,2)
            cv2.circle(img2,(x3, y3), 6, color,2)

        img_stereo = concat_images(img1, img2)

        if points3d_temporal is None:
            try:
                points01, _, _ = triangulate_3d_points(flow0, flow1, cam.calib_K, cam.calib_K, rot_params, trans_params)
            except:
                points01 = points03
        else:
            points01 = points3d_temporal

        img1 = cv2.cvtColor(curr_img, cv2.COLOR_GRAY2BGR)
        img2 = cv2.cvtColor(prev_img, cv2.COLOR_GRAY2BGR)

        # import pdb; pdb.set_trace()
        for pt1, pt3, wp in zip(flow0, flow1, points01):
            x0, y0 = (int(pt1[0]), int(pt1[1]))
            x1, y1 = (int(pt3[0]), int(pt3[1]))
            color = tuple(np.random.randint(0,255,3).tolist())
            depth = "{:.1f}".format(wp[2])
            cv2.putText(img1, depth, (x1 + 10, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, lineType=cv2.LINE_AA) 
            cv2.putText(img2, depth, (x3 + 10, y3 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, lineType=cv2.LINE_AA) 
            cv2.circle(img1,(x0, y0), 6, color,2)
            cv2.circle(img2,(x1, y1), 6, color,2)

        img_tempo = concat_images(img1, img2)

        img_out = stack_images(img_stereo, img_tempo)


        if img_out is not None:
            out_img_name = os.path.join('/tmp/pyego', 'cam' + str(cam_id) + '_' + str(img_idx)+'.jpg')
            cv2.imwrite(out_img_name, img_out)

    def keypointsDetection(self, img_id):
        for i in range(self.num_cams):
            if i in CAMERA_LIST:
                self.navcams[i].keypoint_detection()

    def updateInterOptflow(self):
        for i in range(self.num_cams):
            if i in CAMERA_LIST:
                self.navcams[i].inter_sparse_optflow()

    def updateIntraOptflow(self):
        if self.img_idx > 0:
            for i in range(self.num_cams):
                if i in CAMERA_LIST:
                    self.navcams[i].intra_sparse_optflow()

    def updateCircularOptflow(self):
        if self.img_idx > 0:
            for i in range(self.num_cams):
                if i in CAMERA_LIST:
                    ret = self.navcams[i].circular_optflow()
            

    def outliersRejection(self, debug=False):
        for c in range(self.num_cams):
            if c in CAMERA_LIST:
                self.navcams[c].outliersRejection(debug=debug)

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

    def uploadImages(self, imgs_x4):
        self.img_idx += 1
        for c in range(self.num_cams):        
            self.navcams[c].update_image(imgs_x4)        

    def getOneImage(self, img_idx):
        img = read_kite_image(self.camera_images, self.num_cams, KITE_VIDEO_FORMAT, img_idx)
        timestamp = -1

        if img is not None and self.distorted_image:
            for i in range(4):
                img[i] = self.navcams[i].undistortImage(img[i])
        try:
            json_line = self.json_log_list[img_idx]
            timestamp = json_line['arm_timestamp']
        except:
            pass
        return img, timestamp


    def local_ego_motion_solver(self, cam_list=[0]):
        for c in cam_list:
            if self.navcams[c].flow_intra_inter0 is None:
                continue
            if len(self.navcams[c].flow_intra_inter0) > 0:
                res = self.navcams[c].local_ego_motion_solver()

    def update_global_camera_pose_egomotion(self, R, t):
        if R.shape != (3, 3):
            R = cv2.Rodrigues(R.reshape(3,1))[0]

        self.motion_rotation = R
        self.motion_translation = t

        if norm(t) > 10:
            return self.orientation, self.position
        # C(k) = C(k-1) * T(k, k-1)
        try:
            self.position = self.position + self.orientation.dot(t) 
            self.orientation = self.orientation.dot(R)
        except:
            # import pdb; pdb.set_trace()
            pass
        return self.orientation, self.position

    def get_egomotion(self):
        return self.motion_rotation, self.motion_translation

    def get_global_camera_pose(self):
        return self.orientation, self.position
    
    def reprojection_error(self, x0, cam_obs, z_global, cam_list, points_3d = None):
        if points_3d is None:
            x0_offset = 6
            points = x0
        else:
            x0_offset = 0
            points = points_3d

        y_offset = 0
        cost_err = None
        rotation_list, translation_list = self.transformMotionToCamerai(x0[0:3], x0[3:6])
        for index, c in enumerate(cam_list):
            egomotion_rotation = rotation_list[c]
            egomotion_translation = translation_list[c]

            rot_vecs = cv2.Rodrigues(egomotion_rotation)[0]
            trans_vecs = egomotion_translation

            n_obj_013, n_obj_01 = cam_obs[index]

            cur_cam = self.navcams[c]
            camera_matrix = cur_cam.calib_K
            stereo_matrix = cur_cam.stereo_pair_cam.calib_K
            stereo_rotation = cur_cam.calib_R2
            stereo_translation = cur_cam.calib_t

            if n_obj_013 > 0:      
                points_013 = points[x0_offset: x0_offset + 3 * n_obj_013].reshape(-1, 3)
                flow013_0  = z_global[y_offset: y_offset + n_obj_013]
                flow013_1  = z_global[y_offset+ n_obj_013: y_offset + 2 * n_obj_013]
                flow013_3  = z_global[y_offset + 2 * n_obj_013: y_offset + 3 * n_obj_013]

                points_3x1 = points_013
                observations_2x1 = flow013_1
                camera_matrix_3x3 = camera_matrix
                rotation_vector_3x1 = rot_vecs
                translation_vector_3x1 = trans_vecs

                flow0_err = reprojection_error(points_013, flow013_0, camera_matrix)
                flow1_err = reprojection_error(points_013, flow013_1, camera_matrix, rot_vecs, trans_vecs)
                flow3_err = reprojection_error(points_013, flow013_3, stereo_matrix, stereo_rotation, stereo_translation)
                
                if self.points_params:
                    flow013_errs = np.vstack((flow0_err, flow1_err, flow3_err))
                else:
                    flow013_errs = flow1_err

                x0_offset += 3 * n_obj_013
                y_offset += 3 * n_obj_013

                if cost_err is None:
                    cost_err = flow013_errs
                else:
                    cost_err = np.vstack([cost_err, flow013_errs])

            if n_obj_01 > 0:
                points_01  = points[x0_offset: x0_offset+3*n_obj_01].reshape(-1, 3)
                flow01_0  = z_global[y_offset: y_offset + n_obj_01]
                flow01_1  = z_global[y_offset + n_obj_01: y_offset + 2 * n_obj_01]
                
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

    def estimate_mono_motion(self, cam_list):
        mono_rotation = {}
        mono_translation = {}
        masks = {}
        for cam_idx in cam_list:
            try:
                rot, trans, mask = self.navcams[cam_idx].five_point_algo()
                mono_rotation[cam_idx] = rot
                mono_translation[cam_idx] = trans
                masks[cam_idx] = mask
            except ValueError:
                print('estimate R, t from camera {} failed'.format(cam_idx))
                mono_rotation[cam_idx] = None
                mono_translation[cam_idx] = None
                masks[cam_idx] = None

        return mono_rotation, mono_translation, masks

    def get_init_motion_from_mono_estimation(self, cam_list):
        mono_rotation, mono_translation, masks = self.estimate_mono_motion(cam_list)
        for cam_idx in cam_list:
            rot, trans = mono_rotation[cam_idx], mono_translation[cam_idx]
            if all(v is not None for v in [rot, trans]):
                if cam_idx >= 2:
                    rot2, trans2 = transform_egomotion_from_frame_a_to_b(rot, trans, self.rotation_cam2_to_cam0, self.translation_cam2_to_cam0)
                    return rot2, trans2, cam_idx, mono_rotation, mono_translation
                return rot, trans, cam_idx, mono_rotation, mono_translation, masks
        return None, None, -1, None, None, None

    def update_pose_by_prev_motion(self):
        return self.update_global_camera_pose_egomotion(self.prev_rot, self.prev_trans)

    def nonUniqueSolutionDetetion(self, x,  cam_obs, z_global, cam_list, scale=2.5):
        x_scaled = x.copy()
        err0 = self.reprojection_error(x, cam_obs, z_global, cam_list)
        x_scaled[3:] = scale * x_scaled[3:]
        err1 = self.reprojection_error(x_scaled, cam_obs, z_global, cam_list)
        diff = np.absolute(err0 - err1)
        diff_norm = norm(diff)
    
    def transformMotionToCamerai(self, rotation_camera0, translation_camera0):
        rotation_list = []
        translation_list = []

        if rotation_camera0.shape != (3, 3):
            rotation_camera0 = cv2.Rodrigues(rotation_camera0)[0]

        translation_camera0 = translation_camera0.reshape(3, 1)


        rotation_camera1, translation_camera1 = transform_egomotion_from_frame_a_to_b(rotation_camera0, 
                                                                                      translation_camera0, 
                                                                                      self.navcams[0].calib_R, 
                                                                                      self.navcams[0].calib_t)
        

        rotation_list.append(rotation_camera0)
        translation_list.append(translation_camera0)

        rotation_list.append(rotation_camera1)
        translation_list.append(translation_camera1)

        if self.num_cams == 4:
            rotation_camera2, translation_camera2 = transform_egomotion_from_frame_a_to_b(rotation_camera0, 
                                                                                                    translation_camera0, 
                                                                                                    self.rotation_cam0_to_cam2, 
                                                                                                    self.translation_cam0_to_cam2)

            rotation_camera3, translation_camera3 = transform_egomotion_from_frame_a_to_b(rotation_camera2, 
                                                                                                    translation_camera2, 
                                                                                                    self.navcams[2].calib_R, 
                                                                                                    self.navcams[2].calib_t)
            # import pdb; pdb.set_trace()

            rotation_list.append(rotation_camera2)
            translation_list.append(translation_camera2)

            rotation_list.append(rotation_camera3)
            translation_list.append(translation_camera3)
        return rotation_list,  translation_list

    def orientatonToNed(self, img_idx):
        if self.ips_data:
            try:
                js = self.json_log_list[img_idx]
                ips_orient_q_wf = js["ips_orient_q_wf"]
                Rmtx = Q2R(ips_orient_q_wf)  
                return Rmtx
            except:
                print('Read ips_orient_q_wf failed')
                import pdb; pdb.set_trace()
                return np.eye(3)
        else:
            try:
                js = self.json_log_list[img_idx]
            except:
                return np.eye(3)
            try:
                acs_est_orentation = js["acs_est_orentation"]
                return cv2.Rodrigues(acs_est_orentation)[0]
            except:
                try:
                    ahrs_kf4_Qest = js["ahrs_kf4_Qest"]
                    return Q2R(ahrs_kf4_Qest)    
                except:
                    return np.eye(3)

    def gpsNedVelocity(self, img_idx):
        if self.ips_data:
            try:
                js = self.json_log_list[img_idx]
                ned_vel = js["ips_ned_vel"]
                return np.array(ned_vel)
            except:
                return None
        else:
            try:
                js = self.json_log_list[img_idx]
                gps_ned_vel = js["gps_ned_vel"]
                return np.array(gps_ned_vel)
            except:
                return None

    def gpsNedPosition(self, img_idx):
        if self.ips_data:
            try:
                js = self.json_log_list[img_idx]
                gps_ned_pos = js["ips_ned_pos"]
                return np.array(gps_ned_pos)
            except:
                return None
        else:
            try:
                js = self.json_log_list[img_idx]
                gps_ned_pos = js["gps_ned_pos"]
                return np.array(gps_ned_pos)
            except:
                return None

    def getLoggedInitMotion(self, img_idx):
        try:
            json_line = self.json_log_list[img_idx]
            x0 = np.array(json_line['opt_motion']).reshape(-1,1)
            return x0
        except:
            return None


    def getMonoInitMotion(self, cam_list):
        rot0, trans0, camera_index, rot_list, trans_list, masks = self.get_init_motion_from_mono_estimation(cam_list)
        if camera_index == -1:
            print('error: no initial estimation from 5 point are avalaible')
            return None
        if rot0.shape == (3,3):
            rot0 = cv2.Rodrigues(rot0)[0]
        return np.vstack([rot0, trans0])

    def computeEgomotionRansac(self, cam_id = 0, motion_params = None):
        ''' Compute the motion by ransac
        '''    
        left_cam = self.navcams[cam_id]
        
        MIN_PTS = 5

        if left_cam.flow_intra_inter0 is None or left_cam.flow_intra_inter0.shape[0] < MIN_PTS:
            return
        
        points013, _, _ = triangulate_3d_points(left_cam.flow_intra_inter0, left_cam.flow_intra_inter3, 
                                                left_cam.calib_K, left_cam.stereo_pair_cam.calib_K, 
                                                left_cam.calib_R, left_cam.calib_t)


        MAX_ITERS = 10
        N = len(points013)
        K = MIN_PTS
        

        # RANSAC
        itr = 0
        cam_list = [cam_id]
        best_err_opt = 1000.0
        best_mask = np.array([True] * N)
        best_inliers = -1
        best_res = None
        while itr < MAX_ITERS:
            itr += 1
            ransac_mask = np.array([True] * K + [False] * (N-K))
            np.random.shuffle(ransac_mask)

            res, _, _ = self.leastSquaresOpt([cam_id], motion_params, [ransac_mask])
            if res is None:
                continue        
        
            rot_params = res.x[0:3].reshape(-1, 1)
            trans_params = res.x[3:6].reshape(-1, 1)

            pts1 = points013.ravel().reshape(-1, 1)
            x1 = np.vstack([res.x[0:6].reshape(-1,1), pts1])

            z1 = np.vstack([left_cam.flow_intra_inter0, left_cam.flow_intra_inter1, left_cam.flow_intra_inter3])
            cam_obs = np.zeros([1, 2], dtype=np.int)
            cam_obs[0][0] = N 

            if self.points_params is False:
                points_3d = x1[6:]
            else:
                points_3d = None
                
            err1_opt = self.reprojection_error(x1[0:6], cam_obs, z1, cam_list, points_3d).reshape(-1, 2)
            err_2d = norm(err1_opt, axis=1)

            points01, _, _ = triangulate_3d_points(left_cam.flow_intra_inter0, left_cam.flow_intra_inter1, left_cam.calib_K, left_cam.calib_K, cv2.Rodrigues(rot_params)[0], trans_params)
                
            err_3d = norm(points01 - points013.reshape(-1,3), axis=1)

            mask_2d = err_2d < 0.5
            mask_3d = err_3d < 0.5

            ransac_mask = mask_2d

            if np.sum(ransac_mask) > best_inliers:
                best_mask = ransac_mask
                best_inliers = np.sum(ransac_mask)
                print('=================== n_inliers: {} vs {}'.format(np.sum(ransac_mask), ransac_mask.shape[0] ))
                # import pdb; pdb.set_trace()
        return best_mask

        


    def leastSquaresOpt(self, cam_list, motion_params_6dof, mask_list = None):

        rotation_list, translation_list = self.transformMotionToCamerai(motion_params_6dof[0:3], motion_params_6dof[3:6])

        z_global = None
        x0 = motion_params_6dof.reshape(-1,1)
        num_cams = len(cam_list)
        cam_obs = np.zeros([num_cams, 2], dtype=np.int)

        for index, cam_id  in enumerate(cam_list):
            cur_cam = self.navcams[cam_id]
            if cur_cam.flow_intra_inter0 is None:
                continue
            
            n_obs_i = cur_cam.flow_intra_inter0.shape[0]

            try:
                mask = mask_list[index]
            except:
                mask = np.array([True] * n_obs_i)


            cur_rotation, cur_translation = rotation_list[cam_id], translation_list[cam_id]

            n_obs_i = np.sum(mask)

            if n_obs_i > 0:
                flow0 = cur_cam.flow_intra_inter0[mask==True]
                flow1 = cur_cam.flow_intra_inter1[mask==True]
                flow3 = cur_cam.flow_intra_inter3[mask==True]    


                points03, _, _ = triangulate_3d_points(flow0, flow3, 
                                           cur_cam.calib_K, 
                                           cur_cam.stereo_pair_cam.calib_K, 
                                           cur_cam.calib_R, 
                                           cur_cam.calib_t)

                points03 = points03.ravel().reshape(-1, 1)
                z_cur = np.vstack([flow0, flow1, flow3])
                z_global = z_cur if z_global is None else np.vstack([z_global, z_cur])

                x0 = np.vstack([x0, points03])

      
            if self.use_temportal_stereo:
                n_obs_j = cur_cam.flow_intra0.shape[0]
            else:
                n_obs_j = 0
            if n_obs_j > 0:
                flow0 = cur_cam.flow_intra0
                flow1 = cur_cam.flow_intra1
                try:
                    points01, _, _ = triangulate_3d_points(flow0, flow1, K0, K0, cur_rotation, cur_translation)
                except:
                    import pdb; pdb.set_trace()
                    print('triangulate_3d_points failed ' )
                    return self.update_pose_by_prev_motion()

                x0 = np.vstack([x0, points01.ravel().reshape(-1, 1)])
                z_cur = np.vstack([flow0, flow1])
                z_global = z_cur if z_global is None else np.vstack([z_global, z_cur])

            cam_obs[index][0] = n_obs_i
            cam_obs[index][1] = n_obs_j

        if z_global is None or z_global.shape[0] < 9:
            return None, None, None

        x0 = x0.flatten()

        try:
            if self.points_params is False:
                points_3d = x0[6:]
                params = x0[0:6]
                sparse_A = None
            else:
                points_3d = None
                params = x0
                sparse_A = compute_bundle_adjustment_sparsity(cam_obs, n_cams=num_cams) 


            err0 = self.reprojection_error(x0, cam_obs, z_global, cam_list, points_3d)
            res = least_squares(self.reprojection_error, params, args=(cam_obs, z_global, cam_list, points_3d), jac_sparsity=sparse_A, **LS_PARMS)
            err1 = self.reprojection_error(res.x, cam_obs, z_global, cam_list, points_3d)

        except:
            return None, None, None
        
        return res, err0, err1


    def computeEgomotion(self, img_idx=None, timestamp_ms = None, dt = None, cam_list=[0, 1]):
        x0 = self.getLoggedInitMotion(img_idx)

        if x0 is None or self.use_linear_seed == True:
            x0 = self.getMonoInitMotion(cam_list)
        
        if x0 is None:
            print('image_id {} timestam[ {}: computeEgomotion failed, unable to get initial motion', img_idx, timestamp_ms)
            return None

        
        t0 = datetime.now()

        res, err0, err1 = self.leastSquaresOpt(cam_list, x0)

        if res is None:
            opt_failed = True
            print("1st optimization failed")
            return self.update_pose_by_prev_motion()


        x0 = res.x[0:6].reshape(-1,1)
        rotation_list, translation_list = self.transformMotionToCamerai(x0[0:3], x0[3:6])
        masks = []
        for index, cam_id  in enumerate(cam_list):
            cur_cam = self.navcams[cam_id]

            if cur_cam.flow_intra_inter0 is None:
                continue
            cur_rotation, cur_translation = rotation_list[cam_id], translation_list[cam_id]
            
            z_cur = np.vstack([cur_cam.flow_intra_inter0, cur_cam.flow_intra_inter1, cur_cam.flow_intra_inter3])


            cam_obs_cur = np.zeros([1, 2], dtype=np.int)
            cam_obs_cur[0][0] = cur_cam.flow_intra_inter0.shape[0]


            if cur_cam.flow_intra_inter0.shape[0] <= 0:
                continue
            
            points_3d_03, points_3d_01 = cur_cam.final_3d(x0[0:3], x0[3:6])

            pt_err = norm(points_3d_03 - points_3d_01, axis=1)

            err_cur = self.reprojection_error(x0, cam_obs_cur, z_cur, [cam_id], points_3d_03).reshape(-1,2)
            err_cur_norm = norm(err_cur, axis=1)
            mask_2d = err_cur_norm < 1.0
            
            thresh_3d = 0.5
            while np.sum( pt_err < thresh_3d) < 10:
                thresh_3d *= 2.0
                if thresh_3d > 10.0:
                    break
 
            mask_3d = pt_err < thresh_3d
            masks.append(mask_2d)

            mask = mask_2d# & mask_3d

            flow0 = self.navcams[cam_id].flow_intra_inter0
            flow1 = self.navcams[cam_id].flow_intra_inter1
            flow3 = self.navcams[cam_id].flow_intra_inter3
        
            # self.vis3(cam_id, img_idx, flow0, flow1, flow3, res.x[0:3], res.x[3:6])
        
        # res, _, err2 = self.leastSquaresOpt(cam_list, x0, None)

        if res is None:
            opt_failed = True
            print("2nd optimization failed")
            return self.update_pose_by_prev_motion()


        t1 = datetime.now()
        computation_delay = t1 - t0
       
        velocity_ned, velocity_body = self.computeLinearVelocity(res.x, img_idx, dt)

        if self.points_params:
            covariance_ned, covariance_body = self.computeCovariance(res.jac.toarray(), img_idx, dt * dt )        
        else:
            covariance_ned, covariance_body = self.computeCovariance(res.jac, img_idx, dt * dt )

        gps_ned_vel = self.gpsNedVelocity(img_idx-1)

        valid_vel = True
        valid_vel = self.acc_flt.filter(velocity_body, covariance_body, self.curr_timestamp)
        if valid_vel:
            self.acc_flt.UpdateState(velocity_body, covariance_body, self.curr_timestamp)

        print('image_id: {} timestamp_ms: {} valid: {} err0: {} err1: {}'.format(img_idx, 
                                                                                 self.curr_timestamp, 
                                                                                 valid_vel, 
                                                                                 round(np.linalg.norm(err0), 4), 
                                                                                 round(np.linalg.norm(err1), 4)))

        try:
            velocity_ned_cpp = self.kl.vo_dict['ned_lin_vel'][img_idx-1]
            print('Est-CPP-Norm: {} N: {} E: {} D: {}'.format(round(np.linalg.norm(velocity_ned_cpp),4), 
                                                            round(velocity_ned_cpp[0], 4), 
                                                            round(velocity_ned_cpp[1], 4), 
                                                            round(velocity_ned_cpp[2], 4)))
        except:
            pass

        print(' Est-PY-Norm: {} N: {} E: {} D: {}'.format(round(np.linalg.norm(velocity_ned),4), 
                                                        round(velocity_ned[0], 4), 
                                                        round(velocity_ned[1], 4), 
                                                        round(velocity_ned[2], 4)))


        print('    GPS-Norm: {} N: {} E: {} D: {}'.format(round(np.linalg.norm(gps_ned_vel),4), 
                                                        round(gps_ned_vel[0], 4), 
                                                        round(gps_ned_vel[1], 4), 
                                                        round(gps_ned_vel[2], 4)))


        # import pdb; pdb.set_trace()
        # self.navcams[0].debug_inter_keypoints(self.navcams[0].flow_intra_inter0, self.navcams[0].flow_intra_inter3, '/tmp/pyego/')

        # Log results
        try:
            json_line = self.json_log_list[img_idx]
            json_line["body_lin_vel"] = velocity_body.tolist()
            json_line["ned_lin_vel"] = velocity_ned.tolist()
            json_line["gps_ned_vel"] = gps_ned_vel.tolist()

            # json_line["body_lin_vel_conf"] = covariance_body.tolist()
            # json_line["ned_lin_vel_conf"] = covariance_ned.tolist()
            json_line["status"] = int(not valid_vel)
            if self.json_motion_log:
                json.dump(json_line, self.json_motion_log, sort_keys=True)
                self.json_motion_log.write('\n')            
        except:
            print('Cannot log the VO results')
        print('=================================================\n\n')


    def computeCovariance(self, jacobian, img_idx, dtdt):
        if self.points_params:
            cov_cam = covarinace_svd(jacobian)
        else:
            cov_cam = covariance_mvg_A6_4(jacobian)

        aug_rot0 = np.zeros([6, 6])
        aug_rot0[0:3,0:3] = self.rotation_cam0_to_body
        aug_rot0[3:6,3:6] = self.rotation_cam0_to_body
        aug_rot1 = np.zeros([6, 6])
        orentation_to_ned = self.orientatonToNed(img_idx)
        aug_rot1[0:3,0:3] = orentation_to_ned
        aug_rot1[3:6,3:6] = orentation_to_ned
        cov_body = aug_rot0.dot(cov_cam).dot(aug_rot0.T)
        cov_ned = aug_rot1.dot(cov_body).dot(aug_rot1.T)
        cov_body_diag = cov_body.diagonal()
        cov_ned_diag = cov_ned.diagonal()
        lin_cov_body = cov_body_diag[3:6] / dtdt
        lin_cov_ned = cov_ned_diag[3:6] / dtdt
        lin_cov_body_norm = np.linalg.norm(lin_cov_body)
        lin_cov_ned_norm = np.linalg.norm(lin_cov_ned)
        # if lin_cov_body_norm > 10.0 or lin_cov_ned_norm > 10.0:
        #     lin_cov_body = 1.0
        #     lin_cov_ned = 1.0
        return lin_cov_body, lin_cov_ned

    def computeLinearVelocity(self, x_opt, img_idx, dt):
        rot_vec_body, trans_vec_body = transformVelocityFromAToB(x_opt[0:3], x_opt[3:6], 
                                                                 self.rotation_cam0_to_body, self.translation_cam0_to_body)
        orentation_to_ned = self.orientatonToNed(img_idx)

        body_line_vel = trans_vec_body / dt
        ned_lin_vel = orentation_to_ned.dot(body_line_vel)
        return ned_lin_vel.ravel(), body_line_vel.ravel()
    
    def updateTimestamp(self, timestamp):
        self.prev_timestamp = self.curr_timestamp
        self.curr_timestamp = timestamp
        dt = (timestamp - self.prev_timestamp) / 1e3
        if dt <= 0 or dt > 1.0:
            dt = 1.0
        return dt
   

def _main(args):
    input_path = os.path.expanduser(args.images_path)
    calib_file = os.path.expanduser(args.calib_path)

    json_init_path = os.path.expanduser(args.json_init_path)
    linear_seed_enabled = int(args.linear_seed_enabled)


    num_cam = args.num_cameras
    num_features = args.num_features
    temportal_stereo_enabled = args.temportal_stereo_enable
    points_param_enable = args.points_param_enable

    if not os.path.exists(calib_file):
        print('no valid calib fie')
        return  

    

    em_pars = dict(image_path= input_path, 
                   calib_file= calib_file, 
                   output_path = '/tmp/pyego',
                   num_features = num_features, 
                   init_json_path = json_init_path,
                   num_cams = num_cam,
                   temportal_stereo = temportal_stereo_enabled,
                   points_params = False)

    kv = EgoMotion(**em_pars)
    # failed_indexs = kv.get_failed_kite_list()

    # for img_id in range(kv.num_imgs):
    #     images, timestamp_ms = kv.getOneImage(img_id)
    #     dt = kv.updateTimestamp(timestamp_ms)
    #     kv.uploadImages(images)
    #     if img_id < 2500:
    #         continue
    #     # print('img_id = ', img_id)
    #     if img_id in failed_indexs:
    #         kv.keypointsDetection(img_id)
    #         # kv.updateInterOptflow()
    #         # kv.updateIntraOptflow()
    #         kv.updateCircularOptflow()
    #         kv.outliersRejection(True)
    #         kv.computeEgomotion(img_id, timestamp_ms, dt, cam_list=CAMERA_LIST)

    kv.close()

if __name__ == '__main__':
    _main(parser.parse_args())
