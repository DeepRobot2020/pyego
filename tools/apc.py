import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import json 
import cv2
from pyquaternion import Quaternion
from vision_vo_msg_parser import JsonApcLog

from cfg import *

NUM_LOG_HEADER = 4
MSG_NAMES = ['ACS_METADATA', 'AHRSSTATE', 'GPS_STATE']

VISION_VO_KEYS = ['ACS_METADATA', 'AHRSSTATE', 'GPS_STATE']

FLIGHT_LOG_PATH = '/home/jzhang/vo_data/SR80_901020874/2018-11-15/2018-11-15/2018-11-15-flight_003/log.blog.json'
KITE_OUT_LOG = '/tmp/vo.json'

class ImageAPCSyncer:
    def __init__(self, flight_log_path = FLIGHT_LOG_PATH, KITE_LOG = KITE_OUT_LOG, start_timestamp = -1, msg_list = MSG_NAMES):
        self.json_file = FLIGHT_LOG_PATH
        self.msg_dict = dict.fromkeys(MSG_NAMES, None)
        self.start_timestamp = start_timestamp
        
        self.init_msg_dict()
        self.build_dict()
        # Build a list of json containing cleaned vison_vo_meas and vision_vo_debug_info
        self.vision_vo_log = JsonApcLog(FLIGHT_LOG_PATH)
        self.vision_vo_log.init()
        self.vision_vo_log.combine_into_jsons(start_timestamp = start_timestamp)
        self.vision_vo_log.dump_into_file(KITE_LOG)
        self.vision_vo_list = self.vision_vo_log.get_json_lines()
        self.vision_vo_array = self.vision_vo_log.get_array()
        self.prev_init_motion = np.zeros([6,1])


    def get_image_timestamp(self, index):
        ''' Get the timestamp of an image with index
        '''
        try:
            json_line = vision_vo_list[index]
            return json_line['arm_timestamp']
        except:
            print('getting get_image_timestamp() failed', index, len(self.vision_vo_list))
            return 0

    def get_image_dsp_timestamp(self, index):
        ''' Get the timestamp of an image with index
        '''
        try:
            json_line = self.vision_vo_list[index]
            return json_line['time_ms']
        except:
            print('getting get_image_timestamp() failed', index, len(self.vision_vo_list))
            return 0


    def get_gps_ned_vel(self, dsptimestamp_start, dsptimestamp_end):
        gps_dsp_time = self.msg_dict['GPS_STATE']['sgps_time_ms']
        gps_ned_pos = self.msg_dict['GPS_STATE']['ned_pos']
        gps_ned_vel = self.msg_dict['GPS_STATE']['ned_vel']
        idx00 = np.abs(gps_dsp_time - dsptimestamp_start).argmin()
        idx11 = np.abs(gps_dsp_time - dsptimestamp_end).argmin()
        return gps_ned_pos[idx00:idx11], gps_ned_vel[idx00:idx11]


    def init_msg_dict(self):
        for key in self.msg_dict.keys():
            self.msg_dict[key] = {}

    def init_sub_dict(self, name, msg):
        for key in msg.keys():
            self.msg_dict[name][key] = []

    def append_msg_dict(self, name, msg):
        for key in msg.keys():
            data = msg[key]
            if isinstance(data, list):
                data = np.array(data)
            self.msg_dict[name][key].append(data)

    def convert_dicts_to_array(self):
        for key in self.msg_dict.keys():
            for k in self.msg_dict[key]:
                self.msg_dict[key][k] = np.array(self.msg_dict[key][k])

    def build_dict(self):
        print('building dict...')
        with open(self.json_file, "r") as file:
            for i, line in enumerate(file):
                line_dict = json.loads(line)
                if i > NUM_LOG_HEADER:
                    if line_dict['name'] in self.msg_dict.keys():
                        msg_pld = line_dict['payload']
                        msg_name = line_dict['name']
                        if len(self.msg_dict[msg_name].keys()) == 0:
                            self.init_sub_dict(msg_name, msg_pld)
                        self.append_msg_dict(msg_name, msg_pld)
        # Convert the values to array
        self.convert_dicts_to_array()
        print('building dict done')

    def get_closest_acs_metadata(self, image_index):
        try:
            json_line = self.vision_vo_list[image_index]
            image_dsp_timestamp = json_line['time_ms']
            acsmeta_dsp_ts = self.msg_dict['ACS_METADATA']['time_ms']
            found = np.abs(acsmeta_dsp_ts - image_dsp_timestamp).argmin()
            # Construct ACS_META
            out_msg = np.zeros(ACS_ORIENTATION_PSI_DOT + 1)
            out_msg[TX2_TIMESTAMP_INDEX] = self.msg_dict['ACS_METADATA']['time_ms'][found]
            out_msg[DSP_TIMESTAMP_INDEX] = self.msg_dict['ACS_METADATA']['time_ms'][found]

            out_msg[ACS_POSITION_X] = self.msg_dict['ACS_METADATA']['pos_est_ned'][found][0]
            out_msg[ACS_POSITION_Y] = self.msg_dict['ACS_METADATA']['pos_est_ned'][found][1]
            out_msg[ACS_POSITION_Z] = self.msg_dict['ACS_METADATA']['pos_est_ned'][found][2]

            out_msg[ACS_POSITION_X_DOT] = self.msg_dict['ACS_METADATA']['vel_est'][found][0]
            out_msg[ACS_POSITION_Y_DOT] = self.msg_dict['ACS_METADATA']['vel_est'][found][1]
            out_msg[ACS_POSITION_Z_DOT] = self.msg_dict['ACS_METADATA']['vel_est'][found][2]

            out_msg[ACS_ORIENTATION_PHI] = self.msg_dict['ACS_METADATA']['orient_est'][found][0]
            out_msg[ACS_ORIENTATION_THETA] = self.msg_dict['ACS_METADATA']['orient_est'][found][1]
            out_msg[ACS_ORIENTATION_PSI] = self.msg_dict['ACS_METADATA']['orient_est'][found][2]

            out_msg[ACS_ORIENTATION_PHI_DOT] = self.msg_dict['ACS_METADATA']['orient_rate'][found][0]
            out_msg[ACS_ORIENTATION_THETA_DOT] = self.msg_dict['ACS_METADATA']['orient_rate'][found][1]
            out_msg[ACS_ORIENTATION_PSI_DOT] = self.msg_dict['ACS_METADATA']['orient_rate'][found][2]
            return out_msg
        except:
            print('get_closest_acs_metadata failed ')
            import pdb; pdb.set_trace()
            return None

    def get_init_motion(self, image_index):
        try:
            json_line = self.vision_vo_list[image_index]
            est_motion = np.array(json_line['init_motion'])
        except:
            print('get_init_motion failed', image_index)
            est_motion = self.prev_init_motion
        self.prev_init_motion = est_motion
        return est_motion.ravel().reshape(-1, 1)
            
    def get_initial_pose(self):
        try:
            json_line = self.vision_vo_list[0]
            acs_est_position = np.array(json_line['acs_est_position'])
            acs_est_orentation = np.array(json_line['acs_est_orentation'])
            return (acs_est_position, acsmeacs_est_orentationa_orient_est)
        except:
            print('get_initial_pose failed')
            return (np.zeros([3,1]), np.zeros([3,1]))

    def get_closest_orentation(self, image_index):

        try:
            json_line = self.vision_vo_list[image_index]
            acs_est_orentation = np.array(json_line['acs_est_orentation'])
            return cv2.Rodrigues(acs_est_orentation)[0]
        except:
            print('get_closest_orentation from vision_vo_log failed, trying from AHRSSTATE')
            json_line = self.vision_vo_list[image_index]

            ahrs_dsp_ts = self.msg_dict['AHRSSTATE']['time_ms']
            ahrs_dsp_Qest = self.msg_dict['AHRSSTATE']['kf4_Qest']
            image_dsp_timestamp = json_line['time_ms']
            
            ahrs_idx = np.abs(ahrs_dsp_ts - image_dsp_timestamp).argmin()

            Qest = ahrs_dsp_Qest[ahrs_idx]
            q0 = Quaternion(array=Qest)
            return q0.rotation_matrix
            
    




