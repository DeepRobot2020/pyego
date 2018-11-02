import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import json 
from pyquaternion import Quaternion

from cfg import *

NUM_LOG_HEADER = 4
MSG_NAMES = ['VISION_VO_MEAS', 'VISION_VO_DEBUG_INFO', 'ACS_METADATA', 'AHRSSTATE']
LOG_PATH = '/home/jzhang/vo_data/SR80_901020874/2018-11-01/2018-11-01/2018-11-01-flight_001/log.blog.json'

class ImageAPCSyncer:
    def __init__(self, log_path, start_timestamp = -1, end_timestamp = -1, msg_list = MSG_NAMES):
        self.json_file = LOG_PATH
        self.msg_dict = dict.fromkeys(MSG_NAMES, None)
        self.start_ts = start_timestamp
        self.init_msg_dict()
        self.build_dict()
        self.img_ts = self.img_ts_list()
        self.VISION_VO_MEAS = {}
        
    def img_ts_list(self):
        vo_debug_img_ts = self.msg_dict['VISION_VO_DEBUG_INFO']['image_ms']
        # find the 'VISION_VO_DEBUG_INFO' associated with this timestsamp
        image_index = np.where(vo_debug_img_ts == self.start_ts)[0]
        assert(len(image_index) > 0)
        return self.msg_dict['VISION_VO_DEBUG_INFO']['image_ms'][image_index[0]:]

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

    def find_closest_acs_metadata(self, image_ts):
        vo_debug_img_ts = self.msg_dict['VISION_VO_DEBUG_INFO']['image_ms']
        vo_debug_acs_ts = self.msg_dict['VISION_VO_DEBUG_INFO']['nearest_acs_ms']
        acsmeta_dsp_ts = self.msg_dict['ACS_METADATA']['time_ms']
        # find the 'VISION_VO_DEBUG_INFO' associated with this timestsamp
        image_index = np.where(vo_debug_img_ts == image_ts)[0]

        if len(image_index) == 0:
            print('no match VISION_VO_DEBUG_INFO for: ' + str(image_ts))
            return None, None
        matched_acs_dsp_time = vo_debug_acs_ts[image_index[0]][1]

        # find the closest ACS_META message
        # binary search 
        start = 0
        end = len(acsmeta_dsp_ts) -1
        found = -1

        while start + 1 < end:
            mid = start + (end - start) / 2
            diff = abs(acsmeta_dsp_ts[mid] - matched_acs_dsp_time)
            if diff < 10:
                found = mid
                break
            elif acsmeta_dsp_ts[mid] > matched_acs_dsp_time:
                end = mid
            else:
                start = mid

        if found == -1:
            diff1 = abs(acsmeta_dsp_ts[start] - matched_acs_dsp_time)
            diff2 = abs(acsmeta_dsp_ts[end] - matched_acs_dsp_time)
            if diff1 > diff2:
                found = end
            else:
                found = start
        diff = matched_acs_dsp_time - acsmeta_dsp_ts[found]
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

    def get_init_motion(self, image_ts):
        vo_debug_img_ts = self.msg_dict['VISION_VO_DEBUG_INFO']['image_ms']
        vo_debug_init_motion = self.msg_dict['VISION_VO_DEBUG_INFO']['init_motion']
        # find the 'VISION_VO_DEBUG_INFO' associated with this timestsamp
        image_index = np.where(vo_debug_img_ts == image_ts)[0]
        if len(image_index) == 0:
            print('get_closest_velocity: no match VISION_VO_DEBUG_INFO for: ' + str(image_ts))
            assert(0)
            return None
        vel_6d0f = np.array(vo_debug_init_motion[image_index[0]])
        return vel_6d0f

    def get_initial_pose(self):
        msg = self.find_closest_acs_metadata(self.start_ts)
        acsmeta_ned_pos = [msg[ACS_POSITION_X], msg[ACS_POSITION_Y], msg[ACS_POSITION_Z]]
        acsmeta_orient_est =  [msg[ACS_ORIENTATION_PHI], msg[ACS_ORIENTATION_THETA], msg[ACS_ORIENTATION_PSI]]
        return (acsmeta_ned_pos, acsmeta_orient_est)

    def get_closest_orentation(self, image_ts):
        vo_debug_img_ts = self.msg_dict['VISION_VO_DEBUG_INFO']['image_ms']
        vo_debug_acs_ts = self.msg_dict['VISION_VO_DEBUG_INFO']['nearest_acs_ms']
        ahrs_dsp_ts = self.msg_dict['AHRSSTATE']['time_ms']
        ahrs_dsp_Qest = self.msg_dict['AHRSSTATE']['kf4_Qest']

        # find the 'VISION_VO_DEBUG_INFO' associated with this timestsamp
        image_index = np.where(vo_debug_img_ts == image_ts)[0]

        if len(image_index) == 0:
            print('no match VISION_VO_DEBUG_INFO for: ' + str(image_ts))
            return None, None
        matched_acs_dsp_time = vo_debug_acs_ts[image_index[0]][1]

        # find the closest ACS_META message
        # binary search 
        start = 0
        end = len(ahrs_dsp_ts) -1
        found = -1

        while start + 1 < end:
            mid = start + (end - start) / 2
            diff = abs(ahrs_dsp_ts[mid] - matched_acs_dsp_time)
            if diff < 10:
                found = mid
                break
            elif ahrs_dsp_ts[mid] > matched_acs_dsp_time:
                end = mid
            else:
                start = mid

        if found == -1:
            diff1 = abs(ahrs_dsp_ts[start] - matched_acs_dsp_time)
            diff2 = abs(ahrs_dsp_ts[end] - matched_acs_dsp_time)
            if diff1 > diff2:
                found = end
            else:
                found = start
        diff = matched_acs_dsp_time - ahrs_dsp_ts[found]
        Qest = ahrs_dsp_Qest[found]
        q0 = Quaternion(array=Qest)
        return q0.rotation_matrix, diff


# sync = ImageAPCSyncer(LOG_PATH, 636341)
# sync.get_closest_orentation(636341)
# import pdb; pdb.set_trace()



# sync.get_initial_pose()
# sync.get_closest_velocity(636341)






