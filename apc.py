import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import json 
import cv2
from pyquaternion import Quaternion



from vision_vo_msg_parser import JsonApcLog

from cfg import *

class ImageAPCSyncer:
    def __init__(self, flight_log_path = None):
        self.json_file = flight_log_path
        self.json_lines = []
        self._parse_jsons()

    def _parse_jsons(self):
        with open(self.json_file, 'r') as f:
            content = f.readlines()
            for line in content:
                self.json_lines.append(json.loads(line))
    def num_logs(self):
        return len(self.json_lines)

    def get_json_keys(self):
        if len(self.json_lines) == 0:
            return None
        return sorted(self.json_lines[0].keys())

    def get_json_object(self, index):
        try:
            j = self.json_lines[index]
        except:
            return None
        return j

    def get_image_timestamp(self, index):
        ''' Get the timestamp of an image with index
        '''
        try:
            json_line = self.json_lines[index]
            return json_line['arm_timestamp']
        except:
            print('getting get_image_timestamp() failed', index, len(self.vision_vo_list))
            return 0

    def get_image_dsp_timestamp(self, index):
        ''' Get the timestamp of an image with index
        '''
        try:
            json_line = self.json_lines[index]
            return json_line['time_ms']
        except:
            print('getting get_image_timestamp() failed', index, len(self.vision_vo_list))
            return 0

    def get_init_motion(self, index):
        try:
            json_line = self.json_lines[index]
            est_motion = np.array(json_line['init_motion'])
        except:
            print('get_init_motion failed', index)
        return est_motion.ravel().reshape(-1, 1)


    def get_closest_orentation(self, image_index):
        try:
            line = self.json_lines[image_index]
            acs_est_orentation = np.array(line['acs_est_orentation'])
            Qest = np.array(line['kf4_Qest'])
            print('Qest', Qest)
            q0 = Quaternion(array=Qest)

            return q0.rotation_matrix
            if acs_est_orentation[0] <= 0:
                return q0.rotation_matrix
            else:
                return cv2.Rodrigues(acs_est_orentation)[0]
        except:
            print('get_closest_orentation from vision_vo_log failed, trying from AHRSSTATE')
            return np.zeros([3,1])


    def get_closest_position(self, image_index):
        try:
            line = self.json_lines[image_index]
            acs_est_position = np.array(line['acs_est_position'])
            if acs_est_position[0] <= 0:
                return np.array(line['gps_ned_pose'])
            else:
                return acs_est_position
        except:
            print('get_closest_orentation failed')
            return np.zeros([3,1])       

    def get_current_gps_vel(self, image_index):
        try:
            line = self.json_lines[image_index]
            gps_ned_vel = np.array(line['gps_ned_vel'])
            return gps_ned_vel
        except:
            print('get_current_gps_vel failed')
            return np.zeros([3,1])      

    def get_initial_pose(self):
        return (self.get_closest_position(0), self.get_closest_orentation(0))

    def get_closest_acs_metadata(self, index):
        return None
