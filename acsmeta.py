import csv
import matplotlib.pyplot as plt
import numpy as np


from utils import *
from cfg import *

image_base = '/home/jzhang/vo_data/SR80_901020874/cap1'
acs_csv = '/home/jzhang/vo_data/SR80_901020874/cap1/acsmeta0.csv'



class ImageImuSyncer:
    def __init__(self, acsmeta_path, base_image_path, start_timestamp = -1):
        self.image_base = base_image_path
        self.acsmeta_path = acsmeta_path
        self.start_timestamp = start_timestamp
        self.data_dict = {}
        self.sorted_dict_keys = None
        self.build_dict()

    def build_dict(self):
        # Load acsmetadata csv file
        with open(self.acsmeta_path) as f:
            reader = csv.reader(f)
            for row in reader:
                key = int(float(row[0]) / 1e3)
                self.data_dict[key] = row

        camera_images = get_cam0_valid_images(self.image_base, self.start_timestamp)
        for i in range(len(camera_images)):
            ts = os.path.splitext(camera_images[i].split('/')[-1])[0]
            key = int(float(ts) / 1e3)
            self.data_dict[key] = camera_images[i]
        self.sorted_dict_keys  = np.array(sorted(self.data_dict.keys()))

    def body_velocity_from_two_poses(self, image_ts):
        image_ts_index = np.where(self.sorted_dict_keys == image_ts)[0][0]
        
        # find two acs_metata message before the timestmap of current image
        acs_message = []
        for ts in range(image_ts_index - 1, image_ts_index - 11, -1):
            # import pdb; pdb.set_trace()
            key = self.sorted_dict_keys[ts]
            if isinstance(self.data_dict[key], list):
                acs_message.append((self.data_dict[key]))
            if len(acs_message) == 2:
                time_diff    = float(acs_message[0][0]) - float(acs_message[1][0]) 
                position1    = np.array([acs_message[0][1], acs_message[0][2], acs_message[0][3]], dtype=np.float32)
                position0    = np.array([acs_message[1][1], acs_message[1][2], acs_message[1][3]], dtype=np.float32)
                orientation1 = np.array([acs_message[0][7], acs_message[0][8], acs_message[0][9]], dtype=np.float32)
                orientation0 = np.array([acs_message[1][7], acs_message[1][8], acs_message[1][9]], dtype=np.float32)
                break
        return np.eye(3)

    def body_velocity_from_one_pose(self, image_ts):
        image_ts_index = np.where(self.sorted_dict_keys == image_ts)[0][0]
        # angular and linear velocity 
        w = [0.0, 0.0, 0.0]; v = [0.0, 0.0, 0.0];

        for ts in range(image_ts_index - 1, image_ts_index - 11, -1):
            # import pdb; pdb.set_trace()
            key = self.sorted_dict_keys[ts]
            if isinstance(self.data_dict[key], list):
                msg = self.data_dict[key]
                w = np.array([msg[10], msg[11], msg[12]], dtype=np.float32)
                v = np.array([msg[4], msg[5], msg[6]], dtype=np.float32)
                break
        return w, v


syn = ImageImuSyncer(acs_csv, image_base)
av, lv = syn.body_velocity_from_one_pose(193286729)

import pdb; pdb.set_trace()