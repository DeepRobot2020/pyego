import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from utils import get_kite_image_files

TX2_TIMESTAMP_INDEX = 0
DSP_TIMESTAMP_INDEX = 1

ACS_POSITION_X     = 2
ACS_POSITION_Y     = 3
ACS_POSITION_Z     = 4

ACS_POSITION_X_DOT = 5
ACS_POSITION_Y_DOT = 6
ACS_POSITION_Z_DOT = 7

ACS_ORIENTATION_PHI       = 8
ACS_ORIENTATION_THETA     = 9
ACS_ORIENTATION_PSI       = 10

ACS_ORIENTATION_PHI_DOT   = 11
ACS_ORIENTATION_THETA_DOT = 12
ACS_ORIENTATION_PSI_DOT   = 13


class ImageImuSyncer:
    def __init__(self, acsmeta_logs, image_path, image_format = '4x1', start_timestamp = -1):
        self.image_path = image_path
        self.acsmeta_logs = acsmeta_logs
        self.start_timestamp = start_timestamp
        self.image_format = image_format
        self.data_dict = {}
        self.sorted_dict_keys = None
        self.build_dict()

    def build_dict(self):
        for log in self.acsmeta_logs:
            # Load acsmetadata log file
            with open(log) as f:
                lines = f.readlines()
                for row in lines:
                    if 'KiteStatus.cpp:52' not in row:
                        continue
                    # data = np.array(row.split('|')[-1].split(','), dtype=np.float32)
                    data = row.split('|')[-1].split(',')
                    key = int(float(data[0]))
                    self.data_dict[key] = data
        camera_images = get_kite_image_files(self.image_path, self.image_format)
        # import pdb; pdb.set_trace()
        print('num_images', len(camera_images), 'num_acsmeta', len(self.data_dict.keys()))

        for i in range(len(camera_images)):
            ts = os.path.splitext(camera_images[i].split('/')[-1])[0]
            key = int(float(ts))
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
        for ts in range(image_ts_index - 1, max(0, image_ts_index - 50), -1):
            key = self.sorted_dict_keys[ts]
            self.data_dict[self.sorted_dict_keys[ts]]
            if isinstance(self.data_dict[key], np.ndarray) is True or isinstance(self.data_dict[key], list) is True:
                msg = self.data_dict[key]
                w = np.array([msg[ACS_ORIENTATION_PHI_DOT], msg[ACS_ORIENTATION_THETA_DOT], msg[ACS_ORIENTATION_PSI_DOT]], dtype=np.float32)
                v = np.array([msg[ACS_POSITION_X_DOT], msg[ACS_POSITION_Y_DOT], msg[ACS_POSITION_Z_DOT]], dtype=np.float32)
                return w, v
        return None, None

    def body_pose(self, image_ts):
        image_ts_index = np.where(self.sorted_dict_keys == image_ts)[0][0]
        
        for ts in range(image_ts_index - 1, max(0, image_ts_index - 100), -1):
            # import pdb; pdb.set_trace()
            key = self.sorted_dict_keys[ts]
            if isinstance(self.data_dict[key], np.ndarray) is True or isinstance(self.data_dict[key], list) is True:
                msg = self.data_dict[key]
                pose = np.array([msg[ACS_POSITION_X], msg[ACS_POSITION_Y], msg[ACS_POSITION_Z]], dtype=np.float32)
                return pose
        return None
    
    def get_initial_pose(self):
        for key in self.sorted_dict_keys:
            image_ts = -1
            # if isinstance(self.data_dict[key], basestring) is True: 
            if isinstance(self.data_dict[key], np.ndarray) is False and isinstance(self.data_dict[key], list) is False:
                image_ts = key
                break
                
        if image_ts == -1:
            return None, None

        image_ts_index = np.where(self.sorted_dict_keys == image_ts)[0][0]
        # import pdb; pdb.set_trace()
        # find two acs_metata message before the timestmap of current image
        for ts in range(image_ts_index - 1, max(0, image_ts_index - 11), -1):
            key = self.sorted_dict_keys[ts]
            if isinstance(self.data_dict[key], np.ndarray) is True or isinstance(self.data_dict[key], list) is True:
                msg = self.data_dict[key]
                position = np.array([msg[ACS_POSITION_X], msg[ACS_POSITION_Y], msg[ACS_POSITION_Z]], dtype=np.float32)
                oritention = np.array([msg[ACS_ORIENTATION_PHI], msg[ACS_ORIENTATION_THETA], msg[ACS_ORIENTATION_PSI]], dtype=np.float32)
                return (position, oritention)
        return (None, None)


KITE_VIDEO_FORMAT = '4x1' # 2x2, 4x1, 1x1
INPUT_IMAGE_PATH = '/home/jzhang/vo_data/SR80_901020874/Sep.24-Church/cap1'
INPUT_CALIB_PATH ='/home/jzhang/vo_data/SR80_901020874/nav_calib.cfg'
ACS_META = '/home/jzhang/vo_data/SR80_901020874/Sep.24-Church/2018-09-24-flight_001/aeryon_journal_log'

# iis = ImageImuSyncer([ACS_META], INPUT_IMAGE_PATH, KITE_VIDEO_FORMAT)
# iis.get_initial_pose()
# import pdb; pdb.set_trace()

# # for key in iis.sorted_dict_keys:
# #     # import pdb; pdb.set_trace()
# #     print(iis.data_dict[key])


# av, lv = iis.body_velocity_from_one_pose(182989843)
# import pdb; pdb.set_trace()

# # for key in iis.sorted_dict_keys:
# #     # import pdb; pdb.set_trace()
# #     print(iis.data_dict[key])
