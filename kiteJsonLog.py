import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R

class kiteJsonLog:
    def __init__(self, json_path):
        
        df = pd.read_json(json_path, lines=True)
        # df.dropna(inplace=True)

        # convert every key into a numpy array
        self.vo_dict = {}
        for key in df.keys():
            data = df[key].to_list()
            self.vo_dict[key] = np.array(data)
        self.file_name_ = json_path.split('/')[-1].split('.')[0]

    def fileName(self):
        return self.file_name_

    def get_vo_time_ms(self):
        return self.vo_dict['dsp_timestamp']

    def get_gps_time_ms(self):
        return self.vo_dict['gps_time_ms']

    def get_ahrs_rate(self):
        return self.vo_dict['ahrs_kf4_gyro']
        
    def get_ahrs_time_ms(self):
        return self.vo_dict['ahrs_time_ms']

    def get_body_ang_vel(self):
        return self.vo_dict['body_ang_vel']

    def get_gps_body_vel(self):
        return self.vo_dict['gps_body_vel']
        data = []
        for i in range(len(self.vo_dict['gps_time_ms'])):
            vel = self.vo_dict['gps_ned_vel'][i]
            kf4_Qest = self.vo_dict['ahrs_kf4_Qest'][i]
            orentation_to_ned = R.from_quat(kf4_Qest).as_dcm()
            body_vel = np.array(orentation_to_ned.dot(vel))
            data.append(body_vel)
        return np.array(data)

    def get_gps_ned_vel(self):
        return self.vo_dict['gps_ned_vel']
    
    def get_vo_body_vel_good(self):
        mask = (self.vo_dict['status'] != 0)
        vo_est_vel_good = self.vo_dict['body_lin_vel'].copy()
        # import pdb; pdb.set_trace()
        vo_est_vel_good[mask] = np.nan
        return vo_est_vel_good

    def get_vo_ned_lin_vel_good(self):
        mask = (self.vo_dict['status'] != 0)
        vo_est_vel_good = self.vo_dict['ned_lin_vel']
        vo_est_vel_good[mask] = np.nan
        return vo_est_vel_good


    def write_to_csv(self, file_name):
        ''' Write the results into a CSV file to be processed by Matlab
        '''
        print('writing to a csv file -> {}'.format(file_name))
        with open(file_name, 'w') as f:
            csv_writer = csv.writer(f, delimiter=',') 
            CSV_KEYS = ['arm_timestamp', 
                        'dsp_timestamp', 
                        'ang_vel0', 
                        'ang_vel1', 
                        'ang_vel2',
                        'body_lin_vel0',
                        'body_lin_vel1',
                        'body_lin_vel2',
                        'ang_vel_conf0', 
                        'ang_vel_conf1', 
                        'ang_vel_conf2',
                        'body_vel_conf0', 
                        'body_vel_conf1', 
                        'body_vel_conf2',
                        'ned_lin_vel0', 
                        'ned_lin_vel1', 
                        'ned_lin_vel2',
                        'ned_vel_conf0', 
                        'ned_vel_conf1', 
                        'ned_vel_conf2',
                        'status']
            csv_writer.writerow(CSV_KEYS)
            for i in range(len(self.vo_dict['status'])):
                list_row = []
                list_row.append(self.vo_dict['arm_timestamp'][i] )
                list_row.append(self.vo_dict['dsp_timestamp'][i])

                list_row.extend(self.vo_dict['body_ang_vel'][i].tolist()) 
                list_row.extend(self.vo_dict['body_lin_vel'][i].tolist())
                
                list_row.extend(self.vo_dict['body_ang_vel_conf'][i].tolist()) 
                list_row.extend(self.vo_dict['body_lin_vel_conf'][i].tolist())  

                list_row.extend(self.vo_dict['ned_lin_vel'][i].tolist()) 
                list_row.extend(self.vo_dict['ned_lin_vel_conf'][i].tolist())  
                list_row.append(self.vo_dict['status'][i])
                
                csv_writer.writerow(list_row)   
        print('write to csv done')
