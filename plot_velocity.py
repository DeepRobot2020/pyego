

import matplotlib.pyplot as plt
from apc import ImageAPCSyncer
import json
import csv
from numpy import genfromtxt
import numpy as np
import pandas as pd 
import argparse
import csv 
from pyquaternion import Quaternion
from vision_vo_msg_parser import  JsonApcLog
import pickle


from utils import *
from cfg import *

#  python plot_velocity.py -i /home/jzhang/vo_data/SR80_901020874/2018-11-01/Split/seg06/log.json -s 1.8 -csv /tmp/kite.csv

parser = argparse.ArgumentParser(
    description='Plot Estimated Velocity Vs GPS Velocity')


parser.add_argument(
    '-f',
    '--flight_log_json',
    type=str,
    help='Input flight_log_json file',
    default=None)

parser.add_argument(
    '-i',
    '--input_json',
    type=str,
    help='Input json file',
    default=None)

parser.add_argument(
    '-ic',
    '--input_csv',
    type=str,
    help='Input csv file',
    default=None)

parser.add_argument(
    '-t',
    '--start_time',
    type=int,
    help='Start timestamp',
    default=-1)


parser.add_argument(
    '-s',
    '--cov_threshold',
    type=float,
    help='Threshold for confidence',
    default=1.0)

parser.add_argument(
    '-csv',
    '--out_csv_file',
    type=str,
    help='Whether to output to a csv file or not',
    default=None)

class KiteLog:
    def __init__(self, json_file, csv_file, flight_json_log):
        self.json_file = json_file    
        self.syncer = ImageAPCSyncer(json_file)
        self.msg_list = ['gps_ned_vel', 'body_lin_vel', 'ned_lin_vel', 'body_lin_vel_conf', 'ned_lin_vel_conf', 'body_ang_vel_conf', 'time_ms']
        self.msg_dict_list = None
        self._inited = False
        self.parse_log_to_array()
        self.gps_body_vel = None
        self.csv_data = None
        self.flg = None
        self.flight_json_log = flight_json_log
        if flight_json_log:
            self.flg = JsonApcLog(flight_json_log)
    
        if csv_file is not None:
            csv_data = np.genfromtxt(csv_file, delimiter=',', skip_header=1, invalid_raise=False)
            self.csv_data = csv_data
        self.csv_body_vel = None
        self.csv_ned_vel = None
        self.csv_ned_lin_conf = None
        self.est_vel_norm = None
        self.gps_vel_norm = None
        self.gps_ned_vel = None
        self.gps_time_ms = None
    
    def get_gps_ned_vel(self, start_time_ms = 0):
        if self.gps_time_ms is None:
            gps_ned = self.flg.msg_dict['GPS_STATE']['ned_vel']
            gps_time_ms = self.flg.msg_dict['GPS_STATE']['sgps_time_ms']
            gps_idx = max(0, np.abs(gps_time_ms - start_time_ms).argmin())
            self.gps_time_ms = gps_time_ms[gps_idx:]
            self.gps_ned_vel = gps_ned[gps_idx:]
        return self.gps_ned_vel

    def get_gps_vel_norm(self, start_time_ms = 0):
        gps_vel = self.get_gps_ned_vel(start_time_ms)
        return np.apply_along_axis(np.linalg.norm, 1, gps_vel)

    def get_gps_time_ms(self, start_time_ms = 0):
        if self.gps_time_ms is None:
            self.get_gps_ned_vel(start_time_ms)
        return self.gps_time_ms

    def get_est_vel_norm(self):
        if self.csv_data is None:
            return np.apply_along_axis(np.linalg.norm, 1, self.get_est_ned_vel())
        else:
            return np.apply_along_axis(np.linalg.norm, 1, self.get_csv_body_vel())

    def get_csv_time_ms(self):
        if self.csv_data is None:
            return None
        return self.csv_data[:,1]

    def get_csv_body_vel(self):
        if self.csv_data is None:
            return None
        return self.csv_data[:,5:8]

    def get_csv_body_vel_conf(self):
        if self.csv_data is None:
            return None
        return self.csv_data[:,11:14]

    def get_csv_ned_vel_conf(self):
        if self.csv_data is None:
            return None
        return self.csv_data[:,17:20]

    def get_csv_ned_vel(self):
        if self.csv_data is None:
            return None
        return self.csv_data[:,14:17]

    # def get_csv_ned_vel(self):
    #     csv_body_vel = self.get_csv_body_vel()
    #     if self.csv_ned_vel is None:
    #         Qest = self.msg_dict_list['kf4_Qest']
    #         ned_vel_list = []
    #         for i in range(len(csv_body_vel)):
    #             q0 = Quaternion(array=np.array(Qest[i]))
    #             rot_mtx = q0.rotation_matrix
    #             ned_vel = rot_mtx.dot(np.array(csv_body_vel[i]))
    #             ned_vel_list.append(ned_vel)
    #         self.csv_ned_vel = np.array(ned_vel_list) 
    #     return self.csv_ned_vel
  

    def init_msg_dict(self, keys):
        if self._inited is False:
            self.msg_dict_list = dict.fromkeys(keys, None)
            for k in keys:
                self.msg_dict_list[k] = []
            self._inited = True

    def parse_log_to_array(self):
        ''' Parse every json line and store into array
        '''
        for i in range(self.syncer.num_logs()):
            jo = self.syncer.get_json_object(i)
            self.init_msg_dict(jo.keys())
            try:
                for key in jo.keys():
                    data = jo[key]
                    if isinstance(data, list):
                        data = np.array(data)
                    self.msg_dict_list[key].append(data)
            except:
                import pdb; pdb.set_trace()
        for key in self.msg_dict_list.keys():
            self.msg_dict_list[key] = np.array(self.msg_dict_list[key])

    def get_gps_body_vel(self):
        if self.gps_body_vel is None:
            gps_ned_vel = self.msg_dict_list['gps_ned_vel']
            Qest = self.msg_dict_list['kf4_Qest']
            gps_body_vel_list = []
            for i in range(len(gps_ned_vel)):
                q0 = Quaternion(array=np.array(Qest[i]))
                rot_mtx = q0.rotation_matrix
                body_vel = rot_mtx.T.dot(np.array(gps_ned_vel[i]))
                gps_body_vel_list.append(body_vel)
            self.gps_body_vel = np.array(gps_body_vel_list)
        return self.gps_body_vel

    def get_est_ned_vel(self):
        return self.msg_dict_list['ned_lin_vel']


    def get_est_time_ms(self):
        if self.csv_data is None:
            return self.msg_dict_list['time_ms']
        else:
            return self.csv_data[:,1]

    def get_est_body_vel(self):
        return self.msg_dict_list['body_lin_vel']

    def get_lin_vel_conf(self):
        return self.msg_dict_list['ned_lin_vel_conf']

    def filter_est_vel(self, con_threshold):
        self.filter_json_vel(con_threshold)
        if self.csv_data is not None:
            self.filter_csv_vel(con_threshold)
    

    def filter_json_vel(self, con_threshold):
        est_ned_vel = self.get_est_ned_vel()
        lin_ned_conf = self.get_lin_vel_conf()

        est_ned_vel2 = []
        for i in range(0, len(est_ned_vel)):
            # import pdb; pdb.set_trace()
            conf = lin_ned_conf[i]
            vel = est_ned_vel[i]
            for j in range(3):
                if conf[j] > con_threshold or abs(vel[j]) > 40 / 3.6 and i > 0:
                    # import pdb; pdb.set_trace()
                    vel[j] = min(40 / 3.6, est_ned_vel[i-1][j])
                    self.msg_dict_list['ned_lin_vel_conf'][i][j] = 9.9
                self.msg_dict_list['ned_lin_vel_conf'][i][j] = min(conf[j], 9.9)
            est_ned_vel2.append(vel)
        self.msg_dict_list['ned_lin_vel'] = np.array(est_ned_vel2)


    def filter_csv_vel(self, con_threshold):
        est_vel = self.get_csv_body_vel()
        lin_conf = self.get_csv_body_vel_conf()

        for i in range(0, len(est_vel)):
            conf = lin_conf[i]
            vel = est_vel[i]
            for j in range(3):
                if conf[j] > con_threshold or abs(vel[j]) > 40 / 3.6 and i > 0:
                    vel[j] = min(40 / 3.6, est_vel[i-1][j])
                    lin_conf[i][j] = 9.9
                lin_conf[i][j] = min(conf[j], 9.9)
        self.csv_ned_vel = None

    def plot_ned(self, threshold = 2.0):
        plt.subplot(4, 1, 1)

        est_time_ms = self.get_est_time_ms()
        est_norm = self.get_est_vel_norm()

        gps_time_ms = self.get_gps_time_ms(est_time_ms[0])[0:len(est_time_ms)]
        gps_norm = self.get_gps_vel_norm(est_time_ms[0])[0:len(est_time_ms)]

        plt.plot(est_time_ms, est_norm, 'bo', markersize=2)
        plt.plot(gps_time_ms, gps_norm)
        plt.title('Velocity Norm: VO vs GPS', fontsize='small')

        titles = ['North Velocity VO vs GPS', 'East Velocity VO vs GPS', 'Down Velocity VO vs GPS']
        for i in range(1, 4):
            plt.subplot(4, 1, i+1)
            plt.title(titles[i-1], fontsize='small')
            if self.csv_data is None:
                est_vel = self.get_est_ned_vel()[:, i-1]
                lin_ned_conf = self.get_lin_vel_conf()
            else:
                est_vel = self.get_csv_ned_vel()[:, i-1]
                lin_ned_conf = self.get_csv_ned_vel_conf()

            good = est_vel.copy()
            bad = est_vel.copy()
            good[lin_ned_conf[:, i-1] > threshold] = np.nan
            bad[lin_ned_conf[:, i-1] < threshold] = np.nan
            n_good = np.count_nonzero(np.isnan(bad))
            n_bad = np.count_nonzero(np.isnan(good))
            print(n_good, n_bad)
            plt.plot(est_time_ms, good, 'bo', markersize=2)
            plt.plot(est_time_ms, bad, 'r+', markersize=3)
            plt.plot(gps_time_ms[0:len(est_vel)], self.get_gps_ned_vel()[0:len(est_vel), i-1])
        plt.show()

    def write_to_csv(self, file_name):
        print('writing to a csv...', file_name)
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
                       'valid']
            csv_writer.writerow(CSV_KEYS)
            for row in range(len(self.get_gps_ned_vel())):
                list_row = []
                list_row.append(self.msg_dict_list['arm_timestamp'][row]) # arm_timestamp
                list_row.append(self.msg_dict_list['time_ms'][row]) # dsp_timestamp
                for i in range(3):
                    list_row.append(self.msg_dict_list['body_ang_vel'][row].ravel()[i]) # body_ang_vel
                for i in range(3):
                    list_row.append(self.msg_dict_list['body_lin_vel'][row].ravel()[i]) # body_lin_vel
                for i in range(3):
                    list_row.append(self.msg_dict_list['body_ang_vel_conf'][row].ravel()[i]) # body_ang_vel_conf
                for i in range(3):
                    list_row.append(self.msg_dict_list['body_lin_vel_conf'][row].ravel()[i]) # body_lin_vel_conf

                list_row.append(int(self.msg_dict_list['body_lin_vel_status'][row]))
                csv_writer.writerow(list_row)   
        print('write to csv done')

def _main(args):

    if args.flight_log_json:
        flight_log_json = os.path.expanduser(args.flight_log_json)
    else:
        flight_log_json = None


    if args.input_json:
        input_json = os.path.expanduser(args.input_json)
    else:
        input_json = None

    if args.input_csv:
        input_csv = os.path.expanduser(args.input_csv)
    else:
        input_csv = None

    start_timestamp = int(args.start_time)
    cov_threshold = float(args.cov_threshold)
    csv_out_enabled = False
    csv_name = args.out_csv_file

    if csv_name:
        csv_out_enabled = True
        dirname = os.path.dirname(csv_name)
        if not os.path.exists(dirname):
            os.makedirs(dirname)


    lg = KiteLog(input_json, input_csv, flight_log_json)
    lg.filter_est_vel(cov_threshold)
    if csv_out_enabled:
        lg.write_to_csv(csv_name)
    lg.plot_ned(cov_threshold)


if __name__ == '__main__':
    _main(parser.parse_args())


