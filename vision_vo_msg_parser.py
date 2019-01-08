import argparse
import os
import numpy as np
import json
import errno
import pickle
import cv2
import math
from pyquaternion import Quaternion
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(
    description='Combine VISION_VO_MEAS and VISION_VO_DEBUG_INFO into one json file to be used by Kite')

parser.add_argument(
    '-i',
    '--input_json',
    type=str,
    help='Input json object from flight log',
    default=None)

parser.add_argument(
    '-s',
    '--start_time_ms',
    type=int,
    help='Start timestamp in the video',
    default=-1)

parser.add_argument(
    '-e',
    '--end_time_ms',
    type=int,
    help='End timestamp in the video',
    default=-1)

parser.add_argument(
    '-o',
    '--output_path',
    type=str,
    help='Output json path',
    default='/tmp/vo.json')

parser.add_argument(
    '-d',
    '--duration',
    type=float,
    help='frame duraton',
    default=67.5)


MSG_NAMES = ['VISION_VO_MEAS', 'VISION_VO_DEBUG_INFO', 'AHRSSTATE', 'GPS_STATE', 'ACS_METADATA', 'SACS_']
NUM_LOG_HEADER = 3

def eularAngleToRotationMatrix(eular):
    u = eular[0]; v = eular[1]; w = eular[2];
    cu = math.cos(u); su = math.sin(u);
    cv = math.cos(v); sv = math.sin(v);
    cw = math.cos(w); sw = math.sin(w);

    m00 = cu * cw
    m01 = su * sv * cw - cu * sw
    m02 = su * sw + cu * sv * cw;

    m10 = cv * sw
    m11 = cu * cw + su * sv * sw
    m12 = cu * sv * sw - su * cw 

    m20 = -sv
    m21 = su * cv 
    m22 = cu * cv 
    return np.array([m00, m01, m02, m10, m11, m12, m20, m21, m22]).reshape(3,3)

class JsonApcLog:
    def __init__(self, json_file, msg_list = MSG_NAMES):
        self.json_file = json_file    
        self.msg_dict = dict.fromkeys(msg_list, None)
        self.json_line_list = []
        for key in self.msg_dict.keys():
            self.msg_dict[key] = {}
        self.array_dict = {}
        self.array_dict_init = False

        self.ahrs_data = []
        self.gps_offset_err = []    
        self.gps_time_ms = []
        self.vo_time_ms = []
        self.first_gps_index = 0
        self.load_flight_log()

    def AHRSSTATE_axis_angle(self):
        Qest = self.msg_dict['AHRSSTATE']['kf4_Qest']
        aa_array = []
        for qu in Qest:
            q0 = Quaternion(array=np.array(qu))
            rot_mtx = q0.rotation_matrix
            aa = cv2.Rodrigues(rot_mtx)[0]
            aa_array.append(aa)
        return np.array(aa_array).reshape(-1,3)

    def ACS_METADATA_axis_angle(self):
        eular_angles = self.msg_dict['ACS_METADATA']['orient_est']
        aa_array = []
        for angle in eular_angles:
            rot = eularAngleToRotationMatrix(angle)
            aa = cv2.Rodrigues(rot)[0]
            aa_array.append(aa)
        return np.array(aa_array).reshape(-1,3)

    def get_orentation_from_acs_metadata(self, vo_acs_time_ms):
        acs_aa = self.ACS_METADATA_axis_angle()
        acs_time_ms = self.msg_dict['ACS_METADATA']['time_ms']
        aa_matched = []
        acs_matched_ms = []
        for time_ms in vo_acs_time_ms:
            idx = np.abs(acs_time_ms - time_ms).argmin()
            aa_matched.append(acs_aa[idx])
            acs_matched_ms.append(acs_time_ms[idx])
        return np.array(aa_matched), np.array(acs_matched_ms)

    def get_orentation_from_ahrs_state(self, vo_acs_time_ms):
        ahrs_aa = self.AHRSSTATE_axis_angle()
        ahrs_time_ms = self.msg_dict['AHRSSTATE']['time_ms']
        aa_matched = []
        matched_ms = []
        for time_ms in vo_acs_time_ms:
            idx = np.abs(ahrs_time_ms - time_ms).argmin()
            aa_matched.append(ahrs_aa[idx])
            matched_ms.append(ahrs_time_ms[idx])
        return np.array(aa_matched), np.array(matched_ms)

    def check_orentation(self):
        vo_orientation = self.msg_dict['VISION_VO_DEBUG_INFO']['acs_est_orentation']
        vo_acs_time_ms = self.msg_dict['VISION_VO_DEBUG_INFO']['nearest_acs_ms'][:,1]
        
        # Compare with ACS_METADATA
        acs_aa, acs_time_ms = self.get_orentation_from_acs_metadata(vo_acs_time_ms)

        plt.plot(acs_time_ms, acs_aa[:,0])
        plt.plot(vo_acs_time_ms, vo_orientation[:,0])
        plt.show()

        ahrs_aa, ahrs_time_ms = self.get_orentation_from_ahrs_state(vo_acs_time_ms)

        plt.plot(ahrs_time_ms, ahrs_aa[:,0])
        plt.plot(vo_acs_time_ms, vo_orientation[:,0])

        plt.plot(ahrs_time_ms, ahrs_aa[:,2])
        plt.plot(vo_acs_time_ms, vo_orientation[:,2])
        plt.show()

        import pdb; pdb.set_trace()

    def _init_sub_dict(self, name, msg):
        for key in msg.keys():
            self.msg_dict[name][key] = []
            
    def load_flight_log(self):
        print('building dict...')
        with open(self.json_file, "r") as file:
            for i, line in enumerate(file):
                line_dict = json.loads(line)
                if i > NUM_LOG_HEADER:
                    try:
                        if line_dict['name'] in self.msg_dict.keys():
                            msg_pld = line_dict['payload']
                            msg_name = line_dict['name']
                            if len(self.msg_dict[msg_name].keys()) == 0:
                                self._init_sub_dict(msg_name, msg_pld)
                            self._append_msg_dict(msg_name, msg_pld)
                    except:
                        import pdb; pdb.set_trace()
                    # elif line_dict['name'] == 'CMD_RESULT':
                    #     import pdb; pdb.set_trace()
        # Convert the values to array
        self._convert_dicts_to_array()
        self.filter_duplicated_msgs('GPS_STATE', 'sgps_time_ms')
        self.filter_duplicated_msgs('VISION_VO_MEAS', 'time_ms')
        self.filter_duplicated_msgs('VISION_VO_DEBUG_INFO', 'time_ms')


    def get_gps_time(self):
        return self.msg_dict['GPS_STATE']['sgps_time_ms']

    def get_gps_ned(self):
        return self.msg_dict['GPS_STATE']['ned_vel']

    def get_vo_time(self):
        return self.msg_dict['VISION_VO_MEAS']['time_ms']

    def _append_msg_dict(self, name, msg):
        for key in msg.keys():
            data = msg[key]
            if isinstance(data, list):
                data = np.array(data)
            self.msg_dict[name][key].append(data)

    def _convert_dicts_to_array(self):
        for key in self.msg_dict.keys():
            for k in self.msg_dict[key]:
                self.msg_dict[key][k] = np.array(self.msg_dict[key][k])

    def filter_duplicated_msgs(self, msg_name, key_name):
        key_array = self.msg_dict[msg_name][key_name]
        unique_index = np.unique(key_array,return_index=True)[1]
        for k in self.msg_dict[msg_name].keys():
            data = self.msg_dict[msg_name][k]
            self.msg_dict[msg_name][k] = data[unique_index]


    def generate_json_line(self, index, corrected_arm_timestamp, corrected_dsp_timestamp, vo_meas_index, vo_debug_index):
        vo_debug = self.msg_dict['VISION_VO_DEBUG_INFO']
        vo_meas = self.msg_dict['VISION_VO_MEAS']
        gps_state = self.msg_dict['GPS_STATE']
        ahrs_state = self.msg_dict['AHRSSTATE']

        vo_debug_arm_timestamp = self.msg_dict['VISION_VO_DEBUG_INFO']['image_ms']
        np.abs(vo_debug_arm_timestamp - corrected_arm_timestamp).argmin()

        json_data = {}
        json_data['arm_timestamp'] = corrected_arm_timestamp

        self.vo_time_ms.append(corrected_dsp_timestamp)
        try:
            best_match_gps_idx = np.abs(gps_state['sgps_time_ms'][self.first_gps_index:,] - corrected_dsp_timestamp).argmin()
            best_match_gps_idx += self.first_gps_index
            gps_time_err = gps_state['sgps_time_ms'][best_match_gps_idx] - corrected_dsp_timestamp

            self.gps_offset_err.append(gps_time_err)
            self.gps_time_ms.append(gps_state['sgps_time_ms'][best_match_gps_idx])

            gps_ned_pose = gps_state['ned_pos'][best_match_gps_idx]
            gps_ned_vel = gps_state['ned_vel'][best_match_gps_idx]

            json_data['gps_ned_pose'] = gps_ned_pose.tolist()
            json_data['gps_ned_vel'] = gps_ned_vel.tolist()
        except:
             print('gps_ned_pose error')
        try:
            best_match_ahrs_idx = np.abs(ahrs_state['time_ms'] - corrected_dsp_timestamp).argmin()
            kf4_Qest = ahrs_state['kf4_Qest'][best_match_gps_idx]
            self.ahrs_data.append(ahrs_state['hfield_mag'])
            json_data['kf4_Qest'] = kf4_Qest.tolist()
        except:
             print('gps_ned_pose error')

        try:
            # Log all the information from VISION_VO_MEAS
            for key in vo_meas.keys():
                json_data[key] = vo_meas[key][vo_meas_index].tolist()
                json_data['time_ms'] = corrected_dsp_timestamp
        except:
            print('generate_json_line error due to vo_meas')

        # Log the information from VISION_VO_DEBUG_INFO
        json_data['init_motion'] = vo_debug['init_motion'][vo_debug_index].tolist()
        json_data['opt_motion'] = vo_debug['opt_motion'][vo_debug_index].tolist()

        try:
            json_data['acs_est_position'] = vo_debug['acs_est_position'][vo_debug_index].tolist()
            # angle axis 
            json_data['acs_est_orentation'] = vo_debug['acs_est_orentation'][vo_debug_index].tolist()
        except:
            json_data['acs_est_position'] = [-1, -1, -1]
            json_data['acs_est_orentation'] = [-1, -1, -1]
        return json_data



    def combine_into_jsons(self, start_capture_timestamp = 0, end_capture_time = -1, expected_duration = 67, max_num_entry = 10000):
        vo_debug_cap_time = self.msg_dict['VISION_VO_DEBUG_INFO']['image_ms']
        vo_debug_dsp_time = self.msg_dict['VISION_VO_DEBUG_INFO']['nearest_acs_ms'][:,1]
        vo_meas_cap_timestamp = self.msg_dict['VISION_VO_MEAS']['tx2_time_ms']
        vo_meas_dsp_timestamp = self.msg_dict['VISION_VO_MEAS']['time_ms']

        if start_capture_timestamp <= 0:
            start_capture_timestamp = vo_meas_cap_timestamp[0]
        
        if end_capture_time <= 0:
            end_capture_time = vo_meas_cap_timestamp[-1]
    
        if end_capture_time > start_capture_timestamp:
            num_frames = int((end_capture_time - start_capture_timestamp) / expected_duration)
        else:
            num_frames = 500

        # find the index which are closest to the start_timestamp
        vo_debug_idx0 = max(0, np.abs(vo_debug_cap_time - start_capture_timestamp).argmin())
        vo_meas_idx0  = max(0, np.abs(vo_meas_cap_timestamp - start_capture_timestamp).argmin())
    
        vo_debug_idx1 = max(0, np.abs(vo_debug_cap_time - end_capture_time).argmin())
        vo_meas_idx1 = max(0, np.abs(vo_meas_cap_timestamp - end_capture_time).argmin())

        print('vo_meas', 
             vo_meas_cap_timestamp[vo_meas_idx0], 
             vo_meas_cap_timestamp[vo_meas_idx1],
             vo_meas_dsp_timestamp[vo_meas_idx0],
             vo_meas_dsp_timestamp[vo_meas_idx1])

        print('vo_debug', 
             vo_debug_cap_time[vo_debug_idx0], 
             vo_debug_cap_time[vo_debug_idx1],
             vo_debug_dsp_time[vo_debug_idx0],
             vo_debug_dsp_time[vo_debug_idx1])
        
        dsp_time = np.linspace(
            vo_meas_dsp_timestamp[vo_meas_idx0], 
            vo_meas_dsp_timestamp[vo_meas_idx0] + num_frames * expected_duration,
            num_frames)

        arm_time = np.linspace(
            vo_meas_cap_timestamp[vo_meas_idx0], 
            vo_meas_cap_timestamp[vo_meas_idx1],
            num_frames)

        # import pdb; pdb.set_trace()
        vo_debug_range = len(vo_debug_cap_time) - vo_debug_idx0
        vo_meas_range = len(vo_meas_cap_timestamp) - vo_meas_idx0
    

        start_arm_time = vo_meas_cap_timestamp[vo_meas_idx0]
        start_dsp_time = vo_meas_dsp_timestamp[vo_meas_idx0] - expected_duration

        prev_arm_time = start_arm_time - expected_duration
        prev_dsp_time = start_dsp_time - expected_duration

        for i in range(0, num_frames):
            expected_dsp_timestamp = (dsp_time[i])
            expected_arm_timestamp = (arm_time[i])

            vo_meas_idx = max(0, np.abs(vo_meas_dsp_timestamp - expected_dsp_timestamp).argmin())
            vo_debug_idx = max(0, np.abs(vo_debug_cap_time - expected_arm_timestamp).argmin())
           
            # import pdb; pdb.set_trace()
            
            vo_meas_dsp_time_err = abs(vo_meas_dsp_timestamp[vo_meas_idx] - expected_dsp_timestamp)
            vo_debug_cap_time_err = abs(vo_debug_cap_time[vo_debug_idx] - expected_arm_timestamp)


            gps_state = self.msg_dict['GPS_STATE']
            best_match_gps_idx = np.abs(gps_state['sgps_time_ms'][self.first_gps_index:,] - expected_dsp_timestamp).argmin()
            best_match_gps_idx += self.first_gps_index
            
            gps_time_err = gps_state['sgps_time_ms'][best_match_gps_idx] - expected_dsp_timestamp

            print(expected_dsp_timestamp, 
                vo_debug_dsp_time[vo_debug_idx], 
                gps_state['sgps_time_ms'][best_match_gps_idx])

            # if timestamp_err0 > 5 or timestamp_err1 >  5:
            print(i, 'expected_dsp_timestamp:', expected_dsp_timestamp, 
                  'vo_meas_dsp_time_err:', vo_meas_dsp_time_err,
                  'vo_debug_cap_time_err:', vo_debug_cap_time_err,
                  'gps_time_err:', gps_time_err)


            json_line = self.generate_json_line(
                i, 
                expected_arm_timestamp, 
                expected_dsp_timestamp, 
                vo_meas_idx, vo_debug_idx)

            if self.array_dict_init is False:
                self.array_dict = dict.fromkeys(json_line.keys(), None)
                for key in self.array_dict.keys():
                    self.array_dict[key] = []
                self.array_dict_init = True
            for key in self.array_dict.keys():
                self.array_dict[key].append(json_line[key])
            self.json_line_list.append(json_line)

        for key in self.array_dict.keys():
            self.array_dict[key] = np.array(self.array_dict[key])
        
        self.vo_time_ms = np.array(self.vo_time_ms)
        self.gps_time_ms = np.array(self.gps_time_ms)

        for k in self.msg_dict['VISION_VO_MEAS'].keys():
            data = self.msg_dict['VISION_VO_MEAS'][k] 
            self.msg_dict['VISION_VO_MEAS'][k] = data[vo_meas_idx0:]
        
    def get_array(self):
        return self.array_dict

    def get_json_lines(self):
        return self.json_line_list

    def dump_into_file(self, filename):
        with open(filename, 'w') as f:
            for json_line in self.json_line_list:
                json.dump(json_line, f, sort_keys=True)
                f.write('\n')

def _main(args):
    input_path = os.path.expanduser(args.input_json)
    start_timestamp = int(args.start_time_ms)
    end_timestamp = int(args.end_time_ms)

    output_file = os.path.expanduser(args.output_path)
    duration = args.duration

    # Check whether the input file exist 
    if not os.path.exists(input_path):
        raise Exception(input_path + ' does not exist')

    # Create outfile dir and file
    if not os.path.exists(os.path.dirname(output_file)):
        try:
            os.makedirs(os.path.dirname(output_file))
        except OSError as exc: # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise Exception('Unable to create ' + output_file)

    pickle_file = input_path+'.pickle'
    try:
        with open(pickle_file, 'rb') as cached_pickle:
            log = pickle.load(cached_pickle)
    except:
        log = JsonApcLog(input_path)

    # import pdb; pdb.set_trace()
    # log.check_orentation()
    
    log.combine_into_jsons(start_capture_timestamp = start_timestamp, end_capture_time=end_timestamp, expected_duration = duration)
    log.dump_into_file(output_file)
    # if not os.path.exists(pickle_file):
    #     with open(input_path+'.pickle', 'wb') as cached_pickle:
    #         pickle.dump(log, cached_pickle)

if __name__ == '__main__':
    _main(parser.parse_args())
