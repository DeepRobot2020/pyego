import argparse
import os
import numpy as np
import json
import errno
import pickle
import cv2

from image_reader import ImageReader

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
    '-p',
    '--input_images_path',
    type=str,
    help='Start timestamp in the video',
    default=None)

parser.add_argument(
    '-o',
    '--output_path',
    type=str,
    help='Output json path',
    default='/tmp/vo.json')


POS_MSG_NAMES = ['ACS_METADATA', 'AHRSSTATE', 'GPS_STATE']
VO_MSG_NAMES = ['VISION_VO_MEAS', 'VISION_VO_DEBUG_INFO']
NUM_LOG_HEADER = 3

def load_image_timestamp(csv_file):
    print('loading image timestamps from csv...')
    if csv_file is None:
        print('loading image timestamps from csv failed')
        return None
    timestamp_list = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        for row in reader:
            timestamp_list.append(int(row[0]))
    return np.array(timestamp_list)


def fill_message_field(d, msg, key_name):
    try:
        d[key_name] = msg[key_name]
    except:
        pass
    return d


class TimeStampSmoother:
    def __init__(self, list_timestamps):
        self.timestamps = np.array(list_timestamps)
        self.estimated_duration = 33  # default 33ms

    def get_prev_good_timestamp(self, cur_idx):
        index_good = -1
        for i in range(cur_idx-1, max(0, cur_idx-100), -1):
            if self.timestamps[i] > 0:
                index_good = i
                break
        if index_good > 0:
            return index_good, self.timestamps[index_good]
        return -1, -1

    def get_next_good_timestamp(self, cur_idx):
        index_good = -1
        for i in range(cur_idx+1, min(len(self.timestamps), cur_idx+100)):
            if self.timestamps[i] > 0:
                index_good = i
                break
        if index_good > 0:
            return index_good, self.timestamps[index_good]
        return -1, -1

    def smooth_timestamp(self):
        for i in range(len(self.timestamps)):
            org_ts = self.timestamps[i]
            if org_ts > 0:
                continue
            index0, ts0 = self.get_prev_good_timestamp(i)
            index1, ts1 = self.get_next_good_timestamp(i)

            if index0 > 0 and index1 > 0:
                estimated_duration = (ts1 - ts0) / (index1 - index0)
                corrected_ts = ts0 + (i - index0) * estimated_duration
                self.estimated_duration = estimated_duration
            elif index0 > 0:
                corrected_ts = ts0 + (i - index0) * self.estimated_duration
            elif index1 > 0:
                corrected_ts = ts1 - (index1 - i) * self.estimated_duration
            self.timestamps[i] = corrected_ts
        return self.timestamps


class JsonApcLog:
    def __init__(self, json_file, image_timestamps, vo_msg_list=VO_MSG_NAMES, pos_msg_list=POS_MSG_NAMES):
        self.json_file = json_file
        self.vo_msg_list = vo_msg_list
        self.pos_msg_list = pos_msg_list
        self.image_timestamps = image_timestamps
        self.vo_meas_dict = {}
        self.vo_debug_dict = {}

        # Log the those position logs for debugging purpose
        self.pose_msg_dict = dict.fromkeys(self.pos_msg_list, None)

        for key in self.pose_msg_dict.keys():
            self.pose_msg_dict[key] = {}

        self.json_line_list = []

        self.load_flight_log()
        self.vo_log_list = self.find_vo_logs_for_timestamp_list()
        self.vo_log_list = self.find_nearest_pos_logs_for(self.vo_log_list)
 
    def _init_sub_dict(self, msg_name, msg_payload):
        for key in msg_payload.keys():
            self.pose_msg_dict[msg_name][key] = []

    def _append_msg_dict(self, name, msg):
        for key in msg.keys():
            data = msg[key]
            if isinstance(data, list):
                data = np.array(data)
            self.pose_msg_dict[name][key].append(data)

    def _convert_to_array(self):
        for key in self.pose_msg_dict.keys():
            for k in self.pose_msg_dict[key]:
                self.pose_msg_dict[key][k] = np.array(
                    self.pose_msg_dict[key][k])

    def load_flight_log(self):
        print('loading flight log...')
        with open(self.json_file, "r") as file:
            for i, line in enumerate(file):
                line_dict = json.loads(line)
                if i < NUM_LOG_HEADER:
                    continue
                msg_name = line_dict['name']
                payload = line_dict['payload']
                if msg_name == 'VISION_VO_MEAS':
                    self.vo_meas_dict[payload['tx2_time_ms']] = payload
                elif msg_name == 'VISION_VO_DEBUG_INFO':
                    self.vo_debug_dict[payload['image_ms']] = payload
                elif msg_name in self.pose_msg_dict.keys():
                    if len(self.pose_msg_dict[msg_name].keys()) == 0:
                        self._init_sub_dict(msg_name, payload)
                    self._append_msg_dict(msg_name, payload)
        self._convert_to_array()

    def get_nearest_gps_messages_for(self, dsp_timestamp):
        target_msg_name = 'GPS_STATE'
        time_ms_key = 'sgps_time_ms'

        try:
            gps_message = self.pose_msg_dict[target_msg_name]
            gps_time_ms = gps_message[time_ms_key]
        except:
            print('get_nearest_gps_messages_for {} failed for message {}'.format(
                dsp_timestamp, target_msg_name))
            return None

        best_match_idx = np.abs(gps_time_ms - dsp_timestamp).argmin()
        match_time_err = gps_time_ms[best_match_idx] - dsp_timestamp
        if abs(match_time_err) > 1000:
            print('get_nearest_gps_messages_for {} has too large error: {} '.format(
                target_msg_name, match_time_err))

        log_dict = {}
        log_dict['gps_time_ms'] = gps_time_ms[best_match_idx]
        log_dict['gps_ned_pos'] = gps_message['ned_pos'][best_match_idx].tolist()
        log_dict['gps_ned_vel'] = gps_message['ned_vel'][best_match_idx].tolist()
        return log_dict

    def get_nearest_acsmeta_orient_est_for(self, dsp_timestamp):
        target_msg_name = 'ACS_METADA'
        time_ms_key = 'time_ms'
        try:
            acs_meta_msg = self.pose_msg_dict[target_msg_name]
            acs_meta_time_ms = acs_meta_msg[time_ms_key]
        except:
            print('get_nearest_acsmeta_orient_est_for {} failed for message {}'.format(
                dsp_timestamp, target_msg_name))
            return None

        best_match_idx = np.abs(acs_meta_time_ms - dsp_timestamp).argmin()
        match_time_err = acs_meta_time_ms[best_match_idx] - dsp_timestamp

        if abs(match_time_err) > 1000:
            print('get_nearest_acsmeta_orient_est_for {} has too large error: {} '.format(
                target_msg_name, match_time_err))

        log_dict = {}
        log_dict['acsmeta_time_ms'] = acs_meta_time_ms[best_match_idx]
        log_dict['acsmeta_orient_est'] = acs_meta_msg['orient_est'][best_match_idx].tolist()
        return log_dict

    def get_nearest_ahrsstate_kf4_Qest_for(self, dsp_timestamp):
        target_msg_name = 'AHRSSTATE'
        time_ms_key = 'time_ms'
        try:
            ahrs_state_msg = self.pose_msg_dict[target_msg_name]
            ahrs_state_time_ms = ahrs_state_msg[time_ms_key]
        except:
            print('get_nearest_ahrsstate_kf4_Qest_for {} failed for message {}'.format(
                dsp_timestamp, target_msg_name))
            return None

        best_match_idx = np.abs(ahrs_state_time_ms - dsp_timestamp).argmin()
        match_time_err = ahrs_state_time_ms[best_match_idx] - dsp_timestamp

        if abs(match_time_err) > 1000:
            print('get_nearest_ahrsstate_kf4_Qest_for {} has too large error: {} '.format(
                target_msg_name, match_time_err))

        log_dict = {}
        log_dict['ahrs_time_ms'] = ahrs_state_time_ms[best_match_idx]
        log_dict['ahrs_kf4_Qest'] = ahrs_state_msg['kf4_Qest'][best_match_idx].tolist()
        return log_dict

    def get_vision_vo_messages_for(self, img_capture_timestamp):
        vo_meas_msg = None
        vo_debug_msg = None
        dsp_timestamp = -1
        log_dict = {}
        log_dict['arm_timestamp'] = img_capture_timestamp

        try:
            vo_meas_meas = self.vo_meas_dict[img_capture_timestamp]
            dsp_timestamp = vo_meas_meas['time_ms']
            log_dict['dsp_timestamp'] = dsp_timestamp

            log_dict = fill_message_field(
                log_dict, vo_meas_meas, 'ned_lin_vel')
            log_dict = fill_message_field(
                log_dict, vo_meas_meas, 'body_lin_vel')
            log_dict = fill_message_field(
                log_dict, vo_meas_meas, 'body_ang_vel')
            log_dict = fill_message_field(
                log_dict, vo_meas_meas, 'ned_lin_vel_conf')
            log_dict = fill_message_field(log_dict, vo_meas_meas, 'status')

        except:
            print('No matching vo_meas_meas for {}'.format(img_capture_timestamp))
            log_dict['dsp_timestamp'] = -1
        try:
            vo_debug_info = self.vo_debug_dict[img_capture_timestamp]

            log_dict = fill_message_field(
                log_dict, vo_debug_info, 'init_motion')
            log_dict = fill_message_field(
                log_dict, vo_debug_info, 'opt_motion')
            log_dict = fill_message_field(
                log_dict, vo_debug_info, 'reproj_err')

            log_dict = fill_message_field(
                log_dict, vo_debug_info, 'acs_est_position')
            log_dict = fill_message_field(
                log_dict, vo_debug_info, 'acs_est_orentation')
            log_dict = fill_message_field(
                log_dict, vo_debug_info, 'nearest_acs_ms')
        except:
            print('No matching vo_debug_info for {}'.format(img_capture_timestamp))

        return log_dict

    def find_vo_logs_for_timestamp_list(self):
        print('generat json logs for timestamps...')
        vo_log_list = []
        dsp_timestamp_list = []
        for img_ts in self.image_timestamps:
            log_entry = self.get_vision_vo_messages_for(img_ts)
            vo_log_list.append(log_entry)
            dsp_timestamp_list.append(log_entry['dsp_timestamp'])

        tser = TimeStampSmoother(dsp_timestamp_list)
        smoothed_timestamp = tser.smooth_timestamp()

        for i in range(len(vo_log_list)):
            if vo_log_list[i]['dsp_timestamp'] < 0:
                vo_log_list[i]['dsp_timestamp'] = smoothed_timestamp[i]
        return vo_log_list

    def find_nearest_pos_logs_for(self, vo_log_list):
        for log in vo_log_list:
            ts = log['dsp_timestamp']
            pos_data = self.get_nearest_gps_messages_for(ts)
            if pos_data is not None:
                log.update(pos_data)

            pos_data = self.get_nearest_acsmeta_orient_est_for(ts)
            if pos_data is not None:
                log.update(pos_data)

            pos_data = self.get_nearest_ahrsstate_kf4_Qest_for(ts)
            if pos_data is not None:
                log.update(pos_data)
        return vo_log_list

    def dump_into_file(self, filename):
        with open(filename, 'w') as f:
            for json_line in self.vo_log_list:
                json.dump(json_line, f, sort_keys=True)
                f.write('\n')


def _main(args):

    input_json_path = args.input_json
    if input_json_path:
        input_json_path = os.path.expanduser(input_json_path)
    if not os.path.exists(input_json_path):
        raise Exception(input_json_path + ' does not exist')

    input_images_path = args.input_images_path
    if input_images_path:
        input_images_path = os.path.expanduser(input_images_path)
    if not os.path.exists(input_images_path):
        raise Exception(input_images_path + ' does not exist')

    output_file = os.path.expanduser(args.output_path)
    # Create outfile dir and file
    if not os.path.exists(os.path.dirname(output_file)):
        try:
            os.makedirs(os.path.dirname(output_file))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise Exception('Unable to create ' + output_file)

    ir = ImageReader(input_images_path)
    log = JsonApcLog(input_json_path, ir.getTimestamps())
    log.dump_into_file(output_file)


if __name__ == '__main__':
    _main(parser.parse_args())
