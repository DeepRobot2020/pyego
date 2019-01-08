
import numpy as np
import cv2
import glob, os

from digits_detect import DigitsDetect

DEFAULT_TIMESTAMP_ROI = (557, 70, 80, 20) #(x, y, w, h)
import csv


class ImageReader:
    def __init__(self, data_path=None, 
                       timestamp_roi = DEFAULT_TIMESTAMP_ROI, 
                       live_mode = False, 
                       default_duration_ms = 30):
        self.data_path = data_path
        self.is_images = os.path.isdir(data_path)
        self.timestamp_list = []
        self.image_files = []
        self.video_capture = None
        self.digits_detector = DigitsDetect()
        self.roi = DEFAULT_TIMESTAMP_ROI
        self.default_duration = default_duration_ms

        self.fps_estimation_window = [self.default_duration] * 5
        self.fps_est_index = 0

        self.image_index = 0
        self.has_frames = self.initMedia()

        if live_mode is False:
            self.readAllTimestamp()
        
    def initMedia(self):
        if not os.path.exists(self.data_path):
            print('{} does not exist'.format(self.data_path))
            return False
        if self.is_images:    
            img_files = sorted(glob.glob(self.data_path + '/*.jpg'))
            img_files.sort(key=lambda f: int(filter(str.isdigit, f)))
            self.image_files = img_files
            if len(self.image_files) == 0:
                return False
        else:
            self.video_capture = cv2.VideoCapture(self.data_path)
            if not self.video_capture.isOpened:
                print('Unable to read from {} '.format(self.data_path))
                return False
        return True

    def getEstimatedFps(self):
        return sum(self.fps_estimation_window) / len(self.fps_estimation_window)

    def estimateFps(self, timestamp):
        if len(self.timestamp_list) == 0:
            return timestamp
        last_ts = self.timestamp_list[-1]
        diff = timestamp - last_ts
        if diff < 0:
            print('>>>>>Timestamp disc detected: {}'.format(timestamp) + '<<<<<<')
            fps_estimated =  self.getEstimatedFps()
            return last_ts + fps_estimated
        elif diff > 0 and diff < 200: 
            self.fps_estimation_window[self.fps_est_index] = diff
            self.fps_est_index = (self.fps_est_index + 1) % len(self.fps_estimation_window)
        else:
            print('>>>>>Timestamp disc detected: {}'.format(timestamp)+ '<<<<<<')    
        return timestamp

    def readImages(self, index):
        try:
            file = self.image_files[index]
        except IndexError:
            return -1, None
        gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        x, y, w, h = self.roi
        gray_roi  = gray[y:y+h, x:x+w]
        ts = self.digits_detector.detect(gray_roi)
        ts = self.estimateFps(ts)
        self.timestamp_list.append(ts)
        return ts, gray


    def readVideo(self):
        if self.video_capture.isOpened() is False:
            return -1, None
        ret, frame = self.video_capture.read()
        if ret is True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x, y, w, h = self.roi
            gray_roi  = gray[y:y+h, x:x+w]
            ts = self.digits_detector.detect(gray_roi)
            ts = self.estimateFps(ts)
            self.timestamp_list.append(ts)
            return ts, gray
        else:
            return -1, None


    def readAllTimestamp(self):
        ''' Simply parse all the frames and record their timestamp
        ''' 
        self.image_index = 0
        self.timestamp_list = []
        while True:
            if self.is_images:
                ts, frame = self.readImages(self.image_index)
            else:
                ts, frame = self.readVideo()
            if ts < 0:
                break
            else:
                self.image_index += 1
        self.dumpTimestampToCSV()


    def dumpTimestampToCSV(self):
        if self.is_images is False:
            csv_path = os.path.dirname(self.data_path)
        else:
            csv_path = self.data_path
        csv_file = os.path.join(csv_path, 'timestamp.csv')
        with open(csv_file, 'w') as output:
            writer = csv.writer(output, lineterminator='\n')
            for ts in self.timestamp_list:
                writer.writerow([ts])
        print('Writing CSV file {} completed'.format(csv_file))
        
    def getTimestamps(self):
        return self.timestamp_list

# IMG_PATH = '/home/jzhang/vo_data/SR80_901020874/2018-11-01/Split2/seg02.TS'
# ir = ImageReader(data_path = IMG_PATH) 
# # ir.dumpTimestampToCSV()
# # import pdb; pdb.set_trace()