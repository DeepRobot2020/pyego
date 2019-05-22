import cv2
import matplotlib.pyplot as plt
from utils import *
from cfg import *
import datetime
import shutil



IMG_PATH = '/home/jzhang/Kalibr/Images'


def restamp_images(image_dir, outdir):
    img_files = glob.glob(image_dir + '/*.jpg')
    left_dir = os.path.join(outdir, 'cam0')
    right_dir = os.path.join(outdir, 'cam1')

    if os.path.exists(left_dir) is True:
        shutil.rmtree(left_dir)


    if os.path.exists(right_dir) is True:
        shutil.rmtree(right_dir)

    os.makedirs(left_dir)
    os.makedirs(right_dir)


    for cnt, img_file in enumerate(img_files):
        imgx4 = split_kite_vertical_images(img_file)
        left_file = os.path.join(left_dir, str(cnt * 500)+'.jpg')
        right_file = os.path.join(right_dir, str(cnt * 500)+'.jpg')
        # import pdb; pdb.set_trace()
        cv2.imwrite(left_file, imgx4[0])
        cv2.imwrite(right_file, imgx4[1])



    

restamp_images(IMG_PATH, '/home/jzhang/Kalibr')