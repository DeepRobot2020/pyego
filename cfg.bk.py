import numpy as np

CAMERA_LIST = [0, 1]


# configs for KITTI dataset
# DATASET = 'kitti' 
DATASET = 'kite'

if DATASET == 'kitti':
    INPUT_IMAGE_PATH='/home/jzhang/vo_data/kitti/dataset/'
    INPUT_CALIB_PATH='/home/jzhang/vo_data/kitti/dataset/sequences/02/calib.txt'
else:
    # configs for KITI dataset
    KITE_VIDEO_FORMAT = '2x2' # 2x2, 4x1, 1x1
    # INPUT_IMAGE_PATH ='/home/jzhang/vo_data/R80_JZ/20180911_patio/cap2'
    INPUT_IMAGE_PATH = '/home/jzhang/vo_data/SN51/video/images'
    INPUT_CALIB_PATH ='/home/jzhang/vo_data/R80_JZ/nav_calib.cfg'
    ACS_TO_CAMEAR0_ROTATION_ANGLE = -45 # 45 degree
    KITE_UNDISTORION_NEEDED = True

# Features for egomotion
AVG_REPROJECTION_ERROR = 5.0
USE_01_FEATURE = False

KITE_KPTS_PATH = '/tmp/kite/'


SHI_TOMASI_MIN_DISTANCE  = 8
SHI_TOMASI_QUALITY_LEVEL = 0.01
MAX_NUM_KEYPOINTS = 64 
SCIPY_LS_VERBOSE_LEVEL = 0

FIVE_POINTS_ALGO_PROB_THRESHOLD = 0.9
FIVE_POINTS_ALGO_EPI_THRESHOLD = 1e-2

DEBUG_KEYPOINTS = True


# Optflow 
INTRA_OPTFLOW_WIN_SIZE = (16, 16)
INTER_OPTFLOW_WIN_SIZE = (16, 16)

# Constans control the egomotion feature erros
INTRA_OPT_FLOW_DESCRIPTOR_THRESHOLD = 30
INTER_OPT_FLOW_DESCRIPTOR_THRESHOLD = 30

INTRA_OPT_FLOW_FW_BW_ERROR_THRESHOLD = 0.25
INTER_OPT_FLOW_FW_BW_ERROR_THRESHOLD = 0.25
INTER_OPT_FLOW_EPILINE_ERROR_THRESHOLD = 5e-4


# Constants to control the egomotion LS 
LS_PARMS = dict(max_nfev=5, 
                verbose=SCIPY_LS_VERBOSE_LEVEL,
                x_scale='jac',
                jac='2-point',
                ftol=1e-4, 
                xtol=1e-4,
                gtol=1e-4,
                method='trf')

vertices_large_0 = np.array([ (557, 0), (639, 0), (639, 105), (436, 293), (386, 272), (557, 0)], dtype=np.int32)
vertices_small_0 = np.array([(0, 247), (134, 264), (134, 305), (0, 277), (0, 247)], dtype=np.int32) 
vertices0 = [vertices_large_0, vertices_small_0]

vertices_large_1 = np.array([ (0, 0), (70, 0), (233, 292), (187, 318), (0, 93), (0, 0)], dtype=np.int32) 
vertices_small_1 = np.array([ (511, 266), (639, 238), (639, 271), (515, 288), (511, 266)], dtype=np.int32) 
vertices1 = [vertices_large_1, vertices_small_1]


vertices_large_2 = np.array([(578, 0), (639, 0), (639, 105),  (427, 288), (406, 278), (578, 0)], dtype=np.int32) 
vertices_small_2 = np.array([(0, 227), (124, 280), (105, 302), (0, 267),(0, 227)], dtype=np.int32) 
vertices2 = [vertices_large_2, vertices_small_2]


vertices_large_3 = np.array([(0, 0), (94, 0), (248, 285), (215, 316), (0, 136), (0, 0)], dtype=np.int32) 
vertices_small_3 = np.array([(500, 266), (639, 228), (639, 261), (515, 290), (500, 266)], dtype=np.int32) 
vertices3 = [vertices_large_3, vertices_small_3]

KITE_MASK_VERTICES = []
KITE_MASK_VERTICES.append(vertices0)
KITE_MASK_VERTICES.append(vertices1)
KITE_MASK_VERTICES.append(vertices2)
KITE_MASK_VERTICES.append(vertices3)

