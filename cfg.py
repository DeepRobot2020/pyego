import numpy as np
import os


# configs for KITTI dataset
# DATASET = 'kitti' 
DATASET = 'kite'


CAMERA_LIST = [0, 1, 2, 3]


KITE_VIDEO_FORMAT = '4x1' # 2x2, 4x1, 1x1
ACS_META = ''

INPUT_CALIB_PATH ='/tmp/nav_calib.cfg'
# INPUT_CALIB_PATH ='/home/jzhang/CalibData/SRH248-1/nav_calib.cfg'

KITE_UNDISTORION_NEEDED = True

VO_INIT_JSON = os.environ["VO_INI"]
INPUT_IMAGE_PATH = '/tmp/kite/images/' #os.environ["VO_IMG"]

VO_KITE_LOG = '/tmp/kite/2019-04-29-14-52-37.json'

PYEGO_DEBUG_OUTPUT = os.path.join(INPUT_IMAGE_PATH, 'pyego')
if not os.path.exists(PYEGO_DEBUG_OUTPUT):
    os.makedirs(PYEGO_DEBUG_OUTPUT)


MAX_BODY_VEL = [40/3.6, 40/3.6, 40/3.6]
MAX_COV_VAL = 99.99

MIN_DEPTH = 0.5
MAX_DEPTH = 1e6


IMU_TO_BODY_ROT = np.array([0.7071, 0.7071, 0, -0.7071, 0.7071, 0, 0, 0, 1]).reshape(3, 3)

# Features for egomotion
AVG_REPROJECTION_ERROR = 5
USE_01_FEATURE = False
TWO_VIEW_FEATURE_WEIGHT = 1.0 

DEBUG_KEYPOINTS = True

KITE_KPTS_PATH = '/tmp/kite/'

SHI_TOMASI_MIN_DISTANCE  = 12
SHI_TOMASI_QUALITY_LEVEL = 0.0001
MAX_NUM_KEYPOINTS        = 64 
SCIPY_LS_VERBOSE_LEVEL   = 0

FIVE_POINTS_ALGO_PROB_THRESHOLD = 0.9
FIVE_POINTS_ALGO_EPI_THRESHOLD = 1e-2

# Optflow 
INTRA_OPTFLOW_WIN_SIZE = (16, 16)
INTER_OPTFLOW_WIN_SIZE = (16, 16)
CIRCULAR_OPTFLOW_WIN_SIZE = (16, 16)

# Constans control the egomotion feature erros
INTRA_OPT_FLOW_DESCRIPTOR_THRESHOLD = 10
INTER_OPT_FLOW_DESCRIPTOR_THRESHOLD = 10

INTRA_OPT_FLOW_FW_BW_ERROR_THRESHOLD = 0.25
INTER_OPT_FLOW_FW_BW_ERROR_THRESHOLD = 0.25
INTER_OPT_FLOW_EPILINE_ERROR_THRESHOLD = 1e-3


# Constants to control the egomotion LS 
LS_PARMS = dict(max_nfev=20, 
                verbose=SCIPY_LS_VERBOSE_LEVEL,
                x_scale='jac',
                jac='3-point',
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
                method='trf')

vertices_large_0 = np.array([ (557, 0), (639, 0), (639, 125), (436, 293), (386, 272), (557, 0)], dtype=np.int32)
vertices_small_0 = np.array([(0, 217), (134, 244), (134, 295), (0, 247), (0, 217)], dtype=np.int32) 
vertices0 = [vertices_large_0, vertices_small_0]

vertices_large_1 = np.array([ (0, 0), (70, 0), (266, 278), (244, 308), (0, 126), (0, 0)], dtype=np.int32) 
vertices_small_1 = np.array([ (533, 246), (639, 228), (639, 265), (542, 277), (533, 246)], dtype=np.int32) 
vertices1 = [vertices_large_1, vertices_small_1]


vertices_large_2 = np.array([(578, 0), (639, 0), (639, 105),  (418, 298), (391, 280), (578, 0)], dtype=np.int32) 
vertices_small_2 = np.array([(0, 227), (124, 262), (105, 302), (0, 267),(0, 227)], dtype=np.int32) 
vertices2 = [vertices_large_2, vertices_small_2]


vertices_large_3 = np.array([(0, 0), (94, 0), (278, 275), (260, 296), (0, 126), (0, 0)], dtype=np.int32) 
vertices_small_3 = np.array([(543, 246), (639, 225), (639, 265), (557, 274), (543, 246)], dtype=np.int32) 
vertices3 = [vertices_large_3, vertices_small_3]

KITE_MASK_VERTICES = []
KITE_MASK_VERTICES.append(vertices0)
KITE_MASK_VERTICES.append(vertices1)
KITE_MASK_VERTICES.append(vertices2)
KITE_MASK_VERTICES.append(vertices3)

