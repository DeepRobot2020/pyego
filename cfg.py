import numpy as np

CAMERA_LIST = [0, 1]


# configs for KITTI dataset
# DATASET = 'kitti' 
DATASET = 'kite'

if DATASET == 'kitti':
    INPUT_IMAGE_PATH='/home/jzhang/vo_data/kitti/dataset/'
    INPUT_CALIB_PATH='/home/jzhang/vo_data/kitti/dataset/sequences/02/calib.txt'
    EGOMOTION_SEED_OPTION = 0 # 0: 5 point algorithem
else:
    # configs for KITI dataset
    KITE_VIDEO_FORMAT = '4x1' # 2x2, 4x1, 1x1
    INPUT_IMAGE_PATH = '/home/jzhang/vo_data/SR80_901020874/Sep.24-Church/cap3'
    INPUT_CALIB_PATH ='/home/jzhang/vo_data/SR80_901020874/nav_calib.cfg'
    ACS_META = ['/home/jzhang/vo_data/SR80_901020874/Sep.24-Church/2018-09-24-flight_001/aeryon_journal_log']
    KITE_UNDISTORION_NEEDED = True
    EGOMOTION_SEED_OPTION = 1 # 0: 5 point algorithem, 1: Velocity 2: Pose 3: prev
    KITE_SKIP_IMAGE_FACTOR = 1 
    # KITE_OUTPUT_POSE_PATH = '/tmp/kite/kv_egomotion.json'
    # KITE_OUTPUT_POSE_PATH = '/tmp/avl.json'
    KITE_OUTPUT_POSE_PATH = None

PYEGO_DEBUG_OUTPUT = '/tmp/pyego'
EGOMOTION_TRAJ_COLOR = (0, 255, 0)
GT_TRAJ_COLOR = (255, 255, 255)

IMU_TO_BODY_ROT = np.array([0.7071, 0.7071, 0, -0.7071, 0.7071, 0, 0, 0, 1]).reshape(3, 3)
# IMU_TO_BODY_ROT = IMU_TO_BODY_ROT.T

# Features for egomotion
AVG_REPROJECTION_ERROR = 2.5
USE_01_FEATURE = False
DEBUG_KEYPOINTS = True

KITE_KPTS_PATH = '/tmp/kite/'

SHI_TOMASI_MIN_DISTANCE  = 8
SHI_TOMASI_QUALITY_LEVEL = 0.01
MAX_NUM_KEYPOINTS = 64 
SCIPY_LS_VERBOSE_LEVEL = 2

FIVE_POINTS_ALGO_PROB_THRESHOLD = 0.9
FIVE_POINTS_ALGO_EPI_THRESHOLD = 1e-2

# Optflow 
INTRA_OPTFLOW_WIN_SIZE = (16, 16)
INTER_OPTFLOW_WIN_SIZE = (16, 16)

# Constans control the egomotion feature erros
INTRA_OPT_FLOW_DESCRIPTOR_THRESHOLD = 25
INTER_OPT_FLOW_DESCRIPTOR_THRESHOLD = 25

INTRA_OPT_FLOW_FW_BW_ERROR_THRESHOLD = 0.5
INTER_OPT_FLOW_FW_BW_ERROR_THRESHOLD = 0.5
INTER_OPT_FLOW_EPILINE_ERROR_THRESHOLD = 5e-4


# Constants to control the egomotion LS 
LS_PARMS = dict(max_nfev=5, 
                verbose=SCIPY_LS_VERBOSE_LEVEL,
                x_scale='jac',
                jac='2-point',
                ftol=1e-5,
                xtol=1e-5,
                gtol=1e-5,
                method='trf')


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

