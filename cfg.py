import numpy as np

# configs for KITTI dataset
# CAMERA_LIST = [0, 1]
# DATASET = 'kitti' 
# KITE_KPTS_PATH = '/tmp/kite/'
# INPUT_IMAGE_PATH='/home/jzhang/vo_data/kitti/dataset/'
# INPUT_CALIB_PATH='/home/jzhang/vo_data/kitti/dataset/sequences/02/calib.txt'

# configs for KITI dataset

KITE_KPTS_PATH = '/tmp/kite/'
CAMERA_LIST = [0, 1, 2, 3]
DATASET = 'kite'
INPUT_IMAGE_PATH='/home/jzhang/vo_data/SN86/short/short'
INPUT_CALIB_PATH='/home/jzhang/vo_data/SN86/nav_calib.cfg'



vertices_small_0 = np.array([[ (0, 227),   (0, 227),   (134, 257), (134, 288), (0, 247)  ]], dtype=np.int32) 
vertices_large_0 = np.array([[ (580, 0),   (639, 0),   (639, 115), (433, 290), (423, 283)]], dtype=np.int32)
vertices_tiny_0  = np.array([[ (284, 479), (284, 475), (284, 475), (368, 475), (368, 479)]], dtype=np.int32)

vertices0 = np.concatenate((vertices_tiny_0, vertices_small_0, vertices_large_0), axis=0)


vertices_large_1 = np.array([[ (0, 0), (62, 0), (233, 273), (220, 282), (0, 88)]], dtype=np.int32) 
vertices_small_1 = np.array([[ (521, 266), (521, 266), (639, 238), (639, 261), (524, 279)]], dtype=np.int32) 
vertices_tiny_1  = np.array([[ (284, 479), (284, 475), (284, 475), (368, 475), (368, 479)]], dtype=np.int32)
vertices1 = np.concatenate((vertices_tiny_1, vertices_small_1, vertices_large_1), axis=0)

vertices_small_2 = np.array([[ (0, 247), (0, 247), (123, 268), (121, 283), (0, 267)  ]], dtype=np.int32) 
vertices_large_2 = np.array([[ (586, 0), (639, 0), (639, 97),  (424, 283), (415, 278)]], dtype=np.int32) 
vertices_tiny_2  = np.array([[ (259, 0), (259, 0), (259, 15),  (389, 15), (389, 0)]], dtype=np.int32)

vertices2 = np.concatenate((vertices_tiny_2, vertices_small_2, vertices_large_2), axis=0)

vertices_large_3 = np.array([[ (0, 0), (82, 0), (244, 267), (231, 275), (0, 96)]], dtype=np.int32) 
vertices_small_3 = np.array([[ (521, 266), (521, 266), (639, 238), (639, 261), (524, 279)]], dtype=np.int32) 
vertices_tiny_3  = np.array([[ (259, 0), (259, 0), (259, 15),  (389, 15), (389, 0)]], dtype=np.int32)
vertices3 = np.concatenate((vertices_tiny_3, vertices_small_3, vertices_large_3), axis=0)

KITE_MASK_VERTICES = []
KITE_MASK_VERTICES.append(vertices0)
KITE_MASK_VERTICES.append(vertices1)
KITE_MASK_VERTICES.append(vertices2)
KITE_MASK_VERTICES.append(vertices3)

