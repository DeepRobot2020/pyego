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
INPUT_IMAGE_PATH='/home/jzhang/vo_data/jzhang_R80/images'
INPUT_CALIB_PATH='/home/jzhang/vo_data/jzhang_R80/nav_calib.cfg'


vertices_large_0 = np.array([[ (557, 0),   (639, 0),   (639, 105), (436, 293), (386, 272)]], dtype=np.int32)
vertices_small_0 = np.array([[ (0, 247),   (0, 247),   (134, 264), (134, 305), (0, 277)  ]], dtype=np.int32) 
# vertices_small_0 = np.array([[ (0, 226),   (0, 227),   (140, 258), (135, 278), (0, 247)  ]], dtype=np.int32) 
# vertices_large_0 = np.array([[ (580, 0),   (639, 0),   (639, 115), (436, 293), (416, 284)]], dtype=np.int32)
vertices_down_0  = np.array([[ (287, 479), (287, 473), (284, 473), (368, 473), (368, 479)]], dtype=np.int32)
vertices_up_0  = np.array([[(273, 0),   (273, 0),   (273, 8),   (369, 8),   (369, 0)]], dtype=np.int32)

vertices0 = np.concatenate((vertices_up_0, vertices_down_0, vertices_small_0, vertices_large_0), axis=0)


vertices_large_1 = np.array([[ (0, 0), (70, 0), (239, 271), (220, 287), (0, 93)]], dtype=np.int32) 
vertices_small_1 = np.array([[ (521, 266), (521, 266), (639, 238), (639, 271), (524, 279)]], dtype=np.int32) 
vertices_down_1  = np.array([[ (278, 479), (278, 473), (278, 473), (369, 473), (369, 479)]], dtype=np.int32)
vertices_up_1  = np.array([[(273, 0),   (273, 0),   (273, 8),   (369, 8),   (369, 0)]], dtype=np.int32)
vertices1 = np.concatenate((vertices_up_1, vertices_down_1, vertices_small_1, vertices_large_1), axis=0)


vertices_small_2 = np.array([[ (0, 227), (0, 227), (123, 268), (121, 283), (0, 267)  ]], dtype=np.int32) 
vertices_large_2 = np.array([[ (578, 0), (639, 0), (639, 105),  (427, 288), (406, 278)]], dtype=np.int32) 
vertices_down_2  = np.array([[ (278, 479), (278, 473), (278, 473), (369, 473), (369, 479)]], dtype=np.int32)
vertices_up_2  = np.array([[(262, 0),   (262, 0),   (262, 8),   (383, 8),   (383, 0)]], dtype=np.int32)
vertices2 = np.concatenate((vertices_up_2, vertices_down_2, vertices_small_2, vertices_large_2), axis=0)


vertices_large_3 = np.array([[ (0, 0), (94, 0), (255, 267), (231, 286), (0, 110)]], dtype=np.int32) 
vertices_small_3 = np.array([[ (521, 266), (521, 266), (639, 228), (639, 261), (524, 279)]], dtype=np.int32) 
vertices_up_3  = np.array([[ (259, 0), (259, 0), (259, 15),  (389, 15), (389, 0)]], dtype=np.int32)
vertices3 = np.concatenate((vertices_up_3, vertices_small_3, vertices_large_3), axis=0)

KITE_MASK_VERTICES = []
KITE_MASK_VERTICES.append(vertices0)
KITE_MASK_VERTICES.append(vertices1)
KITE_MASK_VERTICES.append(vertices2)
KITE_MASK_VERTICES.append(vertices3)

