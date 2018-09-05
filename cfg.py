import numpy as np
CAMERA_LIST = [0, 1]
KITE_KPTS_PATH = '/tmp/kite/'


# configs for KITTI dataset
# DATASET = 'kitti' 
# INPUT_IMAGE_PATH='/home/jzhang/vo_data/kitti/dataset/'
# INPUT_CALIB_PATH='/home/jzhang/vo_data/kitti/dataset/sequences/02/calib.txt'

# configs for KITI dataset
DATASET = 'kite'
KITE_VIDEO_FORMAT = '2x2' # or 4x1
INPUT_IMAGE_PATH ='/home/jzhang/vo_data/jzhang_R80/AeryonYard/00001'
INPUT_CALIB_PATH ='/home/jzhang/vo_data/jzhang_R80/nav_calib.cfg'
ACS_TO_CAMEAR0_ROTATION_ANGLE = 45 # 45 degree

vertices_large_0 = np.array([[ (557, 0),   (639, 0),   (639, 105), (436, 293), (386, 272)]], dtype=np.int32)
vertices_small_0 = np.array([[ (0, 247),   (0, 247),   (134, 264), (134, 305), (0, 277)  ]], dtype=np.int32) 
# vertices_small_0 = np.array([[ (0, 226),   (0, 227),   (140, 258), (135, 278), (0, 247)  ]], dtype=np.int32) 
# vertices_large_0 = np.array([[ (580, 0),   (639, 0),   (639, 115), (436, 293), (416, 284)]], dtype=np.int32)
vertices_down_0  = np.array([[ (287, 479), (287, 473), (284, 473), (368, 473), (368, 479)]], dtype=np.int32)
vertices_up_0  = np.array([[(273, 0),   (273, 0),   (273, 8),   (369, 8),   (369, 0)]], dtype=np.int32)

vertices0 = np.concatenate((vertices_up_0, vertices_down_0, vertices_small_0, vertices_large_0), axis=0)


vertices_large_1 = np.array([[ (0, 0), (70, 0), (233, 292), (187, 318), (0, 93)]], dtype=np.int32) 

vertices_small_1 = np.array([[ (511, 266), (511, 266), (639, 238), (639, 271), (515, 288)]], dtype=np.int32) 
vertices_down_1  = np.array([[ (278, 479), (278, 473), (278, 473), (369, 473), (369, 479)]], dtype=np.int32)
vertices_up_1  = np.array([[(273, 0),   (273, 0),   (273, 8),   (369, 8),   (369, 0)]], dtype=np.int32)
vertices1 = np.concatenate((vertices_up_1, vertices_down_1, vertices_small_1, vertices_large_1), axis=0)


vertices_small_2 = np.array([[ (0, 227), (0, 227), (124, 280), (105, 302), (0, 267)  ]], dtype=np.int32) 
vertices_large_2 = np.array([[ (578, 0), (639, 0), (639, 105),  (427, 288), (406, 278)]], dtype=np.int32) 
vertices_down_2  = np.array([[ (278, 479), (278, 473), (278, 473), (369, 473), (369, 479)]], dtype=np.int32)
vertices_up_2  = np.array([[(262, 0),   (262, 0),   (262, 8),   (383, 8),   (383, 0)]], dtype=np.int32)
vertices2 = np.concatenate((vertices_up_2, vertices_down_2, vertices_small_2, vertices_large_2), axis=0)


vertices_large_3 = np.array([[ (0, 0), (94, 0), (248, 285), (215, 316), (0, 136)]], dtype=np.int32) 
vertices_small_3 = np.array([[ (500, 266), (500, 266), (639, 228), (639, 261), (515, 290)]], dtype=np.int32) 
vertices_up_3  = np.array([[ (259, 0), (259, 0), (259, 15),  (389, 15), (389, 0)]], dtype=np.int32)
vertices3 = np.concatenate((vertices_up_3, vertices_small_3, vertices_large_3), axis=0)

KITE_MASK_VERTICES = []
KITE_MASK_VERTICES.append(vertices0)
KITE_MASK_VERTICES.append(vertices1)
KITE_MASK_VERTICES.append(vertices2)
KITE_MASK_VERTICES.append(vertices3)

