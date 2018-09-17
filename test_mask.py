import  numpy as np
from cfg import *
from utils import *
import matplotlib.pyplot as plt

class Point:
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


def isLeft(P0, P1, P2):
    return (P1.x - P0.x) * (P2.y - P0.y) - (P2.x -  P0.x) * (P1.y - P0.y)

# // wn_PnPoly(): winding number test for a point in a polygon
# //      Input:   P = a point,
# //               V[] = vertex points of a polygon V[n+1] with V[n]=V[0]
# //      Return:  wn = the winding number (=0 only when P is outside)
def wn_PnPoly_outside(P, V, n):
    wn = 0
    # import pdb; pdb.set_trace()
    for i in range(n-1): #// edge from V[i] to  V[i+1]
        if V[i].y <= P.y:           # start y <= P.y
            if V[i+1].y  > P.y and isLeft( V[i], V[i+1], P) > 0:  # P left of  edge
                wn += 1        # have  a valid up intersect
        elif V[i+1].y  <= P.y and isLeft( V[i], V[i+1], P) < 0 : # P right of  edge
                wn -= 1            # have  a valid down intersect
    return wn == 0

def isInsidePolygons(polygons, P):
    for V in polygons:
        if not wn_PnPoly_outside(P, V, len(V)):
            return True
    return False

def generateImageMask(vertices, image_shape = (640, 480)):
    w, h = image_shape[0], image_shape[1] 
    roi_mask = np.zeros([h, w], dtype = np.uint8)
    polygons = []

    for vers in vertices:
        cur_edges = []
        for p in vers:
            cur_edges.append(Point(p[0], p[1]))
        polygons.append(np.array(cur_edges))
    polygons = np.array(polygons)
    for y in range(0, h):
        for x in range(0, w):
            pt = Point(x, y)
            isIn = isInsidePolygons(polygons, pt)
            roi_mask[y][x] = 255 if isIn else 0
    # import pdb; pdb.set_trace()
    return roi_mask
    

img = '/home/jzhang/vo_data/SN51/video/20180913F05_SR80_000010051_NAV_0001/272.jpg'
imgs_x4 = pil_split_rotate_kite_record_image(img)
for i in range(len(imgs_x4)):
    cv2.imwrite('/tmp/' + str(i) + '.jpg', imgs_x4[i])

rois = []
for i in range(0, 4):
    print(i)
    roi = generateImageMask(KITE_MASK_VERTICES[i])
    rois.append(roi)
    print(i, 'done')

for i in range(0, 4):
    plt.imshow(rois[i])
    plt.show()


import pdb; pdb.set_trace()