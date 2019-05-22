
FISHEYE_CALIB = "/home/jzhang/CalibData/SR70_801032481_2.0_org.cfg"
OMNI_CALIB    = "/home/jzhang/CalibData/SR70_801032481_2.0_omni.cfg"


# FISHEYE_CALIB = "/home/jzhang/CalibData/SR70_801032481_2.0_org.cfg"
# OMNI_CALIB    = "/home/jzhang/CalibData/SRH087_2.0_omni.cfg"

IMG_PATH = "/media/jzhang/DK_EXT/vo_data/2019-04-30/Videos/20190430F02_SR70_801032481_NAV_0001"
OUT_PATH = "/media/jzhang/DK_EXT/vo_data/2019-04-30/Videos/20190430F02_SR70_801032481_NAV_0001_OMNI_RAD_TAN"

# IMG_PATH = "/media/jzhang/DK_EXT/vo_data/2019-04-12/Videos/20190412F03_SR70_901020874_NAV_0001"
# OUT_PATH = "/media/jzhang/DK_EXT/vo_data/2019-04-12/Videos/20190412F03_SR70_901020874_NAV_0001_OMNI"

import io
import cv2
import libconf
import numpy as np
from math import sqrt

from utils import *

def computedealPinholeModel(calib_path, num_cams = 4, dim = (640, 480), balance = 0.0):
    Ks = []
    Kc = [] 
    mtx, dist, rot, trans, imu_rot, imu_trans = load_kite_config(calib_path, num_cams)
    for i in range(num_cams):
        K1 = mtx[i].astype(np.float32).reshape(3, 3)
        D = dist[i][0:4].astype(np.float32)
        K2 = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K1, D[0:4], dim, np.eye(3), balance=0)
        
        Ks.append(K1)    
        Kc.append(K2) 
    return Ks, Kc


def omniestimateNewCameraMatrixForUndistortRectify(K, K1, D, xi, image_shape):
    w = image_shape[0]
    h = image_shape[1]
    corners = []
    corners.append(np.array([w / 2.0, 0]))
    corners.append(np.array([w,  h / 2.0]))
    corners.append(np.array([w / 2.0,  h]))
    corners.append(np.array([0,  h / 2.0]))
    
    corners = np.array(corners, dtype=np.float32).reshape(1, -1, 2)
    
    map1, map2 = omniInitUndistortRectifyMap(K, D, xi, K1, image_shape)
    
    import pdb; pdb.set_trace()
    
    uc = cv2.omnidir.undistortPoints(corners, K, D, xi, np.eye(3)) 
    
    
    cn = np.array([np.mean(corners[:,0,0]), np.mean(corners[:,0,1])])

    aspect_ratio = K[0][0] / K [1][1]
    cn[0] *= aspect_ratio
    corners[:,0,1] *= aspect_ratio


    miny = np.min(corners[:,:,1])
    maxy = np.max(corners[:,:,1])
    minx = np.min(corners[:,:,0])
    maxx = np.max(corners[:,:,0])

    f1 = w * 0.5 / (cn[0] - minx)
    f2 = w * 0.5 / (maxx - cn[0])
    f3 = h * 0.5 * aspect_ratio/(cn[1] - miny)
    f4 = h * 0.5 * aspect_ratio/(maxy - cn[1])

    fmin = min(f1, min(f2, min(f3, f4)))
    fmax = max(f1, max(f2, max(f3, f4)))

    f = balance * fmin + (1.0 - balance) * fmax
    new_f = [f, f / aspect_ratio]
    new_c = -cn * f + np.array([w, h * aspect_ratio]) * 0.5
    
    return None

def omniInitUndistortRectifyMap(K, D, xi, Knew, size):
    
    width = size[0]
    height = size[1]
    
    map1 = np.zeros((height, width), dtype=np.float)
    map2 = np.zeros((height, width), dtype=np.float)

    f = np.array([K[0][0],  K[1][1]])
    c = np.array([K[0][2],  K[1][2]])
    _xi = xi
    
    k = np.array([D[0],  D[1]])
    p = np.array([D[2],  D[3]])
    
    PP = Knew
    iKR = np.linalg.inv(PP)

    for i in range(height):
        _x = i*iKR[0][1] + iKR[0][2]
        _y = i*iKR[1][1] + iKR[1][2]
        _w = i*iKR[2][1] + iKR[2][2]
        
        m1f = map1[i]
        m2f = map2[i]
        for j in range(width):
        
            # project back to unit sphere
            r = sqrt(_x*_x + _y*_y + _w*_w)
            Xs = _x / r
            Ys = _y / r
            Zs = _w / r

            #project to image plane
            xu = Xs / (Zs + _xi)
            yu = Ys / (Zs + _xi)
            # import pdb; pdb.set_trace()
            #add distortion
            r2 = xu*xu + yu*yu
            r4 = r2*r2

            xd = (1+k[0]*r2+k[1]*r4)*xu + 2*p[0]*xu*yu + p[1]*(r2+2*xu*xu)
            yd = (1+k[0]*r2+k[1]*r4)*yu + p[0]*(r2+2*yu*yu) + 2*p[1]*xu*yu
            #to image pixel
            u = f[0]*xd + c[0];
            v = f[1]*yd + c[1];

            m1f[j] = u;
            m2f[j] = v;
            
            # import pdb; pdb.set_trace()
             
            _x+=iKR[0][0]
            _y+=iKR[1][0]
            _w+=iKR[2][0]
    return map1, map2



def undistortOmniRadTanImage(distorted_frame, K_omni, Xi_omni, D_radtan, K_pinhole_ideal, new_size = (640, 480)):
    
    map1, map2 = cv2.omnidir.initUndistortRectifyMap(K_omni, D_radtan, Xi_omni, np.eye(3), K_pinhole_ideal, new_size, cv2.CV_32FC1, 1)
    # map1, map2 = omniInitUndistortRectifyMap(K_omni, D_radtan, Xi_omni, K_pinhole_ideal, new_size)
     
    # import pdb; pdb.set_trace()
    undistorted_frame1 = cv2.remap(distorted_frame, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    # undistorted_frame2 = cv2.omnidir.undistortImage(distorted_frame, K_omni, D_radtan, Xi_omni, 1, Knew = K_pinhole_ideal, new_size = new_size)
    return undistorted_frame1

def loadOmniRadTanModels(calib_path, num_cams = 4):
    mtx, dist, rot, trans, imu_rot, imu_trans = load_kite_config(calib_path, num_cams)
    Ks = []
    Ds = []
    Xis = []
    Kps = []
    
    for i in range(num_cams):
        m = mtx[i]
        xi = m[0]
        gamma_x = m[1]
        gamma_y = m[2]
        cx = m[3]
        cy = m[4]
        alpha = xi / (1 + xi)
        K = np.array([gamma_x, 0.0,  cx, 
                      0.0, gamma_y,   cy, 
                      0.0, 0.0,  1.0]).reshape(3, 3).astype(np.float32)
        
        # pinhole 
        fu = gamma_x * ( 1 - alpha) 
        fv = gamma_y * ( 1 - alpha) 
        Kp = np.array([fu, 0.0,  cx, 
                      0.0, fv,   cy, 
                      0.0, 0.0,  1.0]).reshape(3, 3).astype(np.float32)
        
        D = dist[i][0:4].astype(np.float32)
        
        Ks.append(K)
        Kps.append(Kp)
        Xis.append(xi)
        Ds.append(D)
    return Ks, Xis, Ds, Kps

K1s, K2s = computedealPinholeModel(FISHEYE_CALIB)
Ks, Xs, Ds, Kps = loadOmniRadTanModels(OMNI_CALIB)
img_files = get_image_files(IMG_PATH)

num_imgs = len(img_files)

for i in range(1):
    imgs_x4 = read_kite_image(img_files, 4, '4x1', i)
    imgs_undist = []
    for j in range(4):
        distorted_frame = imgs_x4[j]
        K_omni = Ks[j]
        Xi_omni = np.float32([Xs[j]])
        D_radtan = Ds[j]
        Kp = Kps[j]
        
        gamma_x = K_omni[0][0]
        gamma_y = K_omni[1][1]
        aspect_ratio = gamma_y / gamma_x
        
        fx = 215
        Kp[0][0] = fx
        Kp[1][1] = fx * aspect_ratio
        Kp[0][2] = 320
        Kp[1][2] = 240
         
        K_pinhole_ideal = Kp #peModels[j]

        # omniestimateNewCameraMatrixForUndistortRectify(K_omni, K_pinhole_ideal, D_radtan, Xi_omni, (640, 480))


        # import pdb; pdb.set_trace()
        im = undistortOmniRadTanImage(distorted_frame, K_omni, Xi_omni, D_radtan, K_pinhole_ideal)
        
        plt.imshow(im, cmap = 'gray')
        plt.show()
    
        imgs_undist.append(im)
    
    new_img = np.vstack(imgs_undist)    
    
    cv2.imwrite(OUT_PATH + '/' + str(i) + '.jpg',new_img)


    




