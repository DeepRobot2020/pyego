

import matplotlib.pyplot as plt
import cv2
from utils import *
from cfg import *

def fisheyeStereoRectify(K1, D1, K2, D2, R12, t12, zeroDisparity=False, imageSize=(640, 480)):
    flags = cv2.CALIB_ZERO_DISPARITY if zeroDisparity else 0
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(K1, D1, K2, D2, imageSize,R12, t12, flags)
    K1 = P1[0:3, 0:3]
    K2 = P2[0:3, 0:3]
    R12 = np.identity(3)
    tvec = P2[:,3]
    t12 = -(inv(K1).dot(tvec)).reshape(3,1)
    Fmtx = fundamental_matrix(R12, t12, K1, K2)
    return R1, R2, P1, P2, Fmtx

def split_image_x4(img_file):
    return split_kite_vertical_images(img_file)


def rectifyFisheyeCameraPairs(cam0_img, cam1_img, K0, K1, D0, D1, R, T, img_size = (640, 480)):    
    R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(K0, D0, K1, D1, img_size, R, T, 0, img_size, 0.0, 1.0)
    left_maps = cv2.fisheye.initUndistortRectifyMap(K0, D0, R1, P1, img_size, cv2.CV_16SC2)
    right_maps = cv2.fisheye.initUndistortRectifyMap(K1, D1, R2, P2, img_size, cv2.CV_16SC2)
    left_img_remap = cv2.remap(cam0_img, left_maps[0], left_maps[1], cv2.INTER_LANCZOS4)
    right_img_remap = cv2.remap(cam1_img, right_maps[0], right_maps[1], cv2.INTER_LANCZOS4)
    return left_img_remap, right_img_remap, P1, P2

INPUT_IMAGE_PATH = os.environ["VO_IMG"]
INPUT_CALIB_PATH ='/home/jzhang/vo_data/SR80_901020874/nav_calib.cfg'
image_files = sorted(glob.glob(INPUT_IMAGE_PATH + '/*.jpg'))

# image_files.sort(key=lambda f: int(filter(str.isdigit, f)))

mtx, dist, rot, trans, _, _ = load_kite_config(INPUT_CALIB_PATH)

rot_01, trans_01 = rot[1], trans[1]
rot_10, trans_10 = invert_RT(rot_01, trans_01)

rot_23, trans_23 = rot[3], trans[3]
rot_32, trans_32 = invert_RT(rot[3], trans[3])

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2

calibs = fisheyeStereoRectify(
    mtx[0], dist[0][0:4], 
    mtx[1], dist[1][0:4],
    rot_01, trans_01)


for i in range(0, 3):
    img_x4 = split_image_x4(image_files[i])

    img1, img2, _, _ = rectifyFisheyeCameraPairs(
        img_x4[0], 
        img_x4[1], 
        mtx[0], 
        mtx[1], 
        dist[0][0:4], 
        dist[1][0:4], 
        rot_01,
        trans_01)

    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    good = []
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.8*n.distance:
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)

    # import pdb; pdb.set_trace()
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
    F = calibs[-1]
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    pts2[:,1] = pts1[:,1]

    # import pdb; pdb.set_trace()
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(img1,img2,lines1,pts1,pts2)
    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(img2,img1,lines2,pts2,pts1)
    plt.subplot(121),plt.imshow(img5)
    plt.subplot(122),plt.imshow(img3)
    plt.show()
    import pdb; pdb.set_trace()
    img = concat_images(img5, img3)
    cv2.imwrite('/tmp/test/' + str(i) + '.jpg', img)



#     img0, img1, _, _ = rectifyFisheyeCameraPairs(
#         img_x4[0], 
#         img_x4[1], 
#         mtx[0], 
#         mtx[1], 
#         dist[0][0:4], 
#         dist[1][0:4], 
#         rot_01,
#         trans_01)
    
#     roi_mask = region_of_interest_mask(img0.shape, KITE_MASK_VERTICES[0])
#     flow0 = shi_tomasi_corner_detection(img0, roi_mask = roi_mask, kpts_num = 16)

#     roi_mask = region_of_interest_mask(img0.shape, KITE_MASK_VERTICES[1])
#     flow1 = shi_tomasi_corner_detection(img1, roi_mask = roi_mask, kpts_num = 16)

#     img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2BGR)
#     img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)

#     import pdb; pdb.set_trace()


# # plt.imshow(img)

# # import pdb; pdb.set_trace()








