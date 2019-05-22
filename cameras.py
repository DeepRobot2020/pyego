from utils import *
from cfg import *
from copy import deepcopy

class CameraModel:
    def __init__(self, fx = 0, fy = 0, cx = 320, cy = 240, xi = 0, cam_model = 'pinhole'):
        self.cam_model = cam_model
        self.fx = fx
        self.fy = fy    
        self.cx = cx
        self.cy = cy
        self.xi = xi
        self.K = None



class navcam:
    def __init__(self, 
        index, stereo_pair_idx, 
        intrinsic_mtx,  intrinsic_dist,  
        extrinsic_rot, extrinsic_trans, 
        num_features=64,
        stereo_rectify_P = None,
        stereo_rectify_R = None):

        self.calib_d   = intrinsic_dist[0:4].astype(np.float32)
        self.calib_K0  = intrinsic_mtx.astype(np.float32)
        self.calib_R   = extrinsic_rot.astype(np.float32)
        self.calib_t   = extrinsic_trans.astype(np.float32)

        # stereo_rectify_P: Stereo rectified projection matrix 
        # stereo_rectify_R: Stereo rectified rotation matrix 
        self.stereo_rectify_P = stereo_rectify_P
        self.stereo_rectify_R = stereo_rectify_R
        self.omni_radtan_mode = False

        if self.omni_radtan_mode:
            import pdb; pdb.set_trace()
        else:
            if self.stereo_rectify_P[index] is None:
                self.calib_K   = correctCameraMatrix(intrinsic_mtx, self.calib_d)
                self.focal     = (self.calib_K[0,0] + self.calib_K[1,1]) / 2.0
                self.pp         = (self.calib_K[:2,2][0], self.calib_K[:2,2][1])  
            else:
                self.calib_K = self.stereo_rectify_P[index][0:3, 0:3]
                self.focal = (self.calib_K[0,0] + self.calib_K[1,1]) / 2.0
                self.pp = (self.calib_K[:2,2][0], self.calib_K[:2,2][1])  
                self.calib_R = np.identity(3)
                if index % 2 == 0:
                    tvec = -self.stereo_rectify_P[stereo_pair_idx][:,3]
                else:
                    tvec = self.stereo_rectify_P[index][:,3]

            self.calib_t = (inv(self.calib_K).dot(tvec)).reshape(3,1)

        self.calib_R2  = cv2.Rodrigues(self.calib_R)[0].astype(np.float32)

        
        self.num_features = num_features
        self.flow_kpt0 = None

        self.flow_kpt1 = None # temporal matching with left cam
        self.flow_kpt2 = None
        
        self.flow_kpt3 = None # stereo matching with right cam
        self.flow_kpt4 = None

        self.flow_kpt5 = None # temporal matching with right cam
        self.flow_kpt6 = None


        self.stereo_pair_cam = None
        self.stereo_pair_idx = stereo_pair_idx 

        self.stereo_R = None # camera pose rotation in 
        self.stereo_t = None 
        self.mono_cur_R = None
        self.mono_cur_t = None
        self.ego_R = None
        self.ego_t = None
        self.prev_scale = 1.0

        # keypoints which have both intra and inter matching
        self.flow_intra_inter0 = None  # original keypoints
        self.flow_intra_inter1 = None  # intra match left
        self.flow_intra_inter3 = None  # inter match right
        self.flow_intra_inter5 = None  # intra match left
    
        self.flow_inliers_mask = None

        self.flow_intra0  = None # original keypoints
        self.flow_intra1  = None # intra match

        self.flow_inter0 = None # original keypoints
        self.flow_inter3 = None # inter match

        self.intra0 = None # original keypoints
        self.intra1 = None # inter match

        self.curr_img  = None
        self.curr_stereo_img  = None
        self.prev_img  = None
        self.index   = index

        self.img_idx = None

        # K0 = self.calib_K
        # K1 = self.stereo_pair_cam.calib_K
        # rot = self.calib_R
        # trans = self.calib_t
        # self.F = fundamental_matrix(rot, trans, K0, K1) 

        self.F = None
        self.proj_mtx = None

    def undistortImage(self, img_dist):
        return undistortImage(
            img_dist, 
            self.calib_K0, 
            self.calib_K,
            self.calib_d,
            self.stereo_rectify_R[self.index],
            self.stereo_rectify_P[self.index])


    def five_point_algo(self):
        if self.intra0.shape[0] < 5:
            return None, None, None
        try:
            E, e_mask = cv2.findEssentialMat(self.intra0, self.intra1, self.calib_K, method=cv2.RANSAC, prob=0.9, threshold=0.1)
            nin, R, t, mask, pts = cv2.recoverPose(E, self.intra0, self.intra1, self.calib_K, mask = e_mask, distanceThresh=1000.0)
        except:
            return None, None, None
        return R, t, mask


    def set_stereo_pair(self, right_cam):
        '''Set the stereo pair of current camera  '''
        self.stereo_pair_cam = right_cam

    def fun(self, x0, cam_obs, y_meas):
        n_kpts_013, n_kpts_01 = cam_obs
                
        rot_vecs   = x0[0:3]
        trans_vecs = x0[3:6]
        points_013 = x0[6: 6 + 3 * n_kpts_013].reshape(-1, 3)

        flow013_0  = y_meas[0: 1 * n_kpts_013]
        flow013_1  = y_meas[1 * n_kpts_013 : 2 * n_kpts_013]
        flow013_3  = y_meas[2 * n_kpts_013 : 3 * n_kpts_013]

        flow0_err = reprojection_error(points_013, flow013_0, self.calib_K)
        flow1_err = reprojection_error(points_013, flow013_1, self.calib_K, rot_vecs, trans_vecs)
        flow3_err = reprojection_error(points_013, flow013_3, self.stereo_pair_cam.calib_K, self.calib_R2, self.calib_t)
    
        errs = flow1_err
        errs013_03 = np.vstack((flow0_err, flow3_err))
        
        flow01_err0 = None
        flow01_err1 = None

        if n_kpts_01 > 0:
            points_01  = x0[6 + 3 * n_kpts_013 : 6 + 3 * n_kpts_013 + 3 * n_kpts_01].reshape(-1, 3)
            flow01_0  = y_meas[3 * n_kpts_013  : 3 * n_kpts_013 + n_kpts_01]
            flow01_1  = y_meas[3 * n_kpts_013 + n_kpts_01: 3 * n_kpts_013 + 2*n_kpts_01]

            flow0_err = reprojection_error(points_01, flow01_0, self.calib_K)
            flow1_err = reprojection_error(points_01, flow01_1, self.calib_K, rot_vecs, trans_vecs)

            errs = np.vstack((errs, flow1_err))   
            errs013_03 = np.vstack((errs013_03, flow0_err))

        errs = np.vstack((errs, errs013_03))
        return errs.ravel()

    def reprojection_err(self, x0, cam_obs, y_meas):
        n_kpts_013, n_kpts_01 = cam_obs
        reprojection_errs = []     
        rot_vecs   = x0[0:3]
        trans_vecs = x0[3:6]
        points_013 = x0[6: 6 + 3 * n_kpts_013].reshape(-1, 3)
        flow013_0  = y_meas[0 : n_kpts_013]
        flow013_1  = y_meas[n_kpts_013 : 2 * n_kpts_013]
        flow013_3  = y_meas[2 * n_kpts_013 : 3 * n_kpts_013]

        flow0_err = reprojection_error(points_013, flow013_0, self.calib_K)
        flow1_err = reprojection_error(points_013, flow013_1, self.calib_K, rot_vecs, trans_vecs)
        flow3_err = reprojection_error(points_013, flow013_3, self.stereo_pair_cam.calib_K, self.calib_R2, self.calib_t)
    
        reprojection_errs.append(flow0_err)
        reprojection_errs.append(flow1_err)
        reprojection_errs.append(flow3_err)

        if n_kpts_01 > 0:
            points_01  = x0[6 + 3 * n_kpts_013 : 6 + 3 * n_kpts_013 + 3 * n_kpts_01].reshape(-1, 3)
            flow01_0  = y_meas[3 * n_kpts_013  : 3 * n_kpts_013 + n_kpts_01]
            flow01_1  = y_meas[3 * n_kpts_013 + n_kpts_01: 3 * n_kpts_013 + 2*n_kpts_01]
            flow0_err = reprojection_error(points_01, flow01_0, self.calib_K)
            flow1_err = reprojection_error(points_01, flow01_1, self.calib_K, rot_vecs, trans_vecs)
            reprojection_errs.append(flow0_err)
            reprojection_errs.append(flow1_err)
        return reprojection_errs

    def projection_mtx(self):
        if self.proj_mtx is not None:
            return self.proj_mtx
        T = np.zeros([3,4])
        if self.index == 0:
            T[0:3,0:3] = np.identity(3)
        else:
            T[0:3,0:3] = self.calib_R
            T[:,3][:3] = self.calib_t.ravel()
        self.proj_mtx = np.dot(self.calib_K, T)
        return self.proj_mtx
        
    def update_image(self, imgs_x4):
        self.prev_img = copy.deepcopy(self.curr_img)        
        self.curr_img = copy.deepcopy(imgs_x4[self.index])
        self.curr_stereo_img = copy.deepcopy(imgs_x4[self.stereo_pair_idx])
        if self.img_idx is None:
            self.img_idx = 0
        else:
            self.img_idx += 1

    def keypoint_detection(self):
        if self.curr_img is None:
            print('Warning: curr_img is None')
            return
        roi_mask = None
        if DATASET == 'kite':
            roi_mask = region_of_interest_mask(self.curr_img.shape, KITE_MASK_VERTICES[self.index])

        self.flow_kpt0 = shi_tomasi_corner_detection(self.curr_img, 
                                                    quality_level = SHI_TOMASI_QUALITY_LEVEL,
                                                    min_distance = SHI_TOMASI_MIN_DISTANCE, 
                                                    roi_mask = roi_mask, 
                                                    kpts_num = self.num_features)
                    

    def intra_sparse_optflow(self):
        if self.prev_img is not None:
            k0, k1, k2 = sparse_optflow(self.curr_img, self.prev_img, self.flow_kpt0, win_size=INTRA_OPTFLOW_WIN_SIZE)
            self.flow_kpt1 = k1
            self.flow_kpt2 = k2
            compare_descriptor(k0, k1, self.curr_img, self.prev_img, descriptor_threshold=INTRA_OPT_FLOW_DESCRIPTOR_THRESHOLD)
            self.filter_intra_keypoints()


    def inter_sparse_optflow(self):
        k0, k3, k4 = sparse_optflow(self.curr_img, self.curr_stereo_img, self.flow_kpt0, win_size=INTER_OPTFLOW_WIN_SIZE)
        self.flow_kpt3 = k3
        self.flow_kpt4 = k4
        compare_descriptor(k0, k3, self.curr_img, self.curr_stereo_img, descriptor_threshold=INTER_OPT_FLOW_DESCRIPTOR_THRESHOLD)
        self.filter_inter_keypoints()
    
    def circular_optflow(self, win_size = CIRCULAR_OPTFLOW_WIN_SIZE):
        if self.flow_kpt0 is None or len(self.flow_kpt0) == 0:
            return

        # Parameters for lucas kanade optical flow
        lk_params = dict( winSize  = win_size,
                          maxLevel = 4,
                          minEigThreshold=1e-4,
                          criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 4, 0.01))
       

        # import pdb; pdb.set_trace()
        # left -> right  at t
        curr_img = self.curr_img
        targ_img = self.curr_stereo_img

                
        flow_l2r_t, st, err = cv2.calcOpticalFlowPyrLK(curr_img, targ_img, self.flow_kpt0, None, **lk_params)
        self.flow_kpt0 = self.flow_kpt0[st == 1].reshape(-1,1,2)
        flow_l2r_t = flow_l2r_t[st == 1].reshape(-1,1,2)

        # right t -> right t-1
        curr_img = self.curr_stereo_img
        targ_img = self.stereo_pair_cam.prev_img
        flow_r2r_t1_to_t0, st, err = cv2.calcOpticalFlowPyrLK(curr_img, targ_img, flow_l2r_t, None, **lk_params)

        self.flow_kpt0 = self.flow_kpt0[st == 1].reshape(-1,1,2)
        flow_l2r_t = flow_l2r_t[st == 1].reshape(-1,1,2)
        flow_r2r_t1_to_t0 = flow_r2r_t1_to_t0[st == 1].reshape(-1,1,2)


        # right t -1 -> left t-1
        curr_img = self.stereo_pair_cam.prev_img
        targ_img = self.prev_img
        flow_r2l_t0, st, err = cv2.calcOpticalFlowPyrLK(curr_img, targ_img, flow_r2r_t1_to_t0, None, **lk_params)

        self.flow_kpt0 = self.flow_kpt0[st == 1].reshape(-1,1,2)
        flow_l2r_t = flow_l2r_t[st == 1].reshape(-1,1,2)
        flow_r2r_t1_to_t0 = flow_r2r_t1_to_t0[st == 1].reshape(-1,1,2)
        flow_r2l_t0 = flow_r2l_t0[st == 1].reshape(-1,1,2)
        

        # left t -1 -> left t
        curr_img = self.prev_img
        targ_img = self.curr_img
        flow_l2l_t0_2_t1, st, err = cv2.calcOpticalFlowPyrLK(curr_img, targ_img, flow_r2l_t0, None, **lk_params)


        self.flow_kpt0 = self.flow_kpt0[st == 1].reshape(-1,1,2)
        flow_l2r_t = flow_l2r_t[st == 1].reshape(-1,1,2)
        flow_r2r_t1_to_t0 = flow_r2r_t1_to_t0[st == 1].reshape(-1,1,2)
        flow_r2l_t0 = flow_r2l_t0[st == 1].reshape(-1,1,2)
        flow_l2l_t0_2_t1 = flow_l2l_t0_2_t1[st == 1].reshape(-1,1,2)


        error = (flow_l2l_t0_2_t1 - self.flow_kpt0).reshape(-1,2)
        error_max = np.max(np.abs(error), axis=1)
        mask = error_max < 0.25


        self.flow_kpt0 = self.flow_kpt0[mask  == True]

        self.flow_kpt1 = flow_r2l_t0[mask  == True]
        self.flow_kpt2 = self.flow_kpt0

        self.flow_kpt3 = flow_l2r_t[mask  == True]
        self.flow_kpt4 = self.flow_kpt0

        self.flow_kpt5 = flow_r2r_t1_to_t0[mask  == True]
        self.flow_kpt6 = self.flow_kpt3
    
        



        # self.debug_inter_keypoints(self.flow_kpt0, self.flow_kpt3, '/home/jzhang/Desktop/RealSense')
        # import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()
        dist1 = compare_descriptor(self.flow_kpt0, self.flow_kpt3, self.curr_img, self.curr_stereo_img,  descriptor_threshold=35)
        self.filter_inter_keypoints()

        dist2 = compare_descriptor(self.flow_kpt0, self.flow_kpt1,  self.curr_img, self.prev_img, descriptor_threshold=35)
        self.filter_intra_keypoints()

        # self.debug_inter_keypoints(self.flow_kpt0, self.flow_kpt3, '/home/jzhang/RealSense-Basement')

        # dist3 = compare_descriptor(self.flow_kpt3, self.flow_kpt5,  self.curr_stereo_img, self.stereo_pair_cam.prev_img, descriptor_threshold=12)

        # self.debug_inter_keypoints(self.flow_kpt0, self.flow_kpt3, None, '/home/jzhang/RealSense-Basement')
        # import pdb; pdb.set_trace()


    def sift_detect_match(self):
        if DATASET == 'kite':
            roi_mask0 = region_of_interest_mask(self.curr_img.shape, 
                                        KITE_MASK_VERTICES[self.index], 
                                        filler = 1)

            roi_mask1 = region_of_interest_mask(self.curr_img.shape, 
                                KITE_MASK_VERTICES[self.stereo_pair_cam.index], 
                                filler = 1)


        sift = cv2.xfeatures2d.SIFT_create(sigma=0.8)

        # find the keypoints and descriptors with SIFT
        kp0, des0 = sift.detectAndCompute(self.curr_img, None)
        kp1, des1 = sift.detectAndCompute(self.curr_stereo_img, None)
        kp2, des2 = sift.detectAndCompute(self.stereo_pair_cam.prev_img, None)
        kp3, des3 = sift.detectAndCompute(self.prev_img, None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)   # or pass empty dictionary

        flann = cv2.FlannBasedMatcher(index_params,search_params)

        matches = flann.knnMatch(des0 ,des1, k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.5*n.distance:
                good.append(m)

        flow_kpt0 = np.float32([ kp0[m.queryIdx].pt for m in good ]).reshape(-1,2)
        flow_kpt3 = np.float32([ kp1[m.trainIdx].pt for m in good ]).reshape(-1,2)

    def filter_intra_keypoints(self):
        img = None
        if self.prev_img is not None:
            for ct, (pt1, pt2, pt3) in enumerate(zip(self.flow_kpt0, self.flow_kpt1, self.flow_kpt2)):
                x1, y1 = (pt1[0][0], pt1[0][1])
                x2, y2 = (pt2[0][0], pt2[0][1])
                x3, y3 = (pt3[0][0], pt3[0][1])
                xe = abs(x3 - x1)
                ye = abs(y3 - y1)
                if x2 < 0.0 or y2 < 0.0:
                    self.flow_kpt1[ct][0][0] = -10.0
                    self.flow_kpt1[ct][0][1] = -10.0
                    continue
                if xe > INTRA_OPT_FLOW_FW_BW_ERROR_THRESHOLD or ye > INTRA_OPT_FLOW_FW_BW_ERROR_THRESHOLD:
                    self.flow_kpt1[ct][0][0] = -20.0
                    self.flow_kpt1[ct][0][1] = -20.0
                    continue  
        # import pdb; pdb.set_trace()
    def filter_inter_keypoints(self):
        img = None
        if self.prev_img is not None:
            if self.F is None:
                K0 = self.calib_K
                K1 = self.stereo_pair_cam.calib_K
                rot = self.calib_R
                trans = self.calib_t
                self.F = fundamental_matrix(rot, trans, K0, K1)

            ep_err = epi_constraint(self.flow_kpt0, self.flow_kpt3, self.F)
            # import pdb; pdb.set_trace()
            for ct, (err, pt1, pt3, pt4) in enumerate(zip(ep_err, self.flow_kpt0, self.flow_kpt3, self.flow_kpt4)):
                x1, y1 = (pt1[0][0], pt1[0][1])
                x3, y3 = (pt3[0][0], pt3[0][1])
                x4, y4 = (pt4[0][0], pt4[0][1])
                xe = abs(x4 - x1)
                ye = abs(y4 - y1)
                if x3 < 0.0 or y3 < 0.0:
                    self.flow_kpt3[ct][0][0] = -10.0
                    self.flow_kpt3[ct][0][1] = -10.0
                    continue
                if xe > INTER_OPT_FLOW_FW_BW_ERROR_THRESHOLD or ye > INTER_OPT_FLOW_FW_BW_ERROR_THRESHOLD:
                    self.flow_kpt3[ct][0][0] = -20.0
                    self.flow_kpt3[ct][0][1] = -20.0
                    continue  


    def final_3d(self, motion_rot = None, motion_trans = None):
        flow0 = self.flow_intra_inter0
        flow1 = self.flow_intra_inter1
        flow3 = self.flow_intra_inter3    
        try:
            points03, _, _ = triangulate_3d_points(flow0, flow3, self.calib_K, self.stereo_pair_cam.calib_K, self.calib_R, self.calib_t)
            points01, _, _ = triangulate_3d_points(flow0, flow1, self.calib_K, self.calib_K, motion_rot, motion_trans)
            return points03, points01
        except:
            return None, None




    def debug_inter_keypoints(self, kps0, kps1, points013 = None, out_dir='/tmp'):
        img = None
        if self.prev_img is not None:
            flow0 = kps0.reshape(-1,2)
            flow3 = kps1.reshape(-1,2)

            if flow0 is None or flow3 is None:
                return 
            if len(flow0) == 0 or len(flow3) == 0:
                return 
            if points013 is None:
                try:
                    R = self.calib_R
                    t = self.calib_t
                    
                    K0 = self.calib_K,
                    K1 = self.stereo_pair_cam.calib_K

                    points013, _, _ = triangulate_3d_points(flow0, flow3, K0, K1, R, t)
                except:
                    import pdb; pdb.set_trace()

            r, c = self.curr_img.shape

            img1 = cv2.cvtColor(self.curr_img, cv2.COLOR_GRAY2BGR)
            img2 = cv2.cvtColor(self.curr_stereo_img, cv2.COLOR_GRAY2BGR)

            if self.F is None:
                K0 = self.calib_K
                K1 = self.stereo_pair_cam.calib_K
                rot = self.calib_R
                trans = self.calib_t
                self.F = fundamental_matrix(rot, trans, K0, K1)

            epilines = cv2.computeCorrespondEpilines(kps0.reshape(-1,1,2), 2, self.F.T)

            added_features = []

            for ep, pt1, pt3 in zip(epilines, flow0, flow3):
                x1, y1 = (int(pt1[0]), int(pt1[1]))
                x3, y3 = (int(pt3[0]), int(pt3[1]))
                if x3 < 2.0 or y3 < 2.0:
                    continue

                color = tuple(np.random.randint(0,255,3).tolist())
              
                # cv2.putText(img1, depth, (x1 + 10, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.25, color, lineType=cv2.LINE_AA) 
                # cv2.putText(img2, depth, (x3 + 10, y3 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, lineType=cv2.LINE_AA) 

                # import pdb; pdb.set_trace()        
                cv2.circle(img1,(x1, y1), 6, color,2)
                cv2.circle(img2,(x3, y3), 6, color,2)

                ep = ep.ravel()
                x0,y0 = map(int, [0, -ep[2]/ep[1] ])
                x1,y1 = map(int, [c, -(ep[2]+ep[0]*c)/ep[1] ])
                # img2 = cv2.line(img2, (x0,y0), (x1,y1), color,1)

            img = concat_images(img1, img2)
            # import pdb; pdb.set_trace()
            if img is not None:
                out_img_name = os.path.join(out_dir, 'cam_' + str(self.index) + '_inter_' + str(self.img_idx)+'.jpg')
                cv2.imwrite(out_img_name, img)
            # import pdb; pdb.set_trace()



    def debug_intra_keypoints(self, out_dir='/tmp'):
        img = None
        if self.prev_img is not None:
            img1 = cv2.cvtColor(self.curr_img, cv2.COLOR_GRAY2BGR)
            img2 = cv2.cvtColor(self.prev_img, cv2.COLOR_GRAY2BGR)
            for pt1, pt2 in zip(self.flow_intra_inter0, self.flow_intra_inter1):
                x1, y1 = (int(pt1[0]), int(pt1[1]))
                x3, y3 = (int(pt2[0]), int(pt2[1]))
                color = tuple(np.random.randint(0,255,3).tolist())
                cv2.circle(img1,(x1, y1), 6, color,2)
                cv2.circle(img2,(x3, y3), 6, color,2)

            for pt1, pt2 in zip(self.flow_intra0, self.flow_intra1):
                x1, y1 = (int(pt1[0]), int(pt1[1]))
                x3, y3 = (int(pt2[0]), int(pt2[1]))
                color = tuple(np.random.randint(0,255,3).tolist())
                cv2.circle(img1,(x1, y1), 6, color,2)
                cv2.circle(img2,(x3, y3), 6, color,2)

            img = concat_images(img1, img2)
            if img is not None:
                out_img_name = os.path.join(out_dir, 'cam_' + str(self.index) + '_intra_' + str(self.img_idx)+'.jpg')
                cv2.imwrite(out_img_name, img)


    def outliersRejection(self, debug=False, out_dir='/home/jzhang/Pictures/tmp/', max_inter_pts = 256, max_depth = 50):
        if self.prev_img is not None:            
            flow_intra_inter0 = []
            flow_intra_inter1 = []
            flow_intra_inter3 = []

            flow_intra0 = []
            flow_intra1 = []

            flow_inter0 = []
            flow_inter3 = []
            intra0 = []
            intra1 = []
            self.flow_intra_inter0 = None
            self.flow_intra_inter1 = None
            self.flow_intra_inter3 = None

            self.flow_intra0 = None
            self.flow_intra1 = None

            self.flow_inter0 = None
            self.flow_inter3 = None

            self.intra0 = None
            self.intra1 = None
            
            K0 = self.calib_K
            K1 = self.stereo_pair_cam.calib_K
            rot_01 = self.calib_R
            trans_01 = self.calib_t
            
            points013, terr0, terr1 = triangulate_3d_points(np.array(self.flow_kpt0).reshape(-1,2), 
                                                            np.array(self.flow_kpt3).reshape(-1,2), 
                                                            K0, 
                                                            K1, 
                                                            rot_01, 
                                                            trans_01)

            dist01 = np.sum(np.abs(self.flow_kpt0 - self.flow_kpt1)**2,axis=-1)**(1./2)
            dist03 = np.sum(np.abs(self.flow_kpt0 - self.flow_kpt3)**2,axis=-1)**(1./2)

            cfg_dist_03 = 0.25
            cfg_dist_01 = 0.05

            num_flow_013 = 0
            for kp0, kp1, kp3, d01, d03 in zip(self.flow_kpt0, self.flow_kpt1, self.flow_kpt3, dist01, dist03):
                x0, y0 = kp0[0][0], kp0[0][1]
                x1, y1 = kp1[0][0], kp1[0][1]
                x3, y3 = kp3[0][0], kp3[0][1]

                if d03 > cfg_dist_03 and d01 > cfg_dist_01 and x1 > 1.0 and x3 > 1.0 and num_flow_013 < max_inter_pts: # intra and inter
                    wp, _, _ = triangulate_3d_points(np.array([x0, y0]).reshape(-1,2), 
                                                     np.array([x3, y3]).reshape(-1,2), 
                                                     K0, 
                                                     K1, 
                                                     rot_01, 
                                                     trans_01)
                    
                    # import pdb ; pdb.set_trace()
                    world_pt_depth = wp.reshape(-1,)[2]

                    if world_pt_depth <= 0.1 or world_pt_depth > max_depth:
                        # print('noisy point', x0,y0, x3,y3, world_pt_depth)
                        continue
                    num_flow_013 += 1
                    flow_intra_inter0.append(np.array([x0, y0]))
                    flow_intra_inter1.append(np.array([x1, y1]))
                    flow_intra_inter3.append(np.array([x3, y3]))
                    intra0.append(np.array([x0, y0]))
                    intra1.append(np.array([x1, y1]))
                    # n_res += 1
                elif d03 > cfg_dist_03 and d01 > cfg_dist_01 and x1 > 1.0 and x3 < 1.0: # intra only
                    flow_intra0.append(np.array([x0, y0]))
                    flow_intra1.append(np.array([x1, y1]))
                    intra0.append(np.array([x0, y0]))
                    intra1.append(np.array([x1, y1]))
                elif d03 > cfg_dist_03 and d01 > cfg_dist_01 and x1 < 0.0 and x3 > 0.0: # inter only
                    flow_inter0.append(np.array([x0, y0]))
                    flow_inter3.append(np.array([x3, y3]))

            self.flow_intra_inter0 = np.array(flow_intra_inter0, dtype=np.float64)
            self.flow_intra_inter1 = np.array(flow_intra_inter1, dtype=np.float64)
            self.flow_intra_inter3 = np.array(flow_intra_inter3, dtype=np.float64)
            self.flow_inliers_mask = np.array([True] * self.flow_intra_inter0.shape[0])

    
            self.flow_intra0 = np.array(flow_intra0, dtype=np.float64)
            self.flow_intra1 = np.array(flow_intra1, dtype=np.float64)

            self.flow_inter0 = np.array(flow_inter0, dtype=np.float64)
            self.flow_inter3 = np.array(flow_inter3, dtype=np.float64)

            self.intra0 = np.array(intra0, dtype=np.float64)
            self.intra1 = np.array(intra1, dtype=np.float64)

            import pdb ; pdb.set_trace()
            if debug:    
                self.debug_inter_keypoints(self.flow_intra_inter0, self.flow_intra_inter3, out_dir)
                # self.debug_intra_keypoints(out_dir)
                print('img', self.img_idx, 'cam_'+ str(self.index ), 'intra_inter:' + str(len(self.flow_intra_inter0)), 'intra:' + str(len(self.flow_intra0)), 'inter:'+str(len(self.flow_inter0)))


    def filter_keypoints_extra(self, rot, trans, debug=False, out_dir='/home/jzhang/Pictures/tmp/', max_depth = 50):
        ''' Further filter out the bad keypoints by checking with initial motions
        '''
        if self.prev_img is not None:            
            flow_intra_inter0 = self.flow_intra_inter0
            flow_intra_inter3 = self.flow_intra_inter3

            flow_intra0_new = []
            flow_intra1_new = [] 

            K0 = self.calib_K
            K1 = self.stereo_pair_cam.calib_K
            rot_01  = self.calib_R
            trans_01 = self.calib_t

            try:
                points013, terr0, terr1 = triangulate_3d_points(np.array(flow_intra_inter0).reshape(-1,2), 
                                                                np.array(flow_intra_inter3).reshape(-1,2), 
                                                                K0, 
                                                                K1, 
                                                                rot_01, 
                                                                trans_01)
                avg_depth = np.mean(points013[:,2])
                min_depth = np.min(points013[:,2]) / 4
                max_depth = np.max(points013[:,2]) + 2 * avg_depth
                
            except:
                min_depth = 3



            try:
                points01, terr01_0, terr01_1 = triangulate_3d_points(self.flow_intra0, self.flow_intra1, K0, K0, rot, trans)
            except:
                return 0
            n_bad01 = 0

            for idx, pt01_depth in enumerate(points01[:,2]):
                if pt01_depth > min_depth and pt01_depth < max_depth:
                    flow_intra0_new.append(self.flow_intra0[idx])
                    flow_intra1_new.append(self.flow_intra1[idx])
                elif debug:
                    print('noisy 01 point', 
                        self.flow_intra0[idx][0],self.flow_intra0[idx][1], 
                        self.flow_intra1[idx][0],self.flow_intra1[idx][1], 
                        pt01_depth)
                    n_bad01 += 1
            self.flow_intra0 = np.array(flow_intra0_new, dtype=np.float64)
            self.flow_intra1 = np.array(flow_intra1_new, dtype=np.float64)
            return n_bad01

