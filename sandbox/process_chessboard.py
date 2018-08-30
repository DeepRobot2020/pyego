import matplotlib.pyplot as plt

from utils import *
from cfg import *

IMG_PATH = '/home/jzhang/pyego/data'
CALIB_PATH = '/home/jzhang/vo_data/jzhang_R80/nav_calib.cfg'

def split_image_x4(IMG_PATH):
    img_files = glob.glob(IMG_PATH + '/*.jpg')
    img_files = sorted(img_files)

    for img_file in img_files:
        imgx4 = pil_split_rotate_kite_record_image(img_file)
        img_name = img_file.split('/')[-1]
        for camera_index in range(4):
            out_path = os.path.join(IMG_PATH, 'cam' + str(camera_index))
            out_path = os.path.join(out_path, img_name)
            cv2.imwrite(out_path, imgx4[camera_index])

def undistort(img_path, K, D):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def undistort2(img_path, K, D, balance=0.0):
    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    dim2 = dim1
    dim3 = dim1
    # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(K, D, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

mtx, dist, rot, trans = load_kite_config(CALIB_PATH)

# split_image_x4(IMG_PATH)
img_path = '/home/jzhang/pyego/data/cam0/12.jpg'

img = cv2.imread(img_path)
h, w = img.shape[:2]
K = mtx[0]
D = dist[0][:4]
DIM = (img.shape[1],img.shape[0])

balance = 0.0
K2 = estimate_new_camera_matrix_for_undistort_rectify(K, D, DIM, balance)
print(K2)

img1 = undistort(img_path, K, D)
img2 = undistort2(img_path, K, D, balance = balance)


undistorted_img = concat_images(img1, img2)
cv2.imshow("undistorted", undistorted_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
