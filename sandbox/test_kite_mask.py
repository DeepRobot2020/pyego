import matplotlib.pyplot as plt

from utils import *
from cfg import *


imgx4 = split_kite_vertical_images('/home/jzhang/vo_data/jzhang_R80/bench/1.jpg')

for camera_index in range(4):
    cv2.imwrite('/home/jzhang/vo_data/jzhang_R80/tmp' + '/f1_' + str(camera_index) + '.jpg', imgx4[camera_index])


for camera_index in range(4):
    img_file = '/home/jzhang/vo_data/jzhang_R80/tmp/f1_{}.jpg'.format(camera_index)
    frame_gray = cv2.imread(img_file, 0)

    # import pdb; pdb.set_trace()
    roi_mask1 = region_of_interest_mask(frame_gray.shape, KITE_MASK_VERTICES[camera_index])
    roi_mask = region_of_interest_mask(frame_gray.shape, KITE_MASK_VERTICES[camera_index], filler = 1)

    feature_params = dict( maxCorners = 128,
                       qualityLevel = 0.05,
                       minDistance = 12,
                       blockSize = 7,
                       mask=roi_mask)

    key_points = cv2.goodFeaturesToTrack(frame_gray, **feature_params)


    frame_gray = apply_mask_image(frame_gray, roi_mask1)
    frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)

    # if camera_index == 1:        
    #     plt.imshow(frame_bgr)
    #     plt.show()

    for kpt in key_points:
        x, y = (int(kpt[0][0]), int(kpt[0][1]))
        color = tuple(np.random.randint(0,255,3).tolist())
        cv2.circle(frame_bgr,(x, y), 6, color,2)
    
    print(camera_index, len(key_points))
    cv2.imwrite('/home/jzhang/vo_data/jzhang_R80/tmp' + '/kpts_' + str(camera_index) + '.jpg', frame_bgr)




