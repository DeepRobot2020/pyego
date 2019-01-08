import cv2
import numpy as np
from utils import split_kite_vertical_images

import matplotlib.pyplot as plt
IMG_FILE1 = '/home/jzhang/vo_data/SR80_901020874/2018-11-01/Split2/seg01/0.jpg'
IMG_FILE2 = '/home/jzhang/vo_data/SR80_901020874/2018-11-20/20181120F03_SR80_901020874/20181120F03_SR80_901020874_NAV_0005/0.jpg'
IMG_FILE3 = '/home/jzhang/vo_data/SR80_901020874/2018-11-28/20181128F01_SR80_901020874_NAV_0001/0.jpg'


def draw_image_histogram(image, color='b', num_bins = 256, num_regions = 8):
    
    hist = cv2.calcHist([image], [0], None, [num_bins], [0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    mu = 0
    nominator = 0 
    denominator = 0
    region_width = num_bins / num_regions

    for i in range(num_regions):
        x_i = 0
        for j in range( i * region_width, (i + 1) * region_width):
            print(j, hist[j])
            val = j * hist[j]
            x_i += val
        nominator += (i + 1) * x_i
        denominator += x_i

    mu = nominator / float(denominator)
    print('------------> {}'.format(mu))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.subplot(1, 2, 2)
    plt.bar(range(num_bins), hist.ravel().tolist(), color=color)
    plt.xlim([0, num_bins])
    plt.title('Intensity Histrogram mu {} mean {}'.format(mu, np.mean(image)))
    plt.show()

    # import pdb; pdb.set_trace()


# img = cv2.imread(IMG_FILE3, cv2.IMREAD_GRAYSCALE)
imgs = split_kite_vertical_images(IMG_FILE1)
draw_image_histogram(imgs[0], num_bins=256)

imgs = split_kite_vertical_images(IMG_FILE2)
draw_image_histogram(imgs[0], num_bins=256)

imgs = split_kite_vertical_images(IMG_FILE3)
draw_image_histogram(imgs[0], num_bins=256)

# import pdb; pdb.set_trace()