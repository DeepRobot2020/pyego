import shutil, os
from utils import *


infolder = '/home/jzhang/vo_data/SR80_901020874/Oct.5/cap'
outfolder = '/home/jzhang/vo_data/SR80_901020874/Oct.5/cap_renamed'



files = get_kite_image_files(infolder)

for img_id, f in enumerate(files):
    ofname = os.path.join(outfolder, str(img_id) + '.jpg')
    shutil.copy(f, ofname)


