import shutil, os
from utils import *


infolder = '/home/jzhang/vo_data/SR80_901020874/Sep.24-Church/cap3'
outfolder = '/home/jzhang/vo_data/SR80_901020874/Sep.24-Church/cap3_renamed'



files = get_kite_image_files(infolder)

for img_id, f in enumerate(files):
    ofname = os.path.join(outfolder, str(img_id) + '.jpg')
    shutil.copy(f, ofname)


