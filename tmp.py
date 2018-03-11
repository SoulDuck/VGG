import shutil
import os , glob




img_dir = '/Volumes/Seagate Backup Plus Drive/data/fundus/retina_750/'
paths = glob.glob(os.path.join(img_dir ,'*.png'))
for path in paths:
    os.rename(path , path.replace('.png', '' )+'.png')
