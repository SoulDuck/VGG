import shutil
import os , glob

img_dir = '/Volumes/Seagate Backup Plus Drive/data/fundus/retina_actmap/retina_original_actmap'
paths = glob.glob(os.path.join(img_dir ,'*.png'))
print len(paths)
actmap_paths = []
ori_paths = []
for path in paths:
    #get actmap image paths
    if 'drawRect' not in path and 'kmeans' not in  path and 'masked' not in path and 'ori.png' not in path:
        actmap_paths.append(path)
    if 'drawRect' not in path and 'kmeans' not in path and 'masked' not in path and 'ori.png' in path:
        ori_paths.append(path)

paths = glob.glob('/Volumes/Seagate Backup Plus Drive/data/fundus/retina_test/*.png')
names = []
for path in paths:
    name = os.path.split(path)[1]
    names.append(name)
print names


print actmap_paths[0]
print ori_paths[0]
dir_ , _ = os.path.split(actmap_paths[0])
actmap_testImg_paths=[]
ori_testImg_paths=[]
save_dir = '/Volumes/Seagate Backup Plus Drive/data/fundus/retina_actmap/retina_original_actmap/test'

#get activation map
for name in names:
    actmap_path=os.path.join(dir_, name)
    save_path=os.path.join(save_dir , name)
    shutil.move(actmap_path, save_path)
    actmap_testImg_paths.append(save_path)

    ori_path =os.path.join(dir_, name.replace('.png', '_ori.png'))
    save_path = os.path.join(save_dir , name.replace('.png', '_ori.png'))
    shutil.move(ori_path , save_path)
    ori_testImg_paths.append(ori_path)
    #ori_testImg_paths.append(ori_path)


assert len(actmap_testImg_paths) == len(ori_testImg_paths)
print len(actmap_testImg_paths)