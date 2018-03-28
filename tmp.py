import os , shutil , glob
import random
paths=glob.glob('/home/mediwhale/PycharmProjects/fundus_data/cropped_original_fundus/cropped_original_fundus_300x300/glaucoma/*.png')
save_dir = '/home/mediwhale/PycharmProjects/fundus_data/cropped_original_fundus/cropped_original_fundus_300x300/glaucoma/test'
indices=random.sample(range(len(paths)) , 908)

for i in indices:
    name=os.path.split(paths[i])[1]
    shutil.move(src=paths[i] , dst= os.path.join(save_dir , name))