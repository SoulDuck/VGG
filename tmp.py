import os
import shutil
import sys
import glob




#train_normalDir ='../lesion_detection/cropped_bg_500_clahe/'
train_normalDir ='../fundus_data/cropped_original_fundus_300x300/normal_0/*.png'

#test_normalDir='../lesion_detection/bg_cropped_rois'
test_normalDir ='../fundus_data/cropped_original_fundus_300x300/normal_0/Test/*.png'

#train_abnormalDir ='../lesion_detection/margin_crop_rois'
train_abnormalDir ='../fundus_data/cropped_original_fundus_300x300/retina/*.png'
#test_abnormalDir='../lesion_detection/blood_cropped_rois'
test_abnormalDir='../fundus_data/cropped_original_fundus_300x300/retina/Test/*.png'


train_nor=glob.glob(train_normalDir)
print len(train_nor)
test_nor=glob.glob(test_normalDir)
train_abnor=glob.glob(train_abnormalDir)
test_abnor=glob.glob(test_abnormalDir)


train_nor_names=map(lambda path :os.path.split(path)[1] , train_normalDir)
test_nor_names=map(lambda path :os.path.split(path)[1] , test_normalDir)

print len(train_nor_names)
print len(test_nor_names)

