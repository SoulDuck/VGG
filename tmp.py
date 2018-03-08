#-*- coding:utf-8 -*-
import numpy as np
import time
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import fundus
import pickle
import os
import shutil
import PIL
import glob
a=[1,2,3,4,5,6]
print a[-1:]
print a[:-1]

try:
    a=3
    raise ValueError
except:
    pass
print a

"""
a_cls=np.zeros(3)
b_cls=np.ones(3)
a=[[0,1],[0,1]]
b=[[0,1],[1,0]]
print np.ndim(a)
print a_cls==b_cls
print np.argmax( a, axis=1 )



nor_img = Image.open('./tmp/normal_actmap.png')
#plt.imshow(nor_img)
#plt.show()
nor_1_img = Image.open('./tmp/normal_actmap_1.png')
#plt.imshow(nor_1_img)
#plt.show()
test_img = Image.open('./tmp/image_test.png')
#plt.imshow(test_img)
#plt.show()
abnor_img= Image.open('./tmp/abnormal_actmap.png')
#plt.imshow(abnor_img)
#plt.show()

background = test_img.convert("RGBA")
overlay = nor_img.convert("RGBA")
overlay_1 = nor_1_img.convert("RGBA")
overlay_2 = abnor_img.convert("RGBA")
a=[]
a.append(overlay)
plt.imshow(a[0])

overlay.putalpha(128)


plt.imshow(overlay, cmap=plt.cm.jet)
plt.show()

overlay_1=Image.blend(overlay_1 , overlay_2 , 0.5)
plt.imshow(overlay_1, cmap=plt.cm.jet)
plt.show()

overlay_1=Image.blend(overlay , overlay_1 , 0.5)
plt.imshow(overlay_1, cmap=plt.cm.jet)
plt.show()

overlay_img = Image.blend(background, overlay_1, 0.5)
plt.imshow(overlay_img, cmap=plt.cm.jet)
plt.show()
"""
"""
trues=np.load('labels.npy')
preds=np.load('best_preds.npy')
#imgs = np.load('test_images.npy')
trues_cls=np.argmax(trues , axis=1)
preds_cls=np.argmax(preds , axis=1)
print trues_cls[:310]
print preds_cls

tmp=[trues_cls==preds_cls]

tmp=np.squeeze(tmp)
tmp = list(tmp)resize=(299,299)
"""
train_imgs ,train_labs ,train_fnames, test_imgs ,test_labs , test_fnames = fundus.type2(tfrecords_dir='./fundus_300' , onehot=True , resize=(299,299))
"""
print os.walk('./activation_map_')
f = []
for (dirpath, dirnames, filenames) in os.walk('./activation_map_/step_5900_acc_0.841071486473'):
    print dirpath
    filepaths=map(lambda filename : os.path.join(dirpath , filename) , filenames)
    f.extend(filepaths)
"""

##3def tmp(root_folder , target_filenames):
#    os.walk(f)

#####
"""
아래 사용 설명서를읽기 바람

아래 코드는 임형택 선생님에게 코드를 주기 위해서 만든 코드이다 .
테스트 파일에서 (fundus.type1)  을 하면 테스트 이미지 , 테스트 라벨 , 테스트 파일 이름이 나온다 


테스트 파일 이름 리스트를  overlap에다 건내줘서 
중복 이름이 있나 확인한다 30개의 이름이 중복되어 있다 왜 그럴까?
 
 
1. 노말 이미지중에 겹치는게 뭐가 있는지 확인해야 한다 --> abnormal만 겹침 

2. abnormal 이미지중에 cataract_glaucoma 는 cataract , glaucoma 두개 다 들어 가있을 가능성이 매우 높다.

find_images 란 함수에 src_root_dir , target_filenames , save_folder 3가지 파라미터가 있다 
 
src_root_dir 에는 원본 사진들이 들어있다
os.walk() 함수를 이용해 모든 list 들을 가져온다 

그다음에 target_filenames 는 내가 찾고 싶은 filenames 이다 .
src_root_dir 의 모든 파일중에  target_filename 이 있으면 if 구문 으로 들어간다 
그리고 
 


"""
####

def crop_resize_fundus(path):
    debug_flag = False
    """
    file name =1002959_20130627_L.png
    """
    name = path.split('/')[-1]
    start_time = time.time()
    im = Image.open(path)  # Can be many different formats.
    np_img = np.asarray(im)
    mean_pix = np.mean(np_img)
    pix = im.load()
    height, width = im.size  # Get the width and hight of the image for iterating over
    # pix[1000,1000] #Get the RGBA Value of the a pixel of an image
    c_x, c_y = (int(height / 2), int(width / 2))

    for y in range(c_y):
        if sum(pix[c_x, y]) > mean_pix:
            left = (c_x, y)
            break;

    for x in range(c_x):
        if sum(pix[x, c_y]) > mean_pix:
            up = (x, c_y)
            break;

    crop_img = im.crop((up[0], left[1], left[0], up[1]))

    # plt.imshow(crop_img)

    diameter_height = up[1] - left[1]
    diameter_width = left[0] - up[0]

    crop_img = im.crop((up[0], left[1], left[0] + diameter_width, up[1] + diameter_height))
    end_time = time.time()

    if __debug__ == debug_flag:
        print end_time - start_time
        print np.shape(np_img)

    return crop_img, path


def img2np(path , resize):
    img=Image.open(path)
    img=img.resize(resize , PIL.Image.ANTIALIAS)
    img=np.asarray(img)
    return img




def  overlap(list_):
    dict_={}
    for l in list_:
        if not l in dict_:
            dict_[l]=0
        else :
            dict_[l] +=1
    overlap_list=[]
    for k in dict_.keys():
        if dict_[k] > 0 :
            overlap_list.append(k)


    return overlap_list

def find_images(src_root_dir , target_filenames , save_folder):
    """
    :param root_dir:  folder that source original images saved
    :param target_filenames: filename which you want to find from src_root_dir
    :param save_folder: image will save this folder
    :return:
    """
    #이름이 겹치는게 없는지 확인해야 함
    print overlap(target_filenames)
    f=[]
    #assert len(set(target_filenames)) == len(list(target_filenames)) , \
    #'# target filenames set : {} list : {}'.format(len(set(target_filenames)) , len(list(target_filenames)))
    for (dirpath, dirnames, filenames) in os.walk(src_root_dir):
        print dirpath
        filepaths = map(lambda filename: os.path.join(dirpath, filename), filenames)
        f.extend(filepaths)


    for target_name in target_filenames:

        for filepath in f :
            if target_name in filepath:
                if 'normal' in filepath:
                    shutil.copy(src= filepath , dst = os.path.join(save_folder,'normal' , target_name+'.png'))
                else:
                    shutil.copy(src=filepath, dst=os.path.join(save_folder,'abnormal', target_name+'.png'))

if '__main__' == __name__:

    img_paths=glob.glob('./fundus_300/*.jpg')
    print img_paths
    imgs_paths = map(lambda img_path : crop_resize_fundus(img_path) , img_paths)
    imgs=[]
    for i in range(4):
        img=imgs_paths[i][0]
        imgs.append(img)


    #imgs = map(lambda img: Image.fromarray(img), imgs)
    imgs = map(lambda img: img.resize((299,299) , PIL.Image.ANTIALIAS), imgs)
    imgs = map(lambda img: np.asarray(img), imgs)
    print np.shape(imgs)
    np.save('./fundus_300/russian_eyes', imgs )
    for img in imgs:
        plt.imsave('./fundus_300/{}.jpg'.format(i))

    """
    src_root_dir='../fundus_data/original_fundus'
    target_filenames=test_fnames[:]
    save_folder='./original_test_images'
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    if not os.path.isdir(os.path.join(save_folder , 'normal')):
        os.mkdir(os.path.join(save_folder , 'normal'))
    if not os.path.isdir(os.path.join(save_folder , 'abnormal')):
        os.mkdir(os.path.join(save_folder , 'abnormal'))

    find_images(src_root_dir=src_root_dir , target_filenames=target_filenames , save_folder=save_folder)
    """

"""
for i,t in enumerate(tmp):
    img = Image.fromarray(imgs[i])
    if t == True:
        print t , i
        plt.imsave('./out/trues/{}.png'.format(i) , img )

"""

