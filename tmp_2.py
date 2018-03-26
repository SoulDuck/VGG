#-*- coding:utf-8 -*-
import tensorflow as tf
import glob
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os
img_dir='/home/mediwhale/PycharmProjects/vgg/delete_me'
paths=glob.glob(os.path.join(img_dir , '*.png'))
names = []

#get name  e.g) 12345.png
names = []
for path in paths:
    if not 'ori.png' in os.path.split(path)[1]:
        name=os.path.split(path)[1]
        names.append(name)

print '# names : {}'.format(len(names))

actmap_paths=[]
ori_paths=[]
for name in names:
    print name
    #get actmap image
    actmap_path=os.path.join(img_dir , name)
    ori_path = os.path.join(img_dir, name.replace('.png' , '_ori.png'))
    actmap_img , ori_img =map(lambda path : np.asarray(Image.open(path).convert('RGB')) , [actmap_path , ori_path])

    assert np.shape(actmap_img ) ==  np.shape(ori_img)
    h,w,ch=np.shape(actmap_img)

    grey_ori_img=np.sum(ori_img, axis = 2)
    flatted_ori_img = map(lambda np_img: np_img.reshape(-1), [ grey_ori_img])
    # extract background on original image
    masked= [flatted_ori_img[0] > 10][0]
    # overlay mask on activation image
    image=np.zeros([h,w,ch])
    for i in range(3):
        assert len(actmap_img[:, :, i].reshape(-1)) == len(masked)
        masked_img=(masked * actmap_img[:, :, i].reshape(-1))
        masked_img=np.reshape(masked_img , [h,w])
        image[:,:,i] = masked_img

    plt.imsave(arr= image/255. ,fname = os.path.join(img_dir , name.replace('.png' , '_masked_actmap.png')))

    #masked_imgs=masked_imgs.reshape([h,w,ch])










#이름을 얻고 그다음에는
for name in names:
    img = Image.open(path)

    #actmap ima
    img = Image.open(path)


    img=Image.open(path)
    img=np.asarray(img)
    h,w,ch=np.shape(img)
    flatted_img=img.reshape(-1)
    #background
    bg_indices=np.where([flatted_img< 10])[1]
