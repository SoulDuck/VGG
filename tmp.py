import time
from PIL import Image
import numpy as np
import glob
import matplotlib.pyplot as plt
import os , copy
def crop_resize_fundus(path):
    debug_flag=False
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

    #plt.imshow(crop_img)

    diameter_height = up[1] - left[1]
    diameter_width = left[0] - up[0]

    crop_img = im.crop((up[0], left[1], left[0] + diameter_width, up[1] + diameter_height))
    end_time = time.time()

    if __debug__ == debug_flag:
        print end_time - start_time
        print np.shape(np_img)

    return crop_img ,path
paths=glob.glob('./Test_Data/cata_test/*.png')
for path in paths[:25]:
    print path
    crop_img, _ = crop_resize_fundus(path=path)
    crop_img=copy.deepcopy(crop_img)
    name=os.path.split(path)[-1]
    plt.imsave(os.path.join('./Test_Data/cata_test_cropped/' , name) ,crop_img)



