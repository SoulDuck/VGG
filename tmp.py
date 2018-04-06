import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import aug

def random_rotate(img):
    debug_flag = False

    ### usage: map(random_rotate , images) ###
    ind = random.randint(0, 180)
    minus = random.randint(0, 1)
    minus = bool(minus)
    if minus == True:
        ind = ind * -1
    img = img.rotate(ind)
    img = np.asarray(img)

    # image type is must be PIL
    if __debug__ == debug_flag:
        print ind
        plt.imshow(img)
        plt.show()
    img = img
    return img


def clahe_equalized(img):
    img = img.copy()
    if len(np.shape(img)) == 2:  # grey  image (h,w)
        img=np.asarray(img).reshape(list(np.shape) + [1] )


    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    if img.shape[-1] ==3: # if color shape
        for i in range(3):
            img[:, :, i]=clahe.apply(np.array(img[:,:,i], dtype=np.uint8))
    elif img.shape[-1] ==1: # if Greys,
        img = clahe.apply(np.array(img[:,:,0], dtype = np.uint8))
    return img


## py_func test 1
def fn(np_img):
    ret_imgs = tf.py_func(clahe_equalized , [np_img] , [tf.uint8])
    return tf.convert_to_tensor(ret_imgs) , np_img

imgs=[]
for i in range(5):
    img_grey=np.asarray(Image.open('./sample_image_grey.png').convert('RGB'))
    imgs.append(img_grey)
imgs=np.asarray(imgs)
print np.shape(imgs)
batch_xs=aug.random_rotate_90(imgs)

plt.imshow(batch_xs[0])
plt.show()
plt.imshow(batch_xs[1])
plt.show()
plt.imshow(batch_xs[2])
plt.show()
plt.imshow(batch_xs[3])
plt.show()




