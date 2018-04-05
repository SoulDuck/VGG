import numpy as np
import random
from PIL import Image
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

a=[1,2,3,4]


b,c,d,f=a
print b,c,d,f
exit()

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


img_grey=Image.open('./sample_image_grey.png').convert('RGB')
img_color=Image.open('./sample_image_color.png').convert('RGB')
img = np.expand_dims(img_color , 0)
imgs = np.vstack([img , img])
imgs=imgs/255.
imgs =np.rot90(imgs , k=3 , axes=(1,2))
plt.imshow(imgs[0])
plt.show()
plt.imshow(imgs[1])
plt.show()


plt.imshow(img)
plt.show()
img_tensor = tf.Variable(img)
clahe_fundus , ori_fundus = fn(img_tensor)

sess=tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

clahe  , ori =sess.run([clahe_fundus, ori_fundus])
print np.shape(clahe[0])
print np.shape(clahe)
fig = plt.figure()
plt.imshow(clahe[0])
plt.show()


img =np.rot90(img , 1 , axis=1)




