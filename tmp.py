import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import fundus

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

trues=np.load('labels.npy')
preds=np.load('best_preds.npy')
#imgs = np.load('test_images.npy')
trues_cls=np.argmax(trues , axis=1)
preds_cls=np.argmax(preds , axis=1)
print trues_cls[:310]
print preds_cls
tmp=[trues_cls==preds_cls]

tmp=np.squeeze(tmp)
tmp = list(tmp)


resize=(299,299)
train_imgs ,train_labs ,train_fnames, test_imgs ,test_labs , test_fnames = fundus.type2(tfrecords_dir='./fundus_300' , onehot=True , resize=resize)


"""
for i,t in enumerate(tmp):
    img = Image.fromarray(imgs[i])
    if t == True:
        print t , i
        plt.imsave('./out/trues/{}.png'.format(i) , img )

"""

