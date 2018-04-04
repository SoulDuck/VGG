#-*- coding: utf-8 -*-

"""
    model_path = './models/vgg_11_Calc_N_VS_ABN_no_BN_AUG/0/best_acc/step_4000_acc_0.641666710377/model' #calcium score
    test_imgs=np.load('./Test_Data/calc_fundus/test_abnormal_img_300.npy')
    save_dir = './activation_maps/calc_fundus_300_ori'
"""
#-*- coding: utf-8 -*-
import tensorflow as tf
import cam
import numpy as np
import os
import matplotlib
from PIL import Image
import time
#if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    #matplotlib.use('Agg')
#    pass;
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import glob
import copy
import input

import fundus
import kmeans
import draw_contour
## for mnist dataset ##


def reverse_colourmap(cmap, name = 'my_cmap_r'):
    """
    In:
    cmap, name
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """
    reverse = []
    k = []

    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:
            data.append((1-t[0],t[2],t[1]))
        reverse.append(sorted(data))

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r

def max_pool(name,x , k=3 , s=2 , padding='SAME'):
    with tf.variable_scope(name) as scope:
        if __debug__ ==True:

            layer=tf.nn.max_pool(x , ksize=[1,k,k,1] , strides=[1,s,s,1] , padding=padding)
            print 'layer name :', name
            print 'layer shape :', layer.get_shape()
    return layer

def fn(image,model_path, strides,pool_indices,label):
    def _restore_WB(name):
        w = graph.get_tensor_by_name(name + '/kernel:0')
        b = graph.get_tensor_by_name(name + '/bias:0')
        return w,b
    #1.학습 된걸 복원 한다 .
    im_size=np.shape(image)[1:3]
    sess=tf.Session()
    saver = tf.train.import_meta_graph(meta_graph_or_file=model_path + '.meta')
    saver.restore(sess, model_path)
    x_ = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='input')


    graph=tf.get_default_graph()
    layer=x_
    for i in range(8):
        print i
        w,b=_restore_WB('conv_{}'.format(i))
        layer=tf.nn.conv2d(layer, filter=w, strides=[1,strides[i],strides[i],1], padding='SAME')+b
        if i in pool_indices:
            layer = max_pool(name='max_pool_{}'.format(i), x=layer, k=3, s=2)

    top_conv=tf.identity(layer, name='top_conv')
    w = graph.get_tensor_by_name('gap' + '/w:0')


    im_height =tf.shape(x_)[1];
    im_width= tf.shape(x_)[2];
    name='gap'
    x=top_conv
    #im_height = im_size[0] ; im_width = im_size[1]
    out_ch = int(top_conv.get_shape()[-1])
    conv_resize = tf.image.resize_bilinear(x, [im_height, im_width])

    with tf.variable_scope(name, reuse=True) as scope:
        label_w = tf.gather(tf.transpose(w), label)
        label_w = tf.reshape(label_w, [-1, out_ch, 1])
    conv_resize = tf.reshape(conv_resize, [-1, im_height * im_width, out_ch])
    classmap = tf.matmul(conv_resize, label_w, name='classmap')
    classmap = tf.reshape(classmap, [-1, im_height, im_width], name='classmap_reshape')

    saver=tf.train.Saver()
    saver.save(sess,'./saved_models/classmap')

    return classmap
def get_acc(preds , trues):
    #onehot vector check
    np.ndim(preds) == np.ndim(trues) , 'predictions and True Values has same shape and has to be OneHot Vector'
    if np.ndim(preds) == 2:
        preds_cls =np.argmax(preds , axis=1)
        trues_cls = np.argmax(trues, axis=1)

    else:
        preds_cls=preds
        trues_cls = trues
    acc=np.sum([preds_cls == trues_cls])/float(len(preds_cls))
    return acc





def eval(model_path ,test_images , batch_size  , save_root_folder):
    print 'eval'
    b,h,w,c=np.shape(test_images)

    if np.max(test_images) > 1:
        test_images = test_images / 255.
    sess = tf.Session()

    saver = tf.train.import_meta_graph(meta_graph_or_file=model_path+'.meta') #example model path ./models/fundus_300/5/model_1.ckpt
    saver.restore(sess, save_path=model_path) # example model path ./models/fundus_300/5/model_1.ckpt

    x_ = tf.get_default_graph().get_tensor_by_name('x_:0')
    y_ = tf.get_default_graph().get_tensor_by_name('y_:0')
    pred_ = tf.get_default_graph().get_tensor_by_name('softmax:0')
    is_training_=tf.get_default_graph().get_tensor_by_name('is_training:0')
    top_conv = tf.get_default_graph().get_tensor_by_name('top_conv:0')
    logits = tf.get_default_graph().get_tensor_by_name('logits:0')
    cam_ = tf.get_default_graph().get_tensor_by_name('classmap:0')
    cam_ind = tf.get_default_graph().get_tensor_by_name('cam_ind:0')
    cam.eval_inspect_cam(sess, cam_, cam_ind ,top_conv, test_images[:], x_, y_, is_training_,
                                                    logits,save_root_folder)


    share=len(test_images)/batch_size
    print share
    remainder=len(test_images)%batch_size
    predList=[]
    for s in range(share):
        pred = sess.run(pred_ , feed_dict={x_ : test_images[s*batch_size:(s+1)*batch_size],is_training_:False})
        print 'pred_ ' ,pred
        predList.extend(pred)
    if remainder !=0:
        pred = sess.run(pred_, feed_dict={x_: test_images[-1*remainder:], is_training_: False})
        predList.extend(pred)

    assert len(predList) == len(test_images) , '{} {}'.format(len(predList) , len(test_images))
    tf.reset_default_graph()
    print 'pred sample ',predList[:1]
    return np.asarray(predList)


def fn(model_path, strides,pool_indices,label):
    def _restore_WB(name):
        w = graph.get_tensor_by_name(name + '/kernel:0')
        b = graph.get_tensor_by_name(name + '/bias:0')
        return w,b
    #1.학습 된걸 복원 한다 .
    sess=tf.Session()
    saver = tf.train.import_meta_graph(meta_graph_or_file=model_path + '.meta')
    saver.restore(sess, model_path)
    x_ = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='input')


    graph=tf.get_default_graph()
    layer=x_
    for i in range(8):
        print i
        w,b=_restore_WB('conv_{}'.format(i))
        layer=tf.nn.conv2d(layer, filter=w, strides=[1,strides[i],strides[i],1], padding='SAME')+b
        if i in pool_indices:
            layer = max_pool(name='max_pool_{}'.format(i), x=layer, k=3, s=2)

    top_conv=tf.identity(layer, name='top_conv')
    w = graph.get_tensor_by_name('gap' + '/w:0')


    im_height =tf.shape(x_)[1];
    im_width= tf.shape(x_)[2];
    name='gap'
    x=top_conv
    #im_height = im_size[0] ; im_width = im_size[1]
    out_ch = int(top_conv.get_shape()[-1])
    conv_resize = tf.image.resize_bilinear(x, [im_height, im_width])

    with tf.variable_scope(name, reuse=True) as scope:
        label_w = tf.gather(tf.transpose(w), label)
        label_w = tf.reshape(label_w, [-1, out_ch, 1])
    conv_resize = tf.reshape(conv_resize, [-1, im_height * im_width, out_ch])
    classmap = tf.matmul(conv_resize, label_w, name='classmap')
    classmap = tf.reshape(classmap, [-1, im_height, im_width], name='classmap_reshape')

    saver=tf.train.Saver()
    saver.save(sess,'./models/vgg_11/retina_and_normal/model')


    return classmap ,sess ,x_



if __name__ =='__main__':

    #test_images=np.reshape(test_images,[-1,299,299,3])
    model_path = './models/vgg_11/step_9600_acc_0.700000047684/model' #calcium score
    test_nor_imgs = np.load('./Test_Data/fundus300_0_10_300_inf/test_normal_img_300_0_10.npy')
    test_abnor_imgs=np.load('./Test_Data/fundus300_0_10_300_inf/test_abnormal_img_300_300_inf.npy')
    test_imgs=np.vstack([test_nor_imgs , test_abnor_imgs])

    test_imgs=np.reshape(test_imgs , list(np.shape(test_imgs)) + [1,])
    print np.shape(test_nor_imgs)
    print np.shape(test_abnor_imgs)
    print np.shape(test_imgs)
    exit()
    preds=eval(model_path , test_imgs[:] , batch_size=3 , save_root_folder='./activation_maps/calc_fundus_300_ori')
    np.save('cal_preds.npy' , preds)

    save_dir = './activation_maps/calc_fundus_300_ori'





    classmap ,sess, x_ = fn( model_path, strides=[1, 1, 1, 1, 1, 1, 1, 1], pool_indices=[0, 1, 2, 3, 5, 7], label=1)

    thres=0.5
    limit=None
    for  img in test_imgs[:limit]:

        #ori_img=np.asarray(Image.open(path))
        if ori_img.size[0] > 2000: # 이미지가 3000 , 2000 이면 아예 그래픽 카드에 안들어간다 . 그래서 이미지의 크기를 보전하면서 이미지를 줄인다
            pct = 2000 / float(ori_img.size[0])
            ori_img=ori_img.resize( [int(ori_img.size[0]*pct) , int(ori_img.size[1]*pct)])
        ori_img=np.asarray(ori_img) #resize([2000,2000], Image.ANTIALIAS))
        img=ori_img.reshape((1,)+np.shape(ori_img))
        img_h, img_w = ori_img.shape[:2]
        actmap = sess.run(classmap, feed_dict={x_: img/255.})
        actmap = np.squeeze(actmap)
        actmap = np.asarray((map(lambda x: (x - x.min()) / (x.max() - x.min()), actmap)))  # -->why need this?
        overlay = cam.overlay(actmap, ori_img, save_path='tmp_overlay.png', factor=0.1)
        cmap=reverse_colourmap(plt.cm.jet) # reverse actmap Red -> Green - > Blue --> Blue --> Green --> Red
        actmap=cmap(actmap)
        plt.imsave(fname='delete_me.png',arr=actmap)
        actmap=Image.open('delete_me.png').convert('RGB')
        os.remove('delete_me.png')
        actmap=np.asarray(actmap)
        #plt.imshow(actmap, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest', vmin=0, vmax=1)
        actmap=copy.copy(actmap)

        actmap[:, :, 0] = np.zeros(np.shape(actmap)[:2]) # Red 을 지운다
        actmap[:, :, 1] = np.zeros(np.shape(actmap)[:2]) # Green 을 지운다
        plt.imsave(arr=actmap , fname =os.path.join(save_dir,'{}_actmap.png'.format(name)))
        plt.show()
        plt.imsave(arr=ori_img, fname =os.path.join(save_dir,'{}_ori.png'.format(name)))
        plt.show()

        # Mask
        flatted_actmap_b=actmap[:, :, 2].reshape(-1) #
        actmap_b_indices=np.where([flatted_actmap_b > 50])[1] # 왜 50 이상만 뽑는거지? rgb
        #flatted_actmap_g = actmap[:, :, 1].reshape(-1)
        #actmap_g_indices = np.where([flatted_actmap_g > 50])[1]

        # indices_rg 의 목표는 actmap과 혼합된 이미지를 보존하는 것이다 # rg = Red Green 을 뜻한다.
        indices_b = np.hstack([actmap_b_indices]) # 만약 초록색을 추가하기 원한다면 actmap_g_indices 이걸 추가하면 된다.
        indices_b = np.asarray(list(set(indices_b)))

        # Get original image pixels from indices_b
        flatted_ori_img=ori_img.reshape([-1,3])
        flatted_ori_img=flatted_ori_img.copy()
        flatted_ori_img[indices_b]=np.array([0,0,0]) #

        # Save Masked original image
        masked_ori_img = flatted_ori_img.reshape([img_h, img_w, 3])

        # Get rev_indices_rg
        rev_indices_b=set(range(img_h*img_w))
        rev_indices_b=rev_indices_b.difference(indices_b)
        assert img_h*img_w==len(rev_indices_b) + len(indices_b)

        # For getting rid of margin , extract indices
        grey_ori_img=np.sum(ori_img , axis=2)
        flatted_grey_ori_img=grey_ori_img.reshape(-1)
        maring_indices=set(np.where([flatted_grey_ori_img< 10])[1])
        rev_indices_b = list( maring_indices | rev_indices_b )

        # Get Part of actmap from rev_indices_rg
        flatted_overlay = overlay.reshape([-1, 3])
        flatted_overlay= flatted_overlay.copy()
        flatted_overlay[rev_indices_b] = np.array([0, 0, 0])  ##중요

        # Save Masked Actmap image
        masked_actmap=flatted_overlay.reshape([img_h,img_w,3])
        blended_image = masked_actmap + masked_ori_img
        plt.imsave(arr=blended_image, fname =os.path.join(save_dir,'{}_blend.png'.format(name)))