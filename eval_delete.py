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
import glob
import copy
import input

import fundus
import kmeans
import draw_contour
## for mnist dataset ##
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
"""
train_imgs = mnist.train.images.reshape([-1,28,28,1])
train_labs = mnist.train.labels
test_imgs = mnist.test.images.reshape([-1,28,28,1])
test_labs = mnist.test.labels
"""
#for Fundus_300


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

    """
    try:
        print np.shape(vis_abnormal)
        vis_normal=vis_normal.reshape([h,w])
        vis_abnormal = vis_abnormal.reshape([h,w])
        plt.imshow(vis_normal)
        plt.show()
        plt.imshow(vis_abnormal)
        plt.show()
    except Exception as e :
        print e
        pass
    """
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
    saver.save(sess,'./models/vgg_11/blood_and_normal/model')


    return classmap ,sess ,x_



if __name__ =='__main__':
    NORMAL = 0
    ABNORMAL = 1
    print """ ----------------Load ROI Test Data-------------------"""
    print """ -----------------------------------------------------"""
    start = time.time()
    count = 0
    paths = []
    for dir, subdirs, files in os.walk('../lesion_detection/blood_cropped_rois'):
        for file in files:
            path = os.path.join(dir, file)
            paths.append(path)
            count += 1
    print count

    imgs = map(lambda path: np.asarray(Image.open(path)), paths[:2])
    roi_test_imgs = np.asarray(imgs)
    roi_test_labs = np.zeros([len(roi_test_imgs), 2])
    roi_test_labs[:, ABNORMAL] = 1
    print np.shape(roi_test_imgs)
    print np.shape(roi_test_labs)
    print time.time() - start

    print """ ------------- Load Normal Test Data ------------------"""
    print """ -------------------------------------------------------"""
    # normal Data 1000장을 불러온다
    start = time.time()
    paths = []
    count = 0
    for dir, subdirs, files in os.walk('../lesion_detection/bg_cropped_rois'):
        for file in files:
            path = os.path.join(dir, file)
            paths.append(path)
            count += 1
    print count

    imgs = map(lambda path: np.asarray(Image.open(path)), paths[:2])
    bg_test_imgs = np.asarray(imgs)
    bg_test_labs = np.zeros([len(bg_test_imgs), 2])
    bg_test_labs[:, NORMAL] = 1
    print time.time() - start
    print np.shape(bg_test_imgs)
    print np.shape(bg_test_labs)

    test_imgs = np.vstack([roi_test_imgs, bg_test_imgs])
    test_labs = np.vstack([roi_test_labs, bg_test_labs])
    roi_test_imgs = None
    bg_test_imgs = None

    #test_images=np.reshape(test_images,[-1,299,299,3])
    model_path = './models/fundus_500/normal_blood/19/best_acc/step_1300_acc_0.933333396912/model'
    model_path='./models/vgg_11/step_41900_acc_0.900000035763/model'

    #pred=eval(model_path, test_imgs[:],batch_size =1 ,save_root_folder='./activation_map_/blood')
    paths=glob.glob('../lesion_detection/hemo_30_crop/*.png')
    paths=glob.glob('/Users/seongjungkim/Desktop/hemo_30_crop/*.png')
    paths=glob.glob('../retina_original/*.png')
    paths = glob.glob('/Volumes/Seagate Backup Plus Drive/data/fundus/retina_750/*.png')

    save_dir ='./activation_map_/blood_actmap'

    classmap ,sess, x_ = fn( model_path, strides=[1, 1, 1, 1, 1, 1, 1, 1], pool_indices=[0, 1, 2, 3, 5, 7], label=1)

    thres=0.5
    for path in paths[:1] :
        name=os.path.split(path)[1]

        #ori_img=np.asarray(Image.open(path))
        ori_img=Image.open(path).convert('RGB')
        if ori_img.size[0] > 2000: # 이미지가 3000 , 2000 이면 아예 그래픽 카드에 안들어간다 . 그래서 이미지의 크기를 보전하면서 이미지를 줄인다
            pct = 2000 / float(ori_img.size[0])
            ori_img=ori_img.resize( [int(ori_img.size[0]*pct) , int(ori_img.size[1]*pct)])
        ori_img=np.asarray(ori_img) #resize([2000,2000], Image.ANTIALIAS))
        img=ori_img.reshape((1,)+np.shape(ori_img))

        actmap = sess.run(classmap, feed_dict={x_: img})
        actmap = np.squeeze(actmap)
        actmap = np.asarray((map(lambda x: (x - x.min()) / (x.max() - x.min()), actmap)))  # -->why need this?
        h, w = np.shape(actmap)

        # erase value out ot circle
        mask=[]
        img = np.asarray(Image.open(path).convert('RGB'))

        img.setflags(write=True)
        lower_indices=np.where([np.sum(img , axis=2).reshape([-1]) < 5])[1]
        upper_indices = np.where([np.sum(img, axis=2).reshape([-1]) >= 5])[1]
        mask = np.sum(img , axis=2)
        mask=mask.reshape([-1])
        mask[lower_indices] = 0
        mask[upper_indices] = 1

        #copy actmap to binary actmap
        binary_actmap = copy.copy(actmap)
        binary_actmap = binary_actmap .reshape([-1])
        binary_actmap = binary_actmap * mask
        lower_indices = np.where([binary_actmap < thres])[1]
        upper_indices = np.where([binary_actmap >= thres])[1]
        binary_actmap[lower_indices] = 0
        binary_actmap[upper_indices] = 255
        binary_actmap = binary_actmap.reshape([h, w])
        # for cluster
        assert np.shape(ori_img)[:2] == np.shape(actmap)[:2], 'original images {}  actmap images {}'.format(np.shape(ori_img),
                                                                                                    np.shape(actmap))
        h,w,ch=np.shape(ori_img)
        xy=[]
        for ind in upper_indices[:]:
            y = ind / w # 몇 줄에 위치 있는지?
            x = ind % w #
            xy.append([x,y])

        rects=draw_contour.get_rect(ori_img,binary_actmap)
        fig = plt.figure()
        ax=fig.add_subplot(111)
        ax.imshow(ori_img)
        for rect in rects:
            x1,y1,x2,y2=rect
            rect=patches.Rectangle((x1,y1) , x2-x1, y2-y1 , fill=False , edgecolor='r')
            ax.add_patch(rect)
        plt.savefig(os.path.join(save_dir,'drawContour_'+name))

        rects=kmeans.kmeans(xy , 10)
        fig = plt.figure()
        ax=fig.add_subplot(111)
        ax.imshow(ori_img)
        for rect in rects:
            x1,y1,x2,y2=rect
            rect=patches.Rectangle((x1,y1) , x2-x1, y2-y1 , fill=False , edgecolor='r')
            ax.add_patch(rect)
        plt.savefig(os.path.join(save_dir,'kmeans_'+name))
        plt.close()



        #save actmap , original , DrawImage , kmenas_Image
        #plt.imshow(actmap, cmap=plt.cm.jet, alpha=0.5, interpolation='nearest', vmin=0, vmax=1)

        cam.overlay(actmap, ori_img ,save_path=os.path.join(save_dir,name))
        tf.reset_default_graph()

