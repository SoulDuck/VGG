import numpy as np
from PIL import Image
import tensorflow as tf
import copy
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import pandas as pd


def kmeans(xy , n_clauster):

    vectors = tf.constant(xy)
    centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[n_clauster,-1]))

    expanded_vectors = tf.expand_dims(vectors, 0)
    expanded_centroides = tf.expand_dims(centroides, 1)
    assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroides)), 2), 0)
    means = tf.concat([tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where( tf.equal(assignments, c)),[1,-1])),
                                      reduction_indices=[1]) for c in xrange(n_clauster)], 0)
    update_centroides = tf.assign(centroides, means)
    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)

    for step in xrange(100):
       _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])
    data = {"x": [], "y": [], "cluster": []}
    #print assignment_values

    for i in xrange(len(assignment_values)):
      data["x"].append(xy[i][0])
      data["y"].append(xy[i][1])
      data["cluster"].append(assignment_values[i])



    df = pd.DataFrame(data)
    rects=[]
    for i in range(n_clauster):
        df_ind = df[df['cluster'] ==i]# get specific dataframe
        rect=[df_ind['x'].min(),df_ind['y'].min(),df_ind['x'].max(),df_ind['y'].max()]# x1=df['x'].min() y1=df['y'].min()
                                                                        # x2=df['x'].max(),y2=df['y'].max()
        rects.append(rect)


    #sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
    #plt.savefig('./tmp_kmeans.png')
    #plt.show()

    return rects


def get_rect(ori_img , actmap):
    (_,contours,_) = cv2.findContours(actmap.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(ori_img)
    rects=[]
    for contour in contours :
        rect = cv2.boundingRect(contour)

        if rect[2] < 5 or rect[3]  < 5 :
            continue
        rects.append(rect)
        rect=patches.Rectangle((rect[0],rect[1]) , rect[2],rect[3] , fill=False , edgecolor='r')
        ax.add_patch(rect)

        # rect의 x, y ,w ,h 의 조건에 따라 bounding box 출력
        #if rect[2] > 10 and rect[3] > 10 :
        #    print rect
    return rects



def post_preprocessing(sess, classmap, img_path , thres=0.5 , resize=(512,512)):
    ori_img=Image.open(img_path).convert('RGB')
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
    img = copy.copy(ori_img)

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
    binary_actmap[upper_indices] = 1
    binary_actmap = binary_actmap.reshape([h, w])
    binary_actmap=Image.fromarray(binary_actmap)
    binary_actmap=np.asarray(binary_actmap.resize(resize, Image.ANTIALIAS))


    """
    # for cluster Using K means , uncomment this line
    assert np.shape(ori_img)[:2] == np.shape(actmap)[:2], 'original images {}  actmap images {}'.format(np.shape(ori_img),
                                                                                                np.shape(actmap))
    h,w,ch=np.shape(ori_img)
    xy=[]
    for ind in upper_indices[:]:
        y = ind / w # 몇 줄에 위치 있는지?
        x = ind % w #
        xy.append([x,y])

    rects=kmeans(xy , n_cluster)
    """
    return binary_actmap
