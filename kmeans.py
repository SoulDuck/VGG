import numpy as np
num_puntos = 2000
conjunto_puntos = []
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans


XY = tf.placeholder(tf.float32 ,shape=[None,2])
kmeans=KMeans(XY,n_cluasters=2,use_mini_batch=True)
all_scores , clusterd_idx ,clustered_centers_init, init_op ,train_op =kmeans.training_graph()

kmeans=tf.identity(kmeans , 'kmeans')



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






def kmeans_with_placeholder(n_clauster):

    vectors=tf.placeholder(dtype=tf.float32 , shape=[None,2] , name='xy_')
    #vectors = tf.constant(xy)
    centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[n_clauster,-1]))

    expanded_vectors = tf.expand_dims(vectors, 0)
    expanded_centroides = tf.expand_dims(centroides, 1)
    assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroides)), 2), 0)
    means = tf.concat([tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where( tf.equal(assignments, c)),[1,-1])),
                                      reduction_indices=[1]) for c in xrange(n_clauster)], 0)
    update_centroides = tf.assign(centroides, means)

    return vectors , update_centroides , centroides ,assignments

    vectors, update_centroides, centroides, assignments=kmeans_with_placeholder(2)


    xy=np.load('/Users/seongjungkim/PycharmProjects/VGG/activation_map_/blood_actmap/xy_8298468_20160813_R.png.npy')


    init_op = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_op)

    print xy
    exit()
    for step in xrange(100):
        _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments],
                                                     feed_dict={vectors: xy})
"""    
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
"""
if __name__ == '__main__':
    kmeans_with_placeholder(2)
