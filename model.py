import cnn
import tensorflow
import numpy as np
import cam
import aug
import tensorflow as tf

def dropout( _input , is_training , keep_prob=0.8):
    if keep_prob < 1:
        output = tf.cond(is_training, lambda: tf.nn.dropout(_input, keep_prob), lambda: _input)
    else:
        output = _input
    return output


def batch_norm( _input , is_training):
    output = tf.contrib.layers.batch_norm(_input, scale=True, \
                                          is_training=is_training, updates_collections=None)
    return output

def weight_variable_msra(shape , name):
    return tf.get_variable(name=name , shape=shape , initializer=tf.contrib.layers.variance_scaling_initializer())
def weight_variable_xavier( shape , name):
    return tf.get_variable(name=name , shape=shape , initializer=tf.contrib.layers.xavier_initializer())
def bias_variable(shape  , name='bias' ):
    initial=tf.constant(0.0 , shape=shape)
    return tf.get_variable(name,initializer=initial)
def conv2d_with_bias(_input , out_feature , kernel_size , strides , padding):
    in_feature=int(_input.get_shape()[-1])
    kernel=weight_variable_msra([kernel_size,kernel_size,in_feature, out_feature] , name='kernel')
    layer=tf.nn.conv2d(_input, kernel, strides, padding) + bias_variable(shape=[out_feature])
    layer=tf.nn.relu(layer)
    print layer
    return layer

def fc_with_bias(_input , out_features ):
    in_fearues=int(_input.get_shape()[-1])
    kernel=weight_variable_xavier([in_fearues , out_features] , name='kernel')
    layer =tf.matmul(_input, kernel) + bias_variable(shape=[out_features])
    print layer
    return layer

def avg_pool( _input , k ):
    ksize=[1,k,k,1]
    strides=[1,k,k,1]
    padding='VALID'
    output=tf.nn.avg_pool(_input , ksize ,strides,padding)
    return output


def fc_layer(_input ,out_feature , act_func='relu' , dropout='True' ):
    assert len(_input.get_shape()) == 2 , len(_input.get_shape())
    in_features=_input.get_shape()[-1]
    w = weight_variable_xavier([in_features, out_feature], name='W')
    b = bias_variable(shape=out_feature)
    layer= tf.matmul(_input, w) + b
    if act_func =='relu':
        layer=tf.nn.relu(layer)
    return layer


def fc_layer_to_clssses(_input , n_classes):

    in_feature=int(_input.get_shape()[-1])
    W=weight_variable_xavier([in_feature, n_classes] , name ='W')
    bias = bias_variable([n_classes])
    logits=tf.matmul(_input, W)+bias
    return logits
