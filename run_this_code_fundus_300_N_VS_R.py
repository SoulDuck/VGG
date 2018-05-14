#-*- coding:utf-8 -*-
import model
import input
import os
import numpy as np
import argparse
import sys
import tensorflow as tf
import aug
import random
from PIL import Image
import time
import pickle



parser =argparse.ArgumentParser()
#parser.add_argument('--saves' , dest='should_save_model' , action = 'store_true')
#parser.add_argument('--no-saves' , dest='should_save_model', action ='store_false')

parser.add_argument('--optimizer' ,'-o' , type=str ,choices=['sgd','momentum','adam'],help='optimizer')
parser.add_argument('--use_nesterov' , type=bool , help='only for momentum , use nesterov')

parser.add_argument('--aug' , dest='use_aug', action='store_true' , help='augmentation')
parser.add_argument('--no_aug' , dest='use_aug', action='store_false' , help='augmentation')

parser.add_argument('--clahe' , dest='use_clahe', action='store_true' , help='augmentation')
parser.add_argument('--no_clahe' , dest='use_clahe', action='store_false' , help='augmentation')

parser.add_argument('--actmap', dest='use_actmap' ,action='store_true')
parser.add_argument('--no_actmap', dest='use_actmap', action='store_false')

parser.add_argument('--random_crop_resize' , '-r',  type = int  , help='if you use random crop resize , you can choice randdom crop ')

parser.add_argument('--batch_size' ,'-b' , type=int , help='batch size')
parser.add_argument('--max_iter', '-i' , type=int , help='iteration')

parser.add_argument('--l2_loss', dest='use_l2_loss', action='store_true' ,help='l2 loss true or False')
parser.add_argument('--no_l2_loss', dest='use_l2_loss', action='store_false' ,help='l2 loss true or False')
parser.add_argument('--weight_decay', type = float , help='L2 weight decay ')



parser.add_argument('--vgg_model' ,'-m' , choices=['vgg_11','vgg_13','vgg_16', 'vgg_19'])

parser.add_argument('--BN' , dest='use_BN'  , action='store_true' ,   help = 'bn True or not')
parser.add_argument('--no_BN',dest='use_BN' , action = 'store_false', help = 'bn True or not')

parser.add_argument('--data_dir' , help='the folder where the data is saved ' )

parser.add_argument('--init_lr' , type = float , help='init learning rate ')
parser.add_argument('--lr_decay_step' ,type=int , help='decay step for learning rate')

parser.add_argument('--folder_name' ,help='ex model/fundus_300/folder_name/0 .. logs/fundus_300/folder_name/0 , type2/folder_name/0')
args=parser.parse_args()

print 'aug : ' , args.use_aug
print 'actmap : ' , args.use_actmap
print 'use_l2_loss: ' , args.use_l2_loss
print 'weight_decay' , args.weight_decay
print 'BN : ' , args.use_BN
print 'Init Learning rate ' , args.init_lr
print 'Decay step for learning rate, ',args.lr_decay_step

print 'optimizer : ', args.optimizer
print 'use nesterov : ',args.use_nesterov
print 'random crop size : ',args.random_crop_resize
print 'batch size : ',args.batch_size
print 'max iter  : ',args.max_iter
print 'data dir  : ',args.data_dir


def count_trainable_params():
    total_parameters = 0
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parametes = 1
        for dim in shape:
            variable_parametes *= dim.value
        total_parameters += variable_parametes
    print("Total training params: %.1fM" % (total_parameters / 1e6))

def cls2onehot(cls , depth):

    labs=np.zeros([len(cls) , depth])
    for i,c in enumerate(cls):
        labs[i,c]=1
    return labs


def reconstruct_tfrecord_rawdata(tfrecord_path):
    debug_flag_lv0 = True
    debug_flag_lv1 = True
    if __debug__ == debug_flag_lv0:
        print 'debug start | batch.py | class tfrecord_batch | reconstruct_tfrecord_rawdata '

    print 'now Reconstruct Image Data please wait a second'
    reconstruct_image = []
    # caution record_iter is generator
    record_iter = tf.python_io.tf_record_iterator(path=tfrecord_path)

    ret_img_list = []
    ret_lab_list = []
    ret_filename_list = []
    for i, str_record in enumerate(record_iter):
        msg = '\r -progress {0}'.format(i)
        sys.stdout.write(msg)
        sys.stdout.flush()
        example = tf.train.Example()
        example.ParseFromString(str_record)

        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])
        raw_image = (example.features.feature['raw_image'].bytes_list.value[0])
        label = int(example.features.feature['label'].int64_list.value[0])
        filename = (example.features.feature['filename'].bytes_list.value[0])
        image = np.fromstring(raw_image, dtype=np.uint8)
        image = image.reshape((height, width, -1))
        ret_img_list.append(image)
        ret_lab_list.append(label)
        ret_filename_list.append(filename)
    ret_img = np.asarray(ret_img_list)
    ret_lab = np.asarray(ret_lab_list)
    if debug_flag_lv1 == True:
        print ''
        print 'images shape : ', np.shape(ret_img)
        print 'labels shape : ', np.shape(ret_lab)
        print 'length of filenames : ', len(ret_filename_list)
    return ret_img, ret_lab, ret_filename_list





# pickle 형태로 저장되어 있는 데이터를 불러옵니다.
imgs_list=[]
root_dir =args.data_dir

#Load Train imgs ,labs , Test imgs , labs
"""
train_imgs , train_labs , train_fnames = reconstruct_tfrecord_rawdata(os.path.join(root_dir , 'train.tfrecord'))
test_imgs , test_labs , test_fnames = reconstruct_tfrecord_rawdata(os.path.join(root_dir , 'test.tfrecord'))
"""
names = ['normal_train.npy' , 'normal_test.npy' ,'retina_train.npy' , 'retina_test.npy']
normal_train_imgs , normal_test_imgs, abnormal_train_imgs , abnormal_test_imgs,  =\
    map( lambda name : np.load(os.path.join(root_dir ,name)) , names)

NORMAL = 0
ABNORMAL = 1

normal_train_labs=np.zeros([len(normal_train_imgs) , 2])
normal_train_labs[:,NORMAL]=1
abnormal_train_labs=np.zeros([len(abnormal_train_imgs) , 2])
abnormal_train_labs[:,ABNORMAL]=1
normal_test_labs=np.zeros([len(normal_test_imgs) , 2])
normal_test_labs[:,NORMAL]=1
abnormal_test_labs=np.zeros([len(abnormal_test_imgs) , 2])
abnormal_test_labs[:,ABNORMAL]=1


print 'Normal Training Data shape : {}'.format(np.shape(normal_train_imgs))
print 'ABNormal Training Data shape : {}'.format(np.shape(abnormal_train_imgs))
print 'Normal Test Data shape : {}'.format(np.shape(normal_test_imgs))
print 'ABNormal Test Data shape : {}'.format(np.shape(abnormal_test_imgs))

print 'Normal Training Labels shape : {}'.format(np.shape(normal_train_labs))
print 'ABNormal Training Labelsshape : {}'.format(np.shape(abnormal_train_labs))
print 'Normal Test Labelsshape : {}'.format(np.shape(normal_test_labs))
print 'ABNormal Test Labels shape : {}'.format(np.shape(abnormal_test_labs))

# normal 과 abnormal 의 balance 을 맞춥니다
train_imgs = np.vstack([normal_train_imgs , abnormal_train_imgs ,abnormal_train_imgs,abnormal_train_imgs,\
                        abnormal_train_imgs,abnormal_train_imgs,abnormal_train_imgs])
train_labs = np.vstack([normal_train_labs , abnormal_train_labs ,abnormal_train_labs ,abnormal_train_labs,\
                        abnormal_train_labs ,abnormal_train_labs ,abnormal_train_labs])


test_imgs = np.vstack([normal_test_imgs , abnormal_test_imgs])
test_labs = np.vstack([normal_test_labs, abnormal_test_labs])

#train_labs=cls2onehot(train_labs , 2)
#test_labs=cls2onehot(test_labs , 2)

print 'Train Images Shape : {} '.format(np.shape(train_imgs))
print 'Train Labels Shape : {} '.format(np.shape(train_labs))
print 'Test Images Shape : {} '.format(np.shape(test_imgs))
print 'Test Labels Shape : {} '.format(np.shape(test_labs))

# Apply Clahe
if args.use_clahe:
    print 'Apply clahe ....'
    import matplotlib.pyplot as plt
    train_imgs= map(aug.clahe_equalized, train_imgs)
    test_imgs = map(aug.clahe_equalized, test_imgs)
    train_imgs , test_imgs = map(np.asarray , [train_imgs , test_imgs])


#normalize
print np.shape(test_labs)
if np.max(test_imgs) > 1:
    #train_imgs=train_imgs/255.
    test_imgs=test_imgs/255.

print 'test_imgs max :', np.max(test_imgs)


h,w,ch=train_imgs.shape[1:]
print h,w,ch
n_classes=np.shape(train_labs)[-1]
print 'the # classes : {}'.format(n_classes)
x_ , y_ , cam_ind, lr_ , is_training ,global_step = model.define_inputs(shape=[None, h ,w, ch ] , n_classes=n_classes )
logits=model.build_graph(x_=x_ , y_=y_ , cam_ind= cam_ind , is_training=is_training , aug_flag=args.use_aug,\
                         actmap_flag=args.use_actmap  , model=args.vgg_model,random_crop_resize=args.random_crop_resize, \
                         bn = args.use_BN)
lr_op= tf.train.exponential_decay(args.init_lr, global_step , decay_steps=int(args.max_iter / args.lr_decay_step ), decay_rate=0.96,
                                           staircase=False)
train_op, accuracy_op , loss_op , pred_op = \
    model.train_algorithm(args.optimizer, logits=logits, labels=y_, learning_rate=lr_op, l2_loss=args.use_l2_loss,
                          weight_decay=args.weight_decay)

log_count =0;
while True:
    logs_root_path='./logs/{}'.format(args.folder_name )
    try:
        os.makedirs(logs_root_path)
    except Exception as e :
        print e
        pass;
    print logs_root_path

    logs_path=os.path.join( logs_root_path , str(log_count))
    if not os.path.isdir(logs_path):
        os.mkdir(logs_path)
        break;
    else:
        log_count+=1
sess, saver , summary_writer =model.sess_start(logs_path)


model_count =0;
while True:
    models_root_path='./models/{}'.format(args.folder_name)
    try:
        os.makedirs(models_root_path)
    except Exception as e:
        print e
        pass;
    models_path=os.path.join(models_root_path , str(model_count))
    if not os.path.isdir(models_path):
        os.mkdir(models_path)
        break;
    else:
        model_count+=1


best_acc_root = os.path.join(models_path, 'best_acc')
best_loss_root = os.path.join(models_path, 'best_loss')
os.mkdir(best_acc_root)
os.mkdir(best_loss_root)

print 'Logs savedir: {}'.format(logs_path)
print 'Model savedir : {}'.format(models_path)
min_loss = 1000.
max_acc = 0.

max_iter=args.max_iter
ckpt=100
batch_size=args.batch_size
start_time=0
train_acc=0
train_val=0

share=len(test_labs)/batch_size
remainder=len(test_labs)/batch_size

train_acc=0.
train_loss=1000.


def show_progress(step, max_iter):
    msg = '\r progress {}/{}'.format(step, max_iter)
    sys.stdout.write(msg)
    sys.stdout.flush()


count_trainable_params()
for step in range(max_iter):
    if step % ckpt==0:
        """ #### testing ### """
        print '### Testing ###'
        test_fetches = [ accuracy_op, loss_op, pred_op , lr_op]
        val_acc_mean , val_loss_mean , pred_all = [] , [] , []
        for i in range(share): #여기서 테스트 셋을 sess.run()할수 있게 쪼갭니다
            test_feedDict = {x_: test_imgs[i * batch_size:(i + 1) * batch_size],
                             y_: test_labs[i * batch_size:(i + 1) * batch_size], is_training: False, global_step: step}
            val_acc, val_loss, pred, learning_rate = sess.run(fetches=test_fetches, feed_dict=test_feedDict)
            val_acc_mean.append(val_acc)
            val_loss_mean.append(val_loss)
            pred_all.append(pred)
        val_acc_mean=np.mean(np.asarray(val_acc_mean))
        val_acc_mean=np.mean(np.asarray(val_acc_mean))
        val_loss_mean=np.mean(np.asarray(val_loss_mean))
        if val_acc_mean > max_acc: #best acc
            max_acc=val_acc_mean
            print 'max acc : {}'.format(max_acc)
            best_acc_folder=os.path.join( best_acc_root, 'step_{}_acc_{}'.format(step , max_acc))
            os.mkdir(best_acc_folder)
            saver.save(sess=sess,save_path=os.path.join(best_acc_folder  , 'model'))
        print 'Step : {} '.format(step)
        print 'Learning Rate : {} '.format(learning_rate)
        print 'Train acc : {} Train loss : {}'.format( train_acc , train_loss)
        print 'validation acc : {} loss : {}'.format( val_acc_mean, val_loss_mean )
        # add learning rate summary
        summary=tf.Summary(value=[tf.Summary.Value(tag='learning_rate' , simple_value = float(learning_rate))])
        summary_writer.add_summary(summary, step)

        model.write_acc_loss( summary_writer, 'validation', loss=val_loss_mean, acc=val_acc_mean, step=step)
        model_path=os.path.join(models_path, str(step))
        os.mkdir(model_path) # e.g) models/fundus_300/100/model.ckpt or model.meta
        #saver.save(sess=sess,save_path=os.path.join(model_path,'model' , folder_name))
    """ #### training ### """
    train_fetches = [train_op, accuracy_op, loss_op ]
    batch_xs, batch_ys , batch_fname= input.next_batch(batch_size, train_imgs, train_labs )
    if args.use_aug:
        batch_xs=aug.random_rotate_90(batch_xs) # random 으로 90 180 , 270 , 360 도를 회전합니다.
    batch_xs=batch_xs/255.
    train_feedDict = {x_: batch_xs, y_: batch_ys, cam_ind: ABNORMAL, lr_: learning_rate, is_training: True,
                      global_step: step}
    _ , train_acc, train_loss = sess.run( fetches=train_fetches, feed_dict=train_feedDict )
    #print 'train acc : {} loss : {}'.format(train_acc, train_loss)
    model.write_acc_loss(summary_writer ,'train' , loss= train_loss , acc=train_acc  ,step= step)



