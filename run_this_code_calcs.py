#-*- coding:utf-8 -*-
import model
import input
import os
import fundus
import numpy as np
import argparse
import sys
import tensorflow as tf
import aug
import random
from PIL import Image
import time


"""
여기서는 Normal VS Retina 을 학습시킨다.
여기서는 Normal Vs Glaucoma 을 학습시킨다. Glaucoma 는 정상과 비정상을 잘 구별하기 위해 데이터를 좀 선별해야 한다 
여기서는 Normal Vs Cataract 을 학습시킨다. Cataract 도 구별을 좀 해야 한다.

"""
parser =argparse.ArgumentParser()
#parser.add_argument('--saves' , dest='should_save_model' , action = 'store_true')
#parser.add_argument('--no-saves' , dest='should_save_model', action ='store_false')

parser.add_argument('--optimizer' ,'-o' , type=str ,choices=['sgd','momentum','adam'],help='optimizer')
parser.add_argument('--use_nesterov' , type=bool , help='only for momentum , use nesterov')

parser.add_argument('--aug' , dest='use_aug', action='store_true' , help='augmentation')
parser.add_argument('--no_aug' , dest='use_aug', action='store_false' , help='augmentation')

parser.add_argument('--actmap', dest='use_actmap' ,action='store_true')
parser.add_argument('--no_actmap', dest='use_actmap', action='store_false')

parser.add_argument('--random_crop_resize' , '-r',  type = int  , help='if you use random crop resize , you can choice randdom crop ')

parser.add_argument('--batch_size' ,'-b' , type=int , help='batch size')
parser.add_argument('--max_iter', '-i' , type=int , help='iteration')

parser.add_argument('--l2_loss', dest='use_l2_loss', action='store_true' ,help='l2 loss true or False')
parser.add_argument('--no_l2_loss', dest='use_l2_loss', action='store_false' ,help='l2 loss true or False')

parser.add_argument('--vgg_model' ,'-m' , choices=['vgg_11','vgg_13','vgg_16', 'vgg_19'])

parser.add_argument('--BN' , dest='use_BN'  , action='store_true' ,   help = 'bn True or not')
parser.add_argument('--no_BN',dest='use_BN' , action = 'store_false', help = 'bn True or not')

parser.add_argument('--folder_name' ,help='ex model/fundus_300/folder_name/0 .. logs/fundus_300/folder_name/0 , type2/folder_name/0')
args=parser.parse_args()

print 'aug : ' , args.use_aug
print 'actmap : ' , args.use_actmap
print 'use_l2_loss: ' , args.use_l2_loss
print 'BN : ' , args.use_BN

print 'optimizer : ', args.optimizer
print 'use nesterov : ',args.use_nesterov
print 'random crop size : ',args.random_crop_resize
print 'batch size : ',args.batch_size
print 'max iter  : ',args.max_iter



def _load_images_labels(dir , label ,limit , random_flag):
    start = time.time()
    paths = []

    for dir, subdirs, files in os.walk(dir):
        for file in files:
            path = os.path.join(dir, file)
            paths.append(path)
    if  random_flag is True:
        indices = random.sample(range(len(paths)), limit)
        paths = np.asarray(paths)[indices]
    imgs=map(lambda path : np.asarray(Image.open(path)) , paths[:limit])
    imgs=np.asarray(imgs)
    labs=np.zeros([len(imgs),2])
    labs[:,label ]=1
    return imgs , labs

NORMAL=0
ABNORMAL =1
#blood 500 Image을 불러온다
start=time.time()
#train_normalDir ='../lesion_detection/cropped_bg_500_clahe/'
train_normalDir ='../fundus_data/cropped_original_fundus_300x300/normal_0'

#test_normalDir='../lesion_detection/bg_cropped_rois'
test_normalDir ='../fundus_data/cropped_original_fundus_300x300/normal_0/Test'

#train_abnormalDir ='../lesion_detection/margin_crop_rois'
#test_abnormalDir='../lesion_detection/blood_cropped_rois'

root_dir='/home/mediwhale-5/PythonProjects/vgg/Test_Data/fundus300_0_10_300_inf'
train_normal_imgs=np.load(os.path.join(root_dir ,'train_normal_img_300_0_10.npy'))
train_abnormal_imgs=np.load(os.path.join(root_dir ,'train_abnormal_img_300_300_inf.npy'))
test_normal_imgs=np.load(os.path.join(root_dir ,'test_normal_img_300_0_10.npy'))
test_abnormal_imgs=np.load(os.path.join(root_dir ,'test_abnormal_img_300_300_inf.npy'))

train_normal_labs=np.zeros([len(train_normal_imgs) , 2 ])
train_abnormal_labs=np.zeros([len(train_abnormal_imgs) , 2 ])
test_normal_labs=np.zeros([len(test_normal_imgs) , 2 ])
test_abnormal_labs=np.zeros([len(test_abnormal_imgs) , 2 ])
train_normal_imgs , test_normal_imgs , train_abnormal_imgs , test_abnormal_imgs=\
    map(lambda imgs: imgs.reshape([-1,300,300,1]) , [train_normal_imgs , test_normal_imgs , train_abnormal_imgs , test_abnormal_imgs])
train_normal_labs[:,0]=1
test_normal_labs[:,0]=1
train_abnormal_labs[:,1]=1
test_abnormal_labs[:,1]=1

print np.shape(test_normal_imgs)
print np.shape(train_normal_imgs)
print np.shape(test_abnormal_imgs)
print np.shape(train_abnormal_imgs)

train_imgs=np.vstack([train_normal_imgs , train_abnormal_imgs])
train_labs=np.vstack([train_normal_labs , train_abnormal_labs])
train_normal_imgs=None
train_abnormal_imgs=None

test_imgs=np.vstack([test_normal_imgs , test_abnormal_imgs])
test_labs=np.vstack([test_normal_labs, test_abnormal_labs])
test_normal_imgs=None
test_abnormal_imgs=None


#normalize
print np.shape(test_labs)
if np.max(train_imgs) > 1:
    #train_imgs=train_imgs/255.
    test_imgs=test_imgs/255.
    print 'train_imgs max :',np.max(train_imgs)
    print 'test_imgs max :', np.max(test_imgs)

h,w,ch=train_imgs.shape[1:]
print h,w,ch
n_classes=np.shape(train_labs)[-1]
print 'the # classes : {}'.format(n_classes)
x_ , y_ , cam_ind, lr_ , is_training = model.define_inputs(shape=[None, h ,w, ch ] , n_classes=n_classes )





logits=model.build_graph(x_=x_ , y_=y_ , cam_ind= cam_ind , is_training=is_training , aug_flag=args.use_aug, \
                         actmap_flag=args.use_actmap  , model=args.vgg_model,random_crop_resize=args.random_crop_resize , bn = args.use_BN)

if args.optimizer=='sgd':
    train_op, accuracy_op , loss_op , pred_op = model.train_algorithm_grad(logits=logits,labels=y_ , learning_rate=lr_ ,
                                                                           l2_loss=args.use_l2_loss)
if args.optimizer=='momentum':
    train_op, accuracy_op, loss_op, pred_op = model.train_algorithm_momentum(logits=logits, labels=y_,
                                                                             learning_rate=lr_,
                                                                             use_nesterov=args.use_nesterov , l2_loss=args.use_l2_loss)
if args.optimizer == 'adam':
    train_op, accuracy_op, loss_op, pred_op = model.train_algorithm_adam(logits=logits, labels=y_, learning_rate=lr_,
                                                                         l2_loss=args.use_l2_loss)


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

for step in range(max_iter):
    def show_progress(step, max_iter):
        msg = '\r progress {}/{}'.format(i, max_iter)
        sys.stdout.write(msg)
        sys.stdout.flush()
    #### learning rate schcedule
    if step < 5000:
        learning_rate = 0.001
    elif step < 45000:
        learning_rate = 0.0005
    elif step < 60000:
        learning_rate = 0.0001
    elif step < 120000:
        learning_rate = 0.0001
    else:
        learning_rate = 0.00001
        ####
    if step % ckpt==0:
        """ #### testing ### """
        print 'test'
        test_fetches = [ accuracy_op, loss_op, pred_op ]
        val_acc_mean , val_loss_mean , pred_all = [] , [] , []
        for i in range(share): #여기서 테스트 셋을 sess.run()할수 있게 쪼갭니다
            test_feedDict = { x_: test_imgs[i*batch_size:(i+1)*batch_size], y_: test_labs[i*batch_size:(i+1)*batch_size],  is_training: False }
            val_acc ,val_loss , pred = sess.run( fetches=test_fetches, feed_dict=test_feedDict )
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
        if val_loss_mean < min_loss: # best loss
            min_loss = val_loss_mean
            print 'min loss : {}'.format(min_loss)
            best_loss_folder = os.path.join(best_loss_root, 'step_{}_loss_{}'.format(step, min_loss ))
            os.mkdir(best_loss_folder)
            saver.save(sess=sess,save_path=os.path.join(best_loss_folder, 'model'))
        print 'Learning Rate : {} '.format(learning_rate)
        print 'Train acc : {} Train loss : {}'.format( train_acc , train_loss)
        print 'validation acc : {} loss : {}'.format( val_acc_mean, val_loss_mean )
        model.write_acc_loss( summary_writer, 'validation', loss=val_loss_mean, acc=val_acc_mean, step=step)
        model_path=os.path.join(models_path, str(step))
        os.mkdir(model_path) # e.g) models/fundus_300/100/model.ckpt or model.meta
        #saver.save(sess=sess,save_path=os.path.join(model_path,'model' , folder_name))
        """image augmentation debug code"""
        """
        aug_images_train = tf.get_default_graph().get_tensor_by_name('aug_:0')
        tf.summary.image(name='ori_images', tensor=x_)
        tf.summary.image(name='aug_images_train', tensor=aug_images_train)
        merged = tf.summary.merge_all()
        summary_train = sess.run(merged, feed_dict={x_: test_imgs[:3], y_: test_labs[:3], lr_: 0.001, is_training: True})
        summary_writer.add_summary(summary_train, step)
        aug_images_test = tf.get_default_graph().get_tensor_by_name('aug_:0')
        tf.summary.image(name='aug_images_test', tensor=aug_images_test)
        summary_test = sess.run(aug_images_test, feed_dict={x_: test_imgs[:3], y_: test_labs[:3], lr_: 0.001, is_training: False})
        print np.shape(summary_test)
        print np.save('test_images.npy', summary_test)
        summary_writer.add_summary(summary_test, step)
        """
    """ #### training ### """
    train_fetches = [train_op, accuracy_op, loss_op]
    batch_xs, batch_ys , batch_fname= input.next_batch(batch_size, train_imgs, train_labs )
    batch_xs=batch_xs/255.
    train_feedDict = {x_: batch_xs, y_: batch_ys, cam_ind:ABNORMAL ,lr_: learning_rate, is_training: True}
    _ , train_acc, train_loss = sess.run( fetches=train_fetches, feed_dict=train_feedDict )
    #print 'train acc : {} loss : {}'.format(train_acc, train_loss)
    model.write_acc_loss(summary_writer ,'train' , loss= train_loss , acc=train_acc  ,step= step)



