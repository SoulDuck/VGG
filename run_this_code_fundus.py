#-*- coding:utf-8 -*-
import model
import input
import os
import fundus
import numpy as np
import tensorflow as tf
import aug
resize=(299,299)
train_imgs ,train_labs ,train_fnames, test_imgs ,test_labs , test_fnames=fundus.type2(tfrecords_dir='./fundus_300' , onehot=True , resize=resize)
#normalize

print np.shape(test_labs)
if np.max(train_imgs) > 1:
    train_imgs=train_imgs/255.
    test_imgs=test_imgs/255.
    print 'train_imgs max :',np.max(train_imgs)
    print 'test_imgs max :', np.max(test_imgs)

h,w,ch=train_imgs.shape[1:]
n_classes=np.shape(train_labs)[-1]
print 'the # classes : {}'.format(n_classes)

x_ , y_ , lr_ , is_training = model.define_inputs(shape=[None, h ,w, ch ] , n_classes=n_classes )
logits=model.build_graph(x_=x_ , y_=y_ ,is_training=is_training , aug_flag=True , actmap_flag=True )
#train_op, accuracy_op , loss_op , pred_op =model.train_algorithm_adam(logits=logits,labels=y_ , learning_rate=lr_ , l2_loss=False)
train_op, accuracy_op , loss_op , pred_op = model.train_algorithm_momentum(logits=logits,labels=y_ , learning_rate=lr_)
log_count =0;
while True:
    logs_path='./logs/fundus_300/{}'.format(log_count)
    if not os.path.isdir(logs_path):
        os.mkdir(logs_path)
        break;
    else:
        log_count+=1

sess, saver , summary_writer =model.sess_start(logs_path)


model_count =0;
while True:
    model_path='./models/fundus_300/{}'.format(model_count)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)
        break;
    else:
        model_count+=1



max_iter=80000
ckpt=100
batch_size=60
start_time=0
train_acc=0
train_val=0

share=len(test_labs)/batch_size
remainder=len(test_labs)/batch_size

for step in range(max_iter):
    #### learning rate schcedule
    if step  < 30000:
        learning_rate = 0.001
    elif step < 50000:
        learning_rate = 0.0001
    else:
        learning_rate = 0.00001
    ####
    if step % ckpt==0:
        """ #### testing ### """
        test_fetches = [ accuracy_op, loss_op, pred_op ]
        val_acc_mean , val_loss_mean , pred_all = [] , [] , []
        for i in range(share): #여기서 테스트 셋을 sess.run()할수 있게 쪼갭니다
            test_feedDict = { x_: test_imgs[i*batch_size:(i+1)*batch_size], y_: test_labs[i*batch_size:(i+1)*batch_size],  is_training: False }
            val_acc ,val_loss , pred = sess.run( fetches=test_fetches, feed_dict=test_feedDict )
            val_acc_mean.append(val_acc)
            val_loss_mean.append(val_loss)
            pred_all.append(pred)
        val_acc_mean=np.mean(np.asarray(val_acc_mean ))
        val_loss_mean=np.mean(np.asarray(val_loss_mean))
        print 'validation acc : {} loss : {}'.format( val_acc_mean, val_loss_mean )

        model.write_acc_loss( summary_writer, 'validation', loss=val_loss_mean, acc=val_acc_mean, step=step)
        saver.save(sess=sess,save_path=os.path.join(model_path,'model_{}'.format(step)))

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
    batch_xs, batch_ys, batch_fs = input.next_batch(batch_size, train_imgs, train_labs, train_fnames)
    train_feedDict = {x_: batch_xs, y_: batch_ys, lr_: learning_rate, is_training: True}
    _ , train_acc, train_loss = sess.run( fetches=train_fetches, feed_dict=train_feedDict )
    #print 'train acc : {} loss : {}'.format(train_acc, train_loss)
    model.write_acc_loss(summary_writer ,'train' , loss= train_loss , acc=train_acc  ,step= step)
    print 'Train acc : {} loss : {}'.format(train_acc, train_loss)


