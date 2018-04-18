#-*- coding:utf-8 -*-
import tensorflow as tf
from PIL import Image
import numpy as np
import os , glob
from skimage import color
from skimage import io
import multiprocessing
import time
import random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import eval
import itertools
import pickle
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--models_path' , type = str , default='./models/fundus_300_N_VS_R/VGG_11/ensemble')
args=parser.parse_args()


def get_ensemble_actmap(model_list , actmap_folder):
    """
    :param model_list: []
    :return:
    """
    overlay_img_path=os.path.join(actmap_folder, 'overlay')
    if not os.path.isdir(overlay_img_path):
        os.mkdir(overlay_img_path)
    tmp_path = os.path.join(actmap_folder, model_list[0])
    _ , subfolders , files =os.walk(tmp_path).next()
    act_imgs=[]
    act_imgs=[]
    for subfolder in subfolders:
        for i, model in enumerate(model_list):
            print model
            act_img_path=os.path.join(actmap_folder , model , subfolder , 'abnormal_actmap.png') #./activation_map/step_5900_acc_0.84/img_0
            ori_img_path = os.path.join(actmap_folder, model, subfolder,
                                        'image_test.png')  # ./activation_map/step_5900_acc_0.84/img_0
            ori_img = Image.open(ori_img_path).convert("RGBA")
            act_img = Image.open(act_img_path).convert("RGBA")
            act_imgs.append(act_img)
        #print len(act_imgs)
        while len(act_imgs) != 1:
            tmp_act_imgs=[]
            remainder=len(act_imgs) % 2
            share=len(act_imgs) / 2
            #print 'remainder {}' , remainder
            #print 'share {}', share
            for s in range(share):
                tmp_act_imgs.append(Image.blend(act_imgs[2*s] ,act_imgs[2*s+1] , 0.5))
            if remainder ==1 :
                act_imgs[-1].putalpha(128)
                tmp_act_imgs.append(act_imgs[-1])
            act_imgs=tmp_act_imgs
            #print 'n actimgs ',len(act_imgs)
            #print len(tmp_act_imgs  )
        #ori_img = plt.rgb2gray(ori_img);
        overlay_img=Image.blend(ori_img , act_imgs[0] , 0.5)
        #cmap=plt.cm.jet
        #overlay_img=cmap(overlay_img)
        plt.imsave( os.path.join(overlay_img_path , '{}.png'.format(subfolder)), overlay_img)


    """
    background = original_img.convert("RGBA")
    overlay = vis_abnormal.convert("RGBA")

    overlay_img = Image.blend(background, overlay, 0.5)
    plt.imshow(overlay_img, cmap=plt.cm.jet)
    plt.show()
    overlay_img.save(filename)
    """


def get_models_paths(dir_path):
    subdir_paths=[path[0] for path in os.walk(dir_path)]
    #subdir_paths = map(lambda name: os.path.join(dir_path, name), subdir_names)
    ret_subdir_paths=[]
    for path in subdir_paths:
        if os.path.isfile(os.path.join(path , 'model.meta')):
            ret_subdir_paths.append(path)
    return ret_subdir_paths



def ensemble_with_all_combination(model_paths, test_images, test_labels, actmap_folder):
    max_acc=0
    max_pred=None
    max_list = []
    f = open('best_ensemble.txt', 'w') # Each Combination holds a list of file paths that have the best accuracy

    # Predictions from Eval is saved in the form of a pickle.
    if not os.path.isfile('predcitions.pkl'):
        p = open('predcitions.pkl' , 'wb')
        pred_dic={}
        for path in model_paths:
            fname=os.path.split(path)[-1]
            name=os.path.splitext(fname)[0]
            path=os.path.join(path , 'model')
            tmp_pred = eval.eval(path, test_images , batch_size=60 , actmap_folder=actmap_folder)
            pred_dic[name]=tmp_pred
        pickle.dump(pred_dic,p)
    else:
        p = open('predcitions.pkl', 'rb')
        pred_dic=pickle.load(p)

    # Run all combinations
    for k in range(2,len(pred_dic.keys())+1):
        k_max_acc = 0
        k_max_list = []
        print 'K : {}'.format(k)

        for cbn_models in itertools.combinations(pred_dic.keys(),k):
            for idx, model in enumerate(cbn_models):
                pred = pred_dic[model]
                if idx == 0:
                    pred_sum = pred
                else:
                    pred_sum += pred
            pred_sum = pred_sum / float(len(cbn_models))
            acc=eval.get_acc(pred_sum , test_labels)
            if max_acc < acc :
                max_acc=acc
                max_pred=pred_sum
                max_list=cbn_models
            if k_max_acc < acc:
                k_max_acc = acc
                k_max_list = cbn_models
        msg = 'k : {} , list : {} , accuracy : {}\n'.format(k, k_max_list , k_max_acc)
        f.write(msg)
        f.flush()
    msg='model list : {} , accuracy : {}'.format(max_list , max_acc)
    f.write(msg)
    f.flush()
    return max_acc , max_list , max_pred


def ensemble_with_all_combination_multiproc(model_paths, test_images, test_labels, actmap_folder):
    max_acc = 0
    max_pred = None
    max_list = []
    f = open('best_ensemble.txt', 'w')  # Each Combination holds a list of file paths that have the best accuracy

    # Predictions from Eval is saved in the form of a pickle.
    if not os.path.isfile('predcitions.pkl'):
        p = open('predcitions.pkl', 'wb')
        pred_dic = {}
        for path in model_paths:
            fname = os.path.split(path)[-1]
            name = os.path.splitext(fname)[0]
            path = os.path.join(path, 'model')
            tmp_pred = eval.eval(path, test_images, batch_size=60, actmap_folder=actmap_folder)
            pred_dic[name] = tmp_pred
        pickle.dump(pred_dic, p)
    else:
        p = open('predcitions.pkl', 'rb')
        pred_dic = pickle.load(p)


    print np.shape(pred_dic)
    # Run all combinations
    def _fn(cbn_models):  # cbn_models  ==> combinatation models

        for idx, model in enumerate(cbn_models):
            pred = pred_dic[model]
            if idx == 0:
                pred_sum = pred
            else:
                pred_sum += pred
        pred_sum = pred_sum / float(len(cbn_models))
        acc = eval.get_acc(pred_sum, test_labels)

        return acc , cbn_models , pred_sum

    pool=multiprocessing.Pool()
    for k in range(2, len(pred_dic.keys()) + 1):
        k_max_acc = 0
        k_max_list = []
        print 'K : {}'.format(k)

        for acc, cbn_models , pred_sum in pool.imap( _fn, itertools.combinations(pred_dic.keys(), k)):
            if max_acc < acc:
                max_acc = acc
                max_pred = pred_sum
                max_list = cbn_models
            if k_max_acc < acc:
                k_max_acc = acc
                k_max_list = cbn_models
        msg = 'k : {} , list : {} , accuracy : {}\n'.format(k, k_max_list, k_max_acc)
        f.write(msg)
        f.flush()
    msg = 'model list : {} , accuracy : {}'.format(max_list, max_acc)
    f.write(msg)
    f.flush()
    return max_acc, max_list, max_pred

def ensemble(model_paths , test_images):
    """
    :param models:
    :return:
    """
    path , subdir_names , _=os.walk(model_paths).next()
    subdir_paths=map(lambda name : os.path.join(path , name) , subdir_names)
    print 'model saved folder paths : {}'.format(subdir_paths)

    for i,subdir_path in enumerate(subdir_paths):
        pred=eval.eval(subdir_path , test_images)
        if i ==0 :
            pred_sum = pred
        else:
            pred_sum+=pred
    pred_sum=pred_sum/float(i+1)
    return pred_sum


def _load_images_labels(dir , label ,limit , random_flag):
    start = time.time()
    paths = []

    paths=glob.glob(os.path.join(dir , '*.png'))
    if random_flag is True:
        indices = random.sample(range(len(paths)), limit)
        paths = np.asarray(paths)[indices]
    imgs=map(lambda path : np.asarray(Image.open(path)) , paths[:limit])
    imgs=np.asarray(imgs)
    labs=np.zeros([len(imgs),2])
    labs[:,label ]=1
    return imgs , labs

if __name__ == '__main__':
    NORMAL = 0
    ABNORMAL = 1

    test_normalDir = '../fundus_data/cropped_original_fundus_300x300/normal_0/Test'
    test_abnormalDir = '../fundus_data/cropped_original_fundus_300x300/retina/Test'

    test_normal_imgs, test_normal_labs = _load_images_labels(test_normalDir, NORMAL, 172, True)
    test_abnormal_imgs, test_abnormal_labs = _load_images_labels(test_abnormalDir, ABNORMAL, None, False)

    test_imgs = np.vstack([test_normal_imgs, test_abnormal_imgs])
    test_labs = np.vstack([test_normal_labs, test_abnormal_labs])
    test_normal_imgs = None
    test_abnormal_imgs = None

    models_path=get_models_paths(args.models_path)
    print 'number of model paths : {}'.format(len(models_path))
    acc, max_list, pred = ensemble_with_all_combination_multiproc(models_path, test_imgs, test_labs, None)
    np.save('./N_VS_R_ensemble_best_preds', pred)
    np.save('./N_VS_R_ensemble_test_labels', test_labs) #
    print 'max Accuracy : ', acc
    print 'best model list : ',max_list
