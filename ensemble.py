#-*- coding:utf-8 -*-
import tensorflow as tf
from PIL import Image
import numpy as np
import os , glob
from skimage import color
from skimage import io

import matplotlib
if "DISPLAY" not in os.environ:
    # remove Travis CI Error
    print 'DISPLAY not in this enviroment'
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import eval
import roc
import fundus
import itertools
import pickle
import argparse

parser=argparse.ArgumentParser()
parser.add_argument('--models_path')
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


def ensemble_with_all_combibation(model_paths , test_images , test_labels):
    max_acc=0
    max_pred=None
    max_list = []
    f = open('best_ensemble.txt', 'w')
    if not os.path.isfile('predcitions.pkl'):
        p = open('predcitions.pkl' , 'w')
        pred_dic={}
        for path in model_paths:
            name=path.split('/')[-1]
            print 'path : ', path
            print 'name : ', name

            path=os.path.join(path , 'model')
            #./models/vgg_11/step_12500_acc_0.841666698456 --> ./models/vgg_11/step_12500_acc_0.841666698456/model
            # activation 이 저장될 세이브 장소를 만든다
            # 파일 경로는 ./activation_map/model_name/
            save_root_folder = './activation_map/{}'.format(name)

            os.mkdir(save_root_folder)
            tmp_pred = eval.eval(path, test_images , batch_size=60 , save_root_folder=save_root_folder)
            print 'tmp_pred' , tmp_pred
            pred_dic[path]=tmp_pred
        #pred_model_path_list=zip(pred_list , model_paths)
        pickle.dump(pred_dic,p)
    else:
        p = open('predcitions.pkl', 'r')
        pred_dic=pickle.load(p)
    print pred_dic.keys()
    for k in range(2,len(pred_dic.keys())+1):
        k_max_acc = 0
        k_max_list = []
        print 'K : {}'.format(k)
        for cbn_models in itertools.combinations(pred_dic.keys(),k):
            print cbn_models
            #cbn_preds=map(lambda cbn_model: pred_dic[cbn_model],cbn_models)
            for idx, model in enumerate(cbn_models):
                pred = pred_dic[model]
                print 'pred shape : {}'.format(np.shape(pred))
                #print idx
                if idx == 0:
                    pred_sum = pred
                else:
                    pred_sum += pred

            """
                for idx ,pred in enumerate(cbn_preds):
                print cbn_models[idx]
                print idx
                print pred[:10]
                if idx ==0 :
                    pred_sum = pred
                else:
                    pred_sum += pred
            """
            print len(cbn_models)

            pred_sum = pred_sum / float(len(cbn_models))
            print 'pred' ,pred[:10]
            acc=eval.get_acc(pred_sum , test_labels)
            print 'accuracy : {}'.format(acc)
            #print cbn_models ,':',acc

            p = open('predcitions.pkl', 'r')
            pred_dic=pickle.load(p)
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

    return acc , max_list , max_pred


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


if __name__ == '__main__':

    models_path=get_models_paths(args.models_path)
    print 'number of model paths : {}'.format(len(models_path))
    train_images, train_labels, train_filenames, test_images, test_labels, test_filenames = fundus.type1(
        './fundus_300', resize=(299, 299))

    acc, max_list , pred =ensemble_with_all_combibation(models_path ,test_images , test_labels)
    f=open('best_pred.pkl' , 'w')
    pickle.dump(pred  , f)
    f.close()
    f = open('label.pkl', 'w')
    pickle.dump(test_labels, f)
    f.close()
    names=map(lambda path: path.split('/')[-2]  ,max_list)
    print 'best model list : ',names

    #get_ensemble_actmap(names , './activation_map')
    #roc.plotROC(pred , test_labels )

    """
    pred_sum=ensemble('./models', test_images )
    acc =eval.get_acc(pred_sum , test_labels)
    print acc

    """
    """
    model_list=['step_13600_acc_0.840000033379' ,'step_14600_acc_0.841666817665' ,'step_15900_acc_0.843333363533']
    print model_list

    
    """
