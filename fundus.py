# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import input
import random
import os

def reconstruct_tfrecord_rawdata(tfrecord_path, resize=(299, 299)):
    print 'now Reconstruct Image Data please wait a second'
    reconstruct_image = []
    # caution record_iter is generator

    record_iter = tf.python_io.tf_record_iterator(path=tfrecord_path)

    ret_img_list = []
    ret_lab_list = []
    ret_fnames = []
    for i, str_record in enumerate(record_iter):
        example = tf.train.Example()
        example.ParseFromString(str_record)

        height = int(example.features.feature['height'].int64_list.value[0])
        width = int(example.features.feature['width'].int64_list.value[0])

        raw_image = (example.features.feature['raw_image'].bytes_list.value[0])
        label = int(example.features.feature['label'].int64_list.value[0])
        filename = example.features.feature['filename'].bytes_list.value[0]
        filename = filename.decode('utf-8')
        image = np.fromstring(raw_image, dtype=np.uint8)
        image = image.reshape((height, width, -1))
        ret_img_list.append(image)
        ret_lab_list.append(label)
        ret_fnames.append(filename)
    ret_imgs = np.asarray(ret_img_list)

    if np.ndim(ret_imgs) == 3:  # for black image or single image ?
        b, h, w = np.shape(ret_imgs)
        h_diff = h - resize[0]
        w_diff = w - resize[1]
        ret_imgs = ret_imgs[h_diff / 2: h_diff / 2 + resize[0], w_diff / 2: w_diff / 2 + resize[1], :]
    elif np.ndim(ret_imgs) == 4:  # Image Up sacle(x) image Down Scale (O)
        b, h, w, ch = np.shape(ret_imgs)
        h_diff = h - resize[0]
        w_diff = w - resize[1]
        ret_imgs = ret_imgs[:, h_diff / 2: h_diff / 2 + resize[0], w_diff / 2: w_diff / 2 + resize[1], :]
    ret_labs = np.asarray(ret_lab_list)
    ret_imgs = np.asarray(ret_imgs)
    ret_fnames = np.asarray(ret_fnames)
    return ret_imgs, ret_labs, ret_fnames

def type1(tfrecords_dir, onehot=True, resize=(299, 299)):
    """type1  데이터 확인 완료 함 """
    # type1 은 cataract_glaucoma , retina_catarct  , retina_glaucoma을 각각의 카테고리에 맞는 곳에 넣었다
    # 늑 cataract_glacucoma 는 cataract , glaucoma 에 넣었다

    images, labels, filenames = [], [], []
    names = ['normal_0', 'glaucoma', 'cataract', 'retina', 'cataract_glaucoma', 'retina_cataract', 'retina_glaucoma']
    for name in names:
        for type in ['train', 'test']:
            imgs, labs, fnames = reconstruct_tfrecord_rawdata(
                tfrecord_path=tfrecords_dir + '/' + name + '_' + type + '.tfrecord', resize=resize)
            print type, ' ', name
            print 'image shape', np.shape(imgs)
            print 'label shape', np.shape(labs)
            images.append(imgs);
            labels.append(labs), filenames.append(fnames)

    n = len(names)
    train_images, train_labels, train_filenames = [], [], []
    test_images, test_labels, test_filenames = [], [], []

    for i in range(n):
        train_images.append(images[i * 2]);
        train_labels.append(labels[i * 2]);
        train_filenames.append(filenames[i * 2])
        test_images.append(images[(i * 2) + 1]);
        test_labels.append(labels[(i * 2) + 1]);
        test_filenames.append(filenames[(i * 2) + 1])

    train_images, train_labels, train_filenames, test_images, test_labels, test_filenames = \
        map(lambda x: np.asarray(x),
            [train_images, train_labels, train_filenames, test_images, test_labels, test_filenames])

    def _fn1(x, a, b):
        x[a] = np.concatenate([x[a], x[b]], axis=0)  # cata_glau을  cata에 더한다
        return x

    """
    4번은 cataract glaucoma 
    5번은 retina cataract 
    6번은 retina glaucoma 
    1번은 glaucoma
    2번은 cataract
    3번은 retina
    """
    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 1, 4),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 1, 4), [test_images, test_labels, test_filenames])
    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 2, 4),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 2, 4), [test_images, test_labels, test_filenames])

    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 2, 5),
                                                      [train_images, train_labels, train_filenames])  # retina cataract을
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 2, 5), [test_images, test_labels, test_filenames])
    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 3, 5),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 3, 5), [test_images, test_labels, test_filenames])

    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 1, 6),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 1, 6), [test_images, test_labels, test_filenames])
    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 3, 6),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 3, 6), [test_images, test_labels, test_filenames])

    for i in range(4):
        print '#', np.shape(train_images[i])
    for i in range(4):
        print '#', np.shape(test_images[i])

    train_labels = train_labels[:4]
    train_filenames = train_filenames[:4]

    test_images = test_images[:4]
    test_labels = test_labels[:4]
    test_filenames = test_filenames[:4]

    train_images, train_labels, train_filenames, test_images, test_labels, test_filenames = \
        map(lambda x: np.concatenate([x[0], x[1], x[2], x[3]], axis=0), \
            [train_images, train_labels, train_filenames, test_images, test_labels, test_filenames])

    print 'train images ', np.shape(train_images)
    print 'train labels ', np.shape(train_labels)
    print 'train fnamess ', np.shape(train_filenames)
    print 'test images ', np.shape(test_images)
    print 'test labels ', np.shape(test_labels)
    print 'test fnames ', np.shape(test_filenames)
    n_classes = 2
    if onehot:
        train_labels = input.cls2onehot(train_labels, depth=n_classes)
        test_labels = input.cls2onehot(test_labels, depth=n_classes)

    return train_images, train_labels, train_filenames, test_images, test_labels, test_filenames

def type2(tfrecords_dir, onehot=True, resize=(299, 299) , random_shuffle = True ):
    # normal : 3000
    # glaucoma : 1000
    # retina : 1000
    # cataract : 1000
    train_images, train_labels, train_filenames = [], [], []
    test_images, test_labels, test_filenames = [], [], []

    names = ['normal_0', 'glaucoma', 'cataract', 'retina', 'cataract_glaucoma', 'retina_cataract', 'retina_glaucoma']
    limits = [3000 , 1000 , 1000 , 1000]
    for ind , name in enumerate(names):
        for type in ['train', 'test']:
            imgs, labs, fnames = reconstruct_tfrecord_rawdata(
                tfrecord_path=tfrecords_dir + '/' + name + '_' + type + '.tfrecord', resize=resize)
            print type, ' ', name
            print 'image shape', np.shape(imgs)
            print 'label shape', np.shape(labs)

            if type =='train':
                random_indices = random.sample(range(len(labs)),
                                               len(labs))  # normal , glaucoma , cataract , retina 만 random shuffle 을 한다
                if random_shuffle and ind < 4:
                    print 'random shuffle On : {} limit : {}'.format(name , limits[ind])
                    limit =limits[ind]
                else :
                    limit = None
                train_images.append(imgs[random_indices[:limit]]);
                train_labels.append(labs[random_indices[:limit]]);
                train_filenames.append(fnames[random_indices[:limit]]);
            else :
                test_images.append(imgs);
                test_labels.append(labs);
                test_filenames.append(fnames);
    def _fn1(x, a, b):
        x[a] = np.concatenate([x[a], x[b]], axis=0)  # cata_glau을  cata에 더한다
        return x

    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 1, 4),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 1, 4), [test_images, test_labels, test_filenames])
    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 2, 4),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 2, 4), [test_images, test_labels, test_filenames])

    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 2, 5),
                                                      [train_images, train_labels, train_filenames])  # retina cataract을
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 2, 5), [test_images, test_labels, test_filenames])
    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 3, 5),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 3, 5), [test_images, test_labels, test_filenames])

    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 1, 6),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 1, 6), [test_images, test_labels, test_filenames])
    train_images, train_labels, train_filenames = map(lambda x: _fn1(x, 3, 6),
                                                      [train_images, train_labels, train_filenames])
    test_images, test_labels, test_filenames = map(lambda x: _fn1(x, 3, 6), [test_images, test_labels, test_filenames])

    for i in range(4):
        print '#', np.shape(train_images[i])
    for i in range(4):
        print '#', np.shape(test_images[i])

    train_labels = train_labels[:4]
    train_filenames = train_filenames[:4]

    test_images = test_images[:4]
    test_labels = test_labels[:4]
    test_filenames = test_filenames[:4]

    train_images, train_labels, train_filenames, test_images, test_labels, test_filenames = \
        map(lambda x: np.concatenate([x[0], x[1], x[2], x[3]], axis=0), \
            [train_images, train_labels, train_filenames, test_images, test_labels, test_filenames])

    print 'train images ', np.shape(train_images)
    print 'train labels ', np.shape(train_labels)
    print 'train fnamess ', np.shape(train_filenames)
    print 'test images ', np.shape(test_images)
    print 'test labels ', np.shape(test_labels)
    print 'test fnames ', np.shape(test_filenames)
    n_classes = 2
    if onehot:
        train_labels = input.cls2onehot(train_labels, depth=n_classes)
        test_labels = input.cls2onehot(test_labels, depth=n_classes)
    if not os.path.isdir('./type2'):
        os.mkdir('./type2')
    count=0
    while True:
        f_path='./type2/{}'.format(count)
        if not os.path.isdir(f_path):
            os.mkdir(f_path)
            break;
        else:
            count += 1


    np.save(os.path.join(f_path , 'train_imgs.npy') , train_images)
    np.save(os.path.join(f_path, 'train_labs.npy'), train_labels)
    np.save(os.path.join(f_path, 'train_fnames.npy'), train_filenames)
    return train_images, train_labels, train_filenames, test_images, test_labels, test_filenames

if '__main__' == __name__:
    type2('./fundus_300_debug')
