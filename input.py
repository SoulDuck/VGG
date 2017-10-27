import random
import numpy as np
def next_batch(batch_size , imgs, labs, fnames=None):
    indices = random.sample(range(np.shape(labs)[0]), batch_size)
    if not type(imgs).__module__ == np.__name__:  # check images type to numpy
        imgs = np.asarray(imgs)
    imgs = np.asarray(imgs)
    batch_xs = imgs[indices]
    batch_ys = labs[indices]
    if not fnames is None:
        batch_fs = fnames[indices]
    else:
        batch_fs = None
    return batch_xs, batch_ys , batch_fs




def cls2onehot(cls, depth):
    debug_flag=False
    if not type(cls).__module__ == np.__name__:
        cls=np.asarray(cls)
    cls=cls.astype(np.int32)
    debug_flag = False
    labels = np.zeros([len(cls), depth] , dtype=np.int32)
    for i, ind in enumerate(cls):
        labels[i][ind:ind + 1] = 1
    if __debug__ == debug_flag:
        print '#### data.py | cls2onehot() ####'
        print 'show sample cls and converted labels'
        print cls[:10]
        print labels[:10]
        print cls[-10:]
        print labels[-10:]
    return labels