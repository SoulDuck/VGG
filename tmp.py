import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np; np.random.seed(0)
#import seaborn as sns;sns.set()
#uniform_data = np.random.rand(10, 12)
#ax = sns.heatmap(uniform_data , annot=True , vmin=0, vmax=1 ,cmap="YlGnBu")
#fig=ax.get_figure()

#fig.savefig('tmp_heatmap.png')

#plt.show(fig)
#plt.imsave('tmp_heatmap.png' , ax)
#plt.show(ax)

a=range(256)
a=np.asarray(a)
a=a.reshape([16,16])
a=a/255.
a[:]
cmap = plt.cm.jet
a=cmap(a)
print a
print np.shape(a)
a[:,:4]=0
plt.imsave('tmp.png',a)
exit()
plt.imsave('tmp.png',cmap(a))
r=range(256)
g=range(256)
g.reverse()
b=range(256)

"""
r,g,b=map(lambda channel: np.asarray(channel).reshape([16,16]) , [r,g,b])

rgb = np.zeros([16,16,3])

rgb[:,:,0] =r
rgb[:,:,1] = 0
rgb[:,:,2] = 0
print rgb[0,0,:]
print rgb[0,1,:]
print np.shape(rgb)
plt.imshow(rgb)
plt.show()
"""


