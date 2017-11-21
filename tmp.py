import numpy as np
a=[1,2,3,4,5,6]
print a[-1:]
print a[:-1]

try:
    a=3
    raise ValueError
except:
    pass
print a


a_cls=np.zeros(3)
b_cls=np.ones(3)
a=[[0,1],[0,1]]
b=[[0,1],[1,0]]
print np.ndim(a)
print a_cls==b_cls
print np.argmax( a, axis=1 )

