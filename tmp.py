import itertools
import multiprocessing
k=2
def _fn(x):
    for ind , j  in enumerate(x):
        print ind , j
    return j , ind
pool = multiprocessing.Pool()
for acc, cbn_models  in pool.imap( _fn, itertools.combinations([1,2,3] , k)):
    print acc, cbn_models