a=[1,2,3,4,5,6]
print a[-1:]
print a[:-1]

try:
    a=3
    raise ValueError
except:
    pass
print a
