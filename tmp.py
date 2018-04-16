from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import aug
mean = 0.0
maxval = 0.0
didextract = True
img=Image.open('tmp_oct.png')

height,width=np.shape(img)
img=np.asarray(img)
mean=np.mean(img)
max_val=np.max(img)
rescaled_img=np.round(255.0 * (img - mean) / (max_val - mean))  # linear scaling
def reverse_colourmap(cmap, name = 'my_cmap_r'):
    """
    In:
    cmap, name
    Out:
    my_cmap_r

    Explanation:
    t[0] goes from 0 to 1
    row i:   x  y0  y1 -> t[0] t[1] t[2]
                   /
                  /
    row i+1: x  y0  y1 -> t[n] t[1] t[2]

    so the inverse should do the same:
    row i+1: x  y1  y0 -> 1-t[0] t[2] t[1]
                   /
                  /
    row i:   x  y1  y0 -> 1-t[n] t[2] t[1]
    """
    reverse = []
    k = []

    for key in cmap._segmentdata:
        k.append(key)
        channel = cmap._segmentdata[key]
        data = []

        for t in channel:
            data.append((1-t[0],t[2],t[1]))
        reverse.append(sorted(data))

    LinearL = dict(zip(k,reverse))
    my_cmap_r = mpl.colors.LinearSegmentedColormap(name, LinearL)
    return my_cmap_r

"""
plt.imshow(Image.open('grey_fundus.png'))
plt.show()

img=np.asarray(Image.open('grey_fundus.png').convert('L'))
cmap_r=reverse_colourmap(mpl.cm.gray)

plt.imshow(cmap_r(img))
plt.show()
"""

color_img=Image.open('sample_image_color.png').convert('RGB')
color_img=aug.clahe_equalized(np.asarray(color_img))
plt.imshow(color_img)
plt.show()

