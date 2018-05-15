import tensorflow as tf
import matplotlib.pyplot as plt

import numpy as np; np.random.seed(0)
import seaborn as sns;sns.set()
uniform_data = np.random.rand(10, 12)
ax = sns.heatmap(uniform_data , annot=True , vmin=0, vmax=1 ,cmap="YlGnBu")

plt.show(ax)