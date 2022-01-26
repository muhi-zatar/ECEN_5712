
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import collections

data = pd.read_csv("Classify-3DwLabels-2.txt")
sorted_list = data.sort_values(by=["label"])
fig = plt.figure()

ax = fig.add_subplot(projection='3d')

indices = [0,109, 249]
#import pdb;pdb.set_trace()
for i in range(len(indices)-1):

    xs = list(sorted_list["feat_1"])[indices[i]: indices[i+1]]
    ys = list(sorted_list["feat_2"])[indices[i]: indices[i+1]]
    zs = list(sorted_list["feat_3"])[indices[i]: indices[i+1]]

    ax.scatter(xs, ys, zs)

ax.set_xlabel('Feature 3')
ax.set_ylabel('Feature 1')
ax.set_zlabel('Feature 2')

plt.show()
