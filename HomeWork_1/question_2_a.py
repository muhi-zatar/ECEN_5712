import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

data = pd.read_csv("Classify-2DwLabels-2.txt")

sns.lmplot('feat_1', 'feat_2', data, hue='label', fit_reg=False)
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.show()
