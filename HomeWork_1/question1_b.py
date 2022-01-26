import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#sns.set(style="ticks", color_codes=True)

data = pd.read_csv("Exp.txt")
#data = data.drop("label",1)
#import pdb;pdb.set_trace()
sns.pairplot(data = data, hue="Label", diag_kind="hist")
plt.show()
