import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score

data = pd.read_csv("Exp.txt")

feat_1 = np.array(list(data["Feature_1"]))
feat_2 = np.array(list(data["Feature_2"]))
feat_3 = np.array(list(data["Feature_3"]))

labels = list(data["Label"])
#import pdb;pdb.set_trace()
#X_train = np.array([feat_1.transpose(), feat_2.transpose(), feat_3.transpose()])
X_train = []
for i, j, k in zip(feat_1, feat_2, feat_3):
    temp = [i, j, k]
    X_train.append(temp)

X_train = np.array(X_train)
poly = PolynomialFeatures(degree=2)
X_ = poly.fit_transform(X_train)

y_train = np.array(labels)
Multiclass_model = LogisticRegression(multi_class='ovr')
Multiclass_model.fit(X_, y_train)

y_pred = Multiclass_model.predict(X_)
