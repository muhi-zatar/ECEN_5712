import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


class Perceptron(object):
     
    def __init__(self, n_iterations=100, random_state=1, learning_rate=0.01):
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.learning_rate = learning_rate

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.coef_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        print(self.coef_)
        #self.coef_ = np.array([0.25,0.75,0])
        import pdb;pdb.set_trace()
        last_loss = 1
        for i in range(self.n_iterations):
            predictions = []
            for xi, expected_value in zip(X, y):
                predicted_value = self.predict(xi)
                self.coef_[1:] = self.coef_[1:] + self.learning_rate * (expected_value - predicted_value) * xi
                self.coef_[0] = self.coef_[0] + self.learning_rate * (expected_value - predicted_value) * 1
                predictions.append(predicted_value)
            loss = accuracy_score(predictions, y)
            temp = abs(last_loss - loss)
            last_loss = loss
            print(self.coef_)
            if temp <= 0.001:
                break
            

        return self.coef_, i
     

    def net_input(self, X):
            weighted_sum = np.dot(X, self.coef_[1:]) + self.coef_[0]
            return weighted_sum
     

    def activation_function(self, X):
            weighted_sum = self.net_input(X)
            return np.where(weighted_sum >= 0.0, 1, 0)
     

data = pd.read_csv("Classify-2DwLabels-2.txt")
feat_1 = np.array(list(data["feat_1"]))
feat_2 = np.array(list(data["feat_2"]))
labels = list(data["label"])
X_train = []
for i, j in zip(feat_1, feat_2):
    temp = [i, j]
    X_train.append(temp)

X_train = np.array(X_train)
y_train = np.array(labels)

prcptrn = Perceptron()

weights, iter = prcptrn.fit(X_train, y_train)
print(iter+1)
