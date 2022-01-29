import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


class Perceptron(object):
     
    def __init__(self, max_iter=100, lr=0.01):
        self.max_iter = max_iter
        self.lr = lr

    def fit(self, samples, label):
        rgen = np.random.RandomState(1)
        self.weights = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        print(self.weights)
        #self.weights = np.array([0.25,0.75,0])
        
        last_loss = 1
        for i in range(self.max_iter):
            predictions = []
            for sample, true_value in zip(samples, labels):
                hypothesis = self.predict(xi)
                self.weights[1:] = self.weights[1:] + self.lr * (true_value - hypothesis) * sample
                self.weights[0] = self.weights[0] + self.lr * (true_value - hypothesis) * 1
                predictions.append(hypothesis)
            loss = accuracy_score(predictions, y)
            temp = abs(last_loss - loss)
            last_loss = loss
            print(self.weights)
            if temp <= 0.001:
                break
            
        return self.weights, i
     
    def predict(self, X):
        return self.activation_function(X)

    def net_input(self, X):
            weighted_sum = np.dot(X, self.weights[1:]) + self.weights[0]
            return weighted_sum
     

    def activation_function(self, X):
     #defining the activation function
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
