from sklearn.dummy import DummyClassifier
from classifiers.Model import Model
import numpy as np
from collections import Counter

class ORModel:
    def fit(self, X, y):
        return None
    def predict(self, X):
        y = []
        for x_row in X:
            if Counter(x_row)[1] >0:
                y.append(1)
            else:
                y.append(0)
        y = np.array(y)
        return y



class ORExistenceModel(Model):
    def __init__(self):
        self.name = "or_existence_model"
        self.model = ORModel()
