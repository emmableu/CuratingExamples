from sklearn.dummy import DummyClassifier
from classifiers.Model import Model


class BaselineModel(Model):
    def __init__(self):
        self.name = "Baseline"
        self.model = DummyClassifier()
