from sklearn.ensemble import BaggingClassifier
from classifiers.Model import Model


class BaggingModel(Model):
    def __init__(self):
        self.name = "BaggingClassifier"
        self.model = BaggingClassifier()
