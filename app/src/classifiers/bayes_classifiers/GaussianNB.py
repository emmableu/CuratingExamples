from sklearn.naive_bayes import GaussianNB
from classifiers.Model import Model


class GaussianNBModel(Model):
    def __init__(self):
        self.name = "Gaussian NB"
        self.model = GaussianNB()
