from sklearn.naive_bayes import BernoulliNB
from classifiers.Model import Model


class BernoulliNBModel(Model):
    def __init__(self):
        self.name = "Bernoulli NB"
        self.model = BernoulliNB()
