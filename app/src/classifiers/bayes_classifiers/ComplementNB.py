from sklearn.naive_bayes import ComplementNB
from classifiers.Model import Model


class ComplementNBModel(Model):
    def __init__(self):
        self.name = "Complement NB"
        self.model = ComplementNB()
