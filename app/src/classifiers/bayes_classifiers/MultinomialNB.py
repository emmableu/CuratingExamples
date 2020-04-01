from sklearn.naive_bayes import MultinomialNB
from classifiers.Model import Model


class MultinomialNBModel(Model):
    def __init__(self):
        self.name = "Multinomial NB"
        self.model = MultinomialNB()
