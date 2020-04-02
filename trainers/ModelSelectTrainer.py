from sklearn.naive_bayes import GaussianNB
from trainers.Trainer import Trainer


class MsTrainer(Trainer):
    def __init__(self, X, y):
        self.comment = "Model Selection Training, no preprocessing and no candidate"
        super
