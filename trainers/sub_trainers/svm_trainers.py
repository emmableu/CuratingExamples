from trainers.Trainer import *

class SVMRandomTrainer(Trainer):
    def __init__(self):
        super().__init__()

    def populate(self, X, y, train_id_list, pool_id_list):
        super().populate(X, y, train_id_list, pool_id_list)

class SVMCertaintyTrainer(Trainer):
    def __init__(self):
        super().__init__()

    def populate(self, X, y, train_id_list, pool_id_list):
        super().populate(X, y, train_id_list, pool_id_list)
        self.sampler = CertaintySampler()

class SVMUncertaintyTrainer(Trainer):
    def __init__(self):
        super().__init__()

    def populate(self, X, y, train_id_list, pool_id_list):
        super().populate(X, y, train_id_list, pool_id_list)
        self.sampler = UncertaintySampler()