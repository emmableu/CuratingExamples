from trainers.Trainer import *

class SVMRandomTrainer(Trainer):
    def __init__(self, X, y, train_id_list, pool_id_list):
        super().__init__(X, y, train_id_list, pool_id_list)

class SVMCertaintyTrainer(Trainer):
    def __init__(self, X, y, train_id_list, pool_id_list):
        super().__init__(X, y, train_id_list, pool_id_list)
        self.sampler = CertaintySampler()

class SVMUncertaintyTrainer(Trainer):
    def __init__(self, X, y, train_id_list, pool_id_list):
        super().__init__(X, y, train_id_list, pool_id_list)
        self.sampler = UncertaintySampler()