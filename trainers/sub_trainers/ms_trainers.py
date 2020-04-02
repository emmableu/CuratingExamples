from trainers.ModelSelectionTrainer import *
from trainers.Trainer import *


class MSRandomTrainer(ModelSelectionTrainer):
    def __init__(self, X, y, train_id_list, pool_id_list):
        super().__init__(X, y, train_id_list, pool_id_list)


class MSCertaintyTrainer(ModelSelectionTrainer):
    def __init__(self, X, y, train_id_list, pool_id_list):
        super().__init__(X, y, train_id_list, pool_id_list)
        self.sampler = CertaintySampler()


class MSUncertaintyTrainer(ModelSelectionTrainer):
    def __init__(self, X, y, train_id_list, pool_id_list):
        super().__init__(X, y, train_id_list, pool_id_list)
        self.sampler = UncertaintySampler()
