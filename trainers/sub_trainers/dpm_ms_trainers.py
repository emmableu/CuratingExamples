from trainers.Trainer import *
from trainers.ModelSelectionTrainer import *


class DPMMSRandomTrainer(ModelSelectionTrainer):
    def __init__(self, X, y, train_id_list, pool_id_list):
        super().__init__(X, y, train_id_list, pool_id_list)
        self.pre_processor = DPMPreProcessor()


class DPMMSCertaintyTrainer(ModelSelectionTrainer):
    def __init__(self, X, y, train_id_list, pool_id_list):
        super().__init__(X, y, train_id_list, pool_id_list)
        self.pre_processor = DPMPreProcessor()
        self.sampler = CertaintySampler()


class DPMMSUncertaintyTrainer(ModelSelectionTrainer):
    def __init__(self, X, y, train_id_list, pool_id_list):
        super().__init__(X, y, train_id_list, pool_id_list)
        self.pre_processor = DPMPreProcessor()
        self.sampler = UncertaintySampler()
