from trainers.Trainer import *

class DPMSVMRandomTrainer(Trainer):
    def __init__(self, X, y, train_id_list, pool_id_list):
        super().__init__(X, y, train_id_list, pool_id_list)
        self.pre_processor = DPMPreProcessor()

class DPMSVMCertaintyTrainer(Trainer):
    def __init__(self, X, y, train_id_list, pool_id_list):
        super().__init__(X, y, train_id_list, pool_id_list)
        self.pre_processor = DPMPreProcessor()
        self.sampler = CertaintySampler()

class DPMSVMUncertaintyTrainer(Trainer):
    def __init__(self, X, y, train_id_list, pool_id_list):
        super().__init__(X, y, train_id_list, pool_id_list)
        self.pre_processor = DPMPreProcessor()
        self.sampler = UncertaintySampler()