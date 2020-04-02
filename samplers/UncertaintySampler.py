from samplers.Sampler import Sampler
import numpy as np

class UncertaintySampler(Sampler):
    def __init__(self):
        super().__init__()

    def sample(self):
        order = np.argsort(np.abs(self.prob-0.5))[:self.step]    ## uncertainty sampling by prediction probability
        most_uncertain = np.array(self.pool_id_list)[order[:self.step]]
        return most_uncertain
