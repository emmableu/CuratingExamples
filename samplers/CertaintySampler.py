from samplers.Sampler import Sampler
import numpy as np

class CertaintySampler(Sampler):
    def __init__(self):
        super().__init__()

    def sample(self):
        order = np.argsort(np.abs(self.prob))[::-1]
        most_certain = np.array(self.pool_id_list)[order[:self.step]]
        return most_certain
