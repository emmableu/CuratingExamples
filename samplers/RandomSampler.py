from samplers.Sampler import Sampler
import numpy as np

class RandomSampler(Sampler):
    def __init__(self):
        super().__init__()

    def sample(self):
        return np.random.choice(self.pool_id_list, size=self.step)
