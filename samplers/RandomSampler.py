from samplers.Sampler import Sampler
import numpy as np
class RandomSampler(Sampler):
    def __init__(self, step):
        super().__init__(step)

    def sample(self, pool_id_list):
        return  np.random.choice(pool_id_list, size=self.step)
