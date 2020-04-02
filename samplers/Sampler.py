class Sampler:
    def __init__(self):
        self.name = None
        self.pool_id_list = None
        self.prob = None
        self.step = None

    def populate(self, step, pool_id_list, prob):
        self.name = 'sample'
        self.pool_id_list = pool_id_list
        self.prob = prob
        self.step = step

