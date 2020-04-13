from trainers.Trainer import *
import numpy as np
from sklearn.model_selection import ShuffleSplit, LeaveOneOut
from trainers.ModelSelectionTrainer import *


class CrossValidationTrainer(ModelSelectionTrainer):
    def __init__(self):
        super().__init__()
        self.cv_total = 10
        self.ss = ShuffleSplit(n_splits=self.cv_total, test_size=1/self.cv_total, random_state=0)
        self.dpm = False
        # self.ss = LeaveOneOut()


    def populate(self, X, y):
        train_id_list = list(range(len(y)))
        pool_id_list = []
        super().populate(X, y, train_id_list, pool_id_list)
        self.pre_processor = PreProcessor()

    def cross_val_get_score(self):
        performance_temp = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "accuracy": 0, "precision": 0, "recall": 0,
                            "f1": 0, "auc": 0}
        for train_id, test_id in self.ss.split(self.train_id_list):
            x_train_new, x_test_new = self.pre_processor.preprocess(self.X_train[train_id], self.y_train[train_id],
                                                                    self.X_train[test_id])
            # print("x_train: ", x_train_new)
            # print("x_test: ", x_test_new)
            # print("y_train: ", self.y_train[train_id])
            # print("y_test: ",self.y_train[test_id])
            if self.dpm:
                add_performance = or_existence.get_performance(x_train_new, x_test_new, self.y_train[train_id],
                                                     self.y_train[test_id])
            else:
                add_performance = lr.get_performance(x_train_new, x_test_new, self.y_train[train_id],
                                                         self.y_train[test_id])
            performance_temp = add_by_ele(performance_temp, add_performance)

        performance = get_dict_average(performance_temp, cv_total=self.cv_total)

        print(performance)
        return performance


class DPMCrossValidationTrainer(CrossValidationTrainer):
    def __init__(self):
        super().__init__()
        self.dpm = True

    def populate(self, X, y):
        super().populate(X, y)
        self.pre_processor = DPMPreProcessor()



class MSCrossValidationTrainer(CrossValidationTrainer):
    def __init__(self):
        super().__init__()

    def populate(self, X, y):
        super().populate(X, y)
        self.pre_processor = DPMPreProcessor()