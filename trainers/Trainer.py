from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_predict
import sys

from preprocessors.PreProcessor import PreProcessor

sys.path.append("/Users/wwang33/Documents/IJAIED20/CuratingExamples/my_module")
from save_load_pickle import *
sys.path.append("/Users/wwang33/Documents/IJAIED20/CuratingExamples")
from model_evaluation import *
from preprocessors.PreProcessor import *
from preprocessors.DPMPreProcessor import *
from sklearn.model_selection import KFold, cross_val_score, LeaveOneOut
from samplers.RandomSampler import RandomSampler
from samplers.Sampler import Sampler
from samplers.CertaintySampler import CertaintySampler
from samplers.UncertaintySampler import UncertaintySampler


class Trainer:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.X_pool =None
        self.y_pool = None
        self.pool_id_list = None
        self.pre_processor = None
        self.sampler =None
        self.step = None


    def populate(self, X, y, train_id_list, pool_id_list):
        self.comment = "Parent class trainer"
        self.X_train = X[train_id_list]
        self.y_train = y[train_id_list]
        self.X_pool = X[pool_id_list]
        self.y_pool = y[pool_id_list]
        self.pool_id_list = pool_id_list
        self.pre_processor = PreProcessor()
        self.sampler = RandomSampler()
        self.step = 10

    def __trainer_preprocess(self):
        return self.pre_processor.preprocess(self.X_train, self.y_train)

    def trainer_preprocess_add_constant(self):
        temp = self.__trainer_preprocess()
        self.X_train = add_constant_term(temp)

    def train(self):
        self.trainer_preprocess_add_constant()
        best_model = svm_linear
        best_model.model.fit(self.X_train, self.y_train)
        pos_at = list(best_model.model.classes_).index(1)
        return best_model, pos_at

    def get_pool_prob(self, model_obj, pos_at):
        self.X_pool = add_constant_term(self.X_pool)
        prob = model_obj.model.predict_proba(self.X_pool)[:, pos_at]
        return prob

    def learn_and_sample(self):
        model_obj, pos_at = self.train()
        prob = self.get_pool_prob(model_obj, pos_at)
        self.sampler.populate(self.step, self.pool_id_list, prob)
        return model_obj, self.sampler.sample()

