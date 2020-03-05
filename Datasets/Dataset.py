import sys
sys.path.append("/Users/wwang33/Documents/IJAIED20/CuratingExamples/my_module")
from save_load_pickle import *
import pandas as pd
from CodeShape import *
from Test import *
from ActionData import *
import os

import pandas as pd
import warnings
import sklearn
from classifiers.Baseline import BaselineModel
from classifiers.knn_classifiers.KNN import KNNModel
from classifiers.lr_classifiers.LogisticRegression import LRModel
from classifiers.svm_classifiers.SVM_C import SVMCModel
from classifiers.svm_classifiers.SVM_Nu import SVMNuModel
from classifiers.svm_classifiers.SVM_Linear import SVMLinearModel
from classifiers.decision_tree_classifiers.DecisionTree import DecisionTreeModel
from classifiers.emsemble_classifiers.AdaBoost import AdaBoostModel
from classifiers.emsemble_classifiers.Bagging import BaggingModel
from classifiers.emsemble_classifiers.RandomForest import RandomForestModel
from classifiers.bayes_classifiers.GaussianNB import GaussianNBModel
from classifiers.bayes_classifiers.BernoulliNB import BernoulliNBModel
from classifiers.bayes_classifiers.MultinomialNB import MultinomialNBModel
from classifiers.bayes_classifiers.ComplementNB import ComplementNBModel
from classifiers.neural_network_classifiers.MLPModel import MLPModel

warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

baseline = BaselineModel()
knn = KNNModel()
lr = LRModel()
svm_c = SVMCModel()
svm_nu = SVMNuModel()
svm_linear = SVMLinearModel()
dt = DecisionTreeModel()
adaboost = AdaBoostModel()
bagging = BaggingModel()
rf = RandomForestModel()
gaussian_nb = GaussianNBModel()
bernoulli_nb = BernoulliNBModel()
multi_nb = MultinomialNBModel()
complement_nb = ComplementNBModel()
mlp = MLPModel()

# Removed baseline from models in case baseline result changes
no_tuning_models = [baseline, knn, lr, svm_c, svm_nu, svm_linear, dt, adaboost, bagging, rf, gaussian_nb,
                    bernoulli_nb, multi_nb, complement_nb, mlp]




class LabelData(object):
    """docstring for LabelData."""

    def __init__(self, action_name):
        self.labeldf = pd.DataFrame(columns=['pid', 'label'])
        self.label = action_name

    def populate(self, data, action_name):
        for i in data.index:
            if data.at[i, 'good'] and data.at[i, 'good'] == True:
                pid = data.at[i, 'pid']
                new_row = {
                    'pid': pid,
                    'label': ('yes' if data.at[i, action_name] == True else 'no')
                }
                self.labeldf.loc[len(self.labeldf)] = new_row
        return self.labeldf
        # save_obj(self.labeldf, label, cwd, 'game_labels_' + str(total_games))


class Dataset:

    def __init__(self, total, code_shape_p_q_list, embedding_param = None, allow_gap = True):
        self.root_dir = "/Users/wwang33/Documents/IJAIED20/CuratingExamples/"
        self.total = total
        self.code_shape_p_q_list = code_shape_p_q_list
        self.embedding_param = embedding_param
        self.file_path = self.root_dir + "Datasets/data/game_label_" + str(total) + ".csv"
        self.data = pd.read_csv(self.file_path)
        self.allow_gap = allow_gap

    def get_code_shape_from_code(self, json_code, code_shape_p_q_list, allow_gap=True):
        if allow_gap:
            test_shape = combination(json_code, code_shape_p_q_list)
        else:
            test_shape = get_code_shape(json_code, 'targets', code_shape_p_q_list)
        return test_shape


    def create_code_state(self):
        '''
        :return: pd DataFrame, columns = ['pid', 'codeshape_count_dict']
        example row: ['2312424', {'sprite|repeat': 3, 'sprite|repeat|else': 1}]
        '''
        code_state = pd.DataFrame(columns = ['pid', 'codeshape_count_dict'] )
        for i in self.data.index:
            print(i)
            pid = self.data.at[i, 'pid']
            json_code = get_json(pid)
            a = self.get_code_shape_from_code(json_code, self.code_shape_p_q_list)
            # print(a)
            new_row = {"pid": pid, "codeshape_count_dict": a}
            code_state.loc[len(code_state)] = new_row
        save_pickle(code_state, "code_state" + str(self.code_shape_p_q_list), self.root_dir+"Datasets/data", "game_labels_" + str(self.total))

        return code_state


    def get_result(self):
        code_state = self.create_code_state()
        action_name_s = ['keymove', 'jump', 'costopall', 'wrap', 'cochangescore', 'movetomouse', 'moveanimate']
        action_name_s = ['cochangescore']
        for action_name in action_name_s:
            print("action_name: ", action_name)
            self.action_data = ActionData(code_state = code_state, game_label = self.data , action_name = action_name)
            self.action_data.get_yes_patterns()
            self.action_data.get_pattern_statistics()
            self.action_data.model_performance()
            save_dir = self.root_dir + "Datasets/data/" + "game_labels_" \
                       + str(self.total) + str(self.code_shape_p_q_list) + "/" + action_name
            x, y = self.action_data.get_xy()
            for model in no_tuning_models:
                for test_size in [3/4, 2/3, 1/2, 1/3]:
                    model.get_and_save_performance(x, y, save_dir, test_size)
                    print("--------------"+  action_name + model.get_name()  + test_size+ "--------------")
                    print(model.get_confusion_matrix())
                    print(model.get_performance())














