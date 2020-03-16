import sys
sys.path.append("/Users/wwang33/Documents/IJAIED20/CuratingExamples/my_module")
from save_load_pickle import *
import pandas as pd
from CodeShape import *
from Test import *
from ActionData import *
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import sys, os
root = os.getcwd().split("src")[0] + "src/src/util"
sys.path.append(root)
from ActiveLearnActionData import *
import pickle
import os
sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets')
from Dataset import *

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
no_tuning_models = [baseline, knn, lr, svm_c, svm_linear, dt, adaboost, bagging, rf, gaussian_nb,
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

    def __init__(self, code_shape_p_q_list, embedding_param = None, allow_gap = False):
        self.code_shape_p_q_list = code_shape_p_q_list
        self.embedding_param = embedding_param
        # self.file_path = self.root_dir + "Datasets/data/SnapASTData/game_label_" + str(total) + ".csv"
        # self.data = pd.read_csv(self.file_path)
        # self.data = self.data[self.data.good == True].reset_index(drop = True)
        # print(self.data)
        self.allow_gap = allow_gap


    def create_code_state(self):
        pid = load_obj('pid', base_dir, "")
        json_folder = root_dir+ "/Datasets/data/SnapJSON_413/"
        code_state_df = pd.DataFrame(index = pid, columns = ["code_state" + str(i) for i in self.code_shape_p_q_list])

        for p in tqdm(pid):
            file = json_folder + p + ".xml.json"
            codegraph =  CodeGraph(file, self.code_shape_p_q_list)
            data = codegraph.collect_all_pqgrams(self.code_shape_p_q_list)
            code_state_df.loc[p] = data

        # print(code_state_df)
        save_pickle(code_state_df, "code_state",  base_dir, "code_state" + str(self.code_shape_p_q_list))


    def get_all_pattern_keys(self):
        code_state = load_obj( "code_state",  base_dir, "code_state" + str(self.code_shape_p_q_list))
        pid_list = load_obj('pid', base_dir, "")
        all_pattern_keys = {}
        for code_shape_p_q in self.code_shape_p_q_list:
            pattern_set = set()
            new_pattern_s = code_shape.keys()
            for pid in pid_list:
                code_shape = code_state.at[pid, "code_state" + str(code_shape_p_q)]
                pattern_set = atomic_add(new_pattern_s, pattern_set)
            all_pattern_keys["code_state" + str(code_shape_p_q)] = pattern_set
        save_obj(all_pattern_keys, "pattern_set",  base_dir, "code_state" + str(self.code_shape_p_q_list))


    #
    # def __get_code_shape_from_pid(self, pid, code_state):
    #     return code_state[pid]

    def save_x_y_to_hard_drive(self, baseline = True):
        code_state = load_obj("code_state", base_dir, "code_state" + str(self.code_shape_p_q_list))
        action_name_s = ['keymove', 'jump', 'costopall', 'wrap', 'cochangescore', 'movetomouse', 'moveanimate']

        for action_name in tqdm(action_name_s):
            print("action_name: ", action_name)
            self.action_data = ActionData(code_state=code_state, game_label=self.data, action_name=action_name, code_shape_p_q_list=self.code_shape_p_q_list)
            for test_size in tqdm(test_size_list):
                cv_total = int(max(1 / (1 - test_size), 1 / test_size))
                for fold in tqdm(range(cv_total)):
                    if baseline:
                        save_dir = root_dir + "Datasets/data/SnapASTData/cv/test_size" + str(test_size) + "/fold" + str(
                            fold) + "/code_state" + str(self.code_shape_p_q_list) + "baseline"  + "/" + action_name
                    else:
                        save_dir = root_dir + "Datasets/data/SnapASTData/cv/test_size" + str(test_size)+ "/fold" + str(fold) + "/code_state" + str(self.code_shape_p_q_list) + "/" + action_name
                    train_pid, test_pid = get_train_test_pid(test_size, fold)
                    self.action_data.save_x_y_train_test(train_pid, test_pid, save_dir, baseline)




    def get_result(self, baseline):
        action_name_s = ['keymove', 'jump', 'costopall', 'wrap', 'cochangescore', 'movetomouse', 'moveanimate']

        for action_name in tqdm(action_name_s):
            print("action_name: ", action_name)
            if baseline:
                save_dir = self.root_dir + "Datasets/data/SnapASTData/" + "game_labels_" \
                           + str(self.total) + "/code_state" + str(self.code_shape_p_q_list) + "baseline" + "/" + action_name
            else:
                save_dir = self.root_dir + "Datasets/data/SnapASTData/" + "game_labels_" \
                           + str(self.total) + "/code_state" + str(self.code_shape_p_q_list) + "/" + action_name

            for model in tqdm(no_tuning_models):
                # for test_size in tqdm([0.9]):
                for test_size in tqdm([0.9, 3/4, 2/3, 1/2, 1/3]):
                    tp, tn, fp, fn, accuracy, precision, recall, f1, roc_auc = 0, 0, 0, 0, 0, 0, 0, 0, 0
                    performance_temp = {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": accuracy, "precision": precision,
                            "recall": recall,
                            "f1": f1, "auc": roc_auc}
                    cv_total = int(max(1 / (1 - test_size), 1 / test_size))
                    for fold in range(cv_total):
                        if baseline:
                            get_dir = root_dir + "Datasets/data/SnapASTData/cv/test_size" + str(test_size) + "/fold" + str(
                                fold) + "/code_state" + str(self.code_shape_p_q_list) + "baseline" + "/" + action_name
                        else:
                            get_dir = root_dir + "Datasets/data/SnapASTData/cv/test_size" + str(test_size) + "/fold" + str(
                                fold) + "/code_state" + str(self.code_shape_p_q_list) + "/" + action_name

                        X_train, X_test, y_train, y_test = get_x_y_train_test(get_dir)
                        add_performance = model.get_performance(X_train, X_test, y_train, y_test)
                        performance_temp = add_by_ele(performance_temp, add_performance)

                    performance = get_dict_average(performance_temp, cv_total)
                    model.save_performance(save_dir, test_size, performance)
















