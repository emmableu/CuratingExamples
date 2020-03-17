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
        self.pid_list = load_obj('pid', base_dir, "")
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
            for pid in pid_list:
                code_shape = code_state.at[pid, "code_state" + str(code_shape_p_q)]
                new_pattern_s = code_shape.keys()
                # print(new_pattern_s)
                pattern_set = atomic_add(pattern_set, new_pattern_s)
            all_pattern_keys["code_state" + str(code_shape_p_q)] = pattern_set
        print(all_pattern_keys)
        save_obj(all_pattern_keys, "pattern_set",  base_dir, "code_state" + str(self.code_shape_p_q_list))


    #
    # def __get_code_shape_from_pid(self, pid, code_state):
    #     return code_state[pid]

    def save_x_y_to_hard_drive(self, selected_p_q_list, baseline = True):
        code_state = load_obj("code_state", base_dir, "code_state" + str(self.code_shape_p_q_list))
        action_name_s = ['keymove', 'jump', 'costopall', 'wrap', 'cochangescore', 'movetomouse', 'moveanimate']
        game_label = pd.read_csv(base_dir + "/game_label_415.csv")
        for action_name in tqdm(action_name_s):
            print("action_name: ", action_name)
            action_data = ActionData(code_state=code_state, game_label=game_label, action_name=action_name, selected_p_q_list=selected_p_q_list)
            for test_size in tqdm(test_size_list):
                if test_size == 0:
                    cv_total = 1
                else:
                    cv_total = 10
                for fold in tqdm(range(cv_total)):
                    if baseline:
                        save_dir = base_dir + "/cv/test_size" + str(test_size) + "/fold" + str(
                            fold) + "/code_state" + str(selected_p_q_list) + "baseline"  + "/" + action_name
                    else:
                        save_dir = base_dir + "/cv/test_size" + str(test_size)+ "/fold" + str(fold) + "/code_state" + str(self.code_shape_p_q_list) + "/" + action_name
                    train_pid, test_pid = get_train_test_pid(test_size, fold)
                    for p in train_pid:
                        if p not in self.pid_list:
                            print("pid not in pid_list!", p)
                    action_data.save_x_y_train_test(train_pid, test_pid, save_dir, baseline)




    def get_result(self, baseline):
        action_name_s = ['keymove', 'jump', 'costopall', 'wrap', 'cochangescore', 'movetomouse', 'moveanimate']

        for action_name in tqdm(action_name_s):
            print("action_name: ", action_name)
            if baseline:
                save_dir = base_dir + "/code_state" + str(self.code_shape_p_q_list) + "baseline" + "/result/" + action_name
            else:
                save_dir = base_dir + "/code_state" + str(
                    self.code_shape_p_q_list) + "" + "/result/" + action_name

            for model in tqdm(no_tuning_models):
                # for test_size in tqdm([0.9]):
                for test_size in test_size_list[:-1]:
                    tp, tn, fp, fn, accuracy, precision, recall, f1, roc_auc = 0, 0, 0, 0, 0, 0, 0, 0, 0
                    performance_temp = {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": accuracy, "precision": precision,
                            "recall": recall,
                            "f1": f1, "auc": roc_auc}
                    cv_total = 10
                    for fold in range(cv_total):
                        if baseline:
                            get_dir = base_dir + "/cv/test_size" + str(test_size) + "/fold" + str(
                                fold) + "/code_state" + str(self.code_shape_p_q_list) + "baseline" + "/" + action_name
                        else:
                            get_dir = base_dir + "/cv/test_size" + str(test_size) + "/fold" + str(
                                fold) + "/code_state" + str(self.code_shape_p_q_list) + "/" + action_name

                        X_train, X_test, y_train, y_test = get_x_y_train_test(get_dir)
                        if len(np.unique(y_train)) == 1:
                            add_performance = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "accuracy": 0, "precision": 0, "recall": 0,
                "f1": 0, "auc": 0}
                        else:
                            add_performance = model.get_performance(X_train, X_test, y_train, y_test)
                        performance_temp = add_by_ele(performance_temp, add_performance)

                    performance = get_dict_average(performance_temp, cv_total)
                    model.save_performance(save_dir, test_size, performance)
















