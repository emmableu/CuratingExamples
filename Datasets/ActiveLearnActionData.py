from __future__ import print_function, division
try:
    import cPickle as pickle
except:
    import pickle
from pdb import set_trace
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
from collections import Counter
from sklearn import svm
import matplotlib.pyplot as plt
import time
import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import sys
sys.path.append("/Users/wwang33/Documents/IJAIED20/CuratingExamples/my_module")
from Test import *
import pandas as pd
import sys
sys.path.append("/Users/wwang33/Documents/IJAIED20/CuratingExamples/my_module")
from save_load_pickle import *
from alms_helper import *
import pandas as pd
from CodeShape import *
from Test import *
from ActionData import *
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm


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

root_dir = "/Users/wwang33/Documents/IJAIED20/CuratingExamples/"


class ActiveLearnActionData(object):

    def __init__(self, code_state, game_label, action_name, code_shape_p_q_list):
        self.code_state = code_state
        self.action_name = action_name
        self.enough = 10
        self.body = self.get_body(game_label)
        self.est_num = Counter(self.body["label"])["yes"]
        self.last_pos = 0
        self.last_neg = 0
        self.record = {"x": [], "pos": []}
        self.est = []
        self.atleast = 3
        self.code_shape_p_q_list = code_shape_p_q_list
        self.baseline = True



    def get_body(self, game_label):
        body = pd.DataFrame(columns=['pid', 'label'])
        for i in game_label.index:
            pid = game_label.at[i, 'pid']
            new_row = {
                'pid': pid,
                'label': ('yes' if game_label.at[i, self.action_name] == True else 'no'),
            }
            body.loc[len(body)] = new_row
        n = len(game_label)
        body["code"] = ["undetermined"] * n
        body["time"] = [0] * n
        body["fixed"] = [0] * n
        return body

    def __get_pattern_df(self, pattern, train_pid):
        pool = self.body[self.body.pid.isin(train_pid)].reset_index(drop=True)

        pattern_df = pd.DataFrame(columns=['pid', 'occurance', 'code'])
        for i in pool.index:
            pid = pool.at[i, 'pid']
            code_shape = self.__get_code_shape_from_pid(pid, self.code_state)

            try:
                occurance = code_shape[pattern]
            except KeyError:
                occurance = 0
            # print(occurance)
            if pool.at[i, "code"] == "yes":
                code = 'yes'
            elif pool.at[i, "code"] == "no" or pool.at[i, "code"] == "undetermined":
                code = 'no'
            new_row = {'pid': pid, 'occurance': occurance, 'code': code}
            # print("new_row" , new_row)
            # print("pattern_df",pattern_df)
            pattern_df.loc[len(pattern_df)] = new_row
        return pattern_df

    def get_pattern_statistics(self, train_pid):
        pattern_set = load_obj("pattern_set", root_dir + "Datasets/data",
                               "game_labels_" + str(415) + "/code_state" + str(self.code_shape_p_q_list))
        if self.baseline:
            return pattern_set
        significant_patterns = []
        for pattern in pattern_set:
            pattern_df = self.__get_pattern_df(pattern, train_pid)
            test = Test(pattern_df)

            if test.freq_compare_test() == "discard":
                continue
            elif test.freq_compare_test() or test.chi_square_test() or test.kruskal_wallis_test():
                significant_patterns.append(pattern)
        print(len(significant_patterns))
        print(significant_patterns)
        return significant_patterns

    def get_x_y_train_test(self, train_pid, test_pid):
        significant_patterns = self.get_pattern_statistics(train_pid)
        num_patterns = len(significant_patterns)

        train_df = self.body[self.body.pid.isin(train_pid)].reset_index(drop=True)
        print("train_df: ", train_df)

        test_df = self.body[self.body.pid.isin(test_pid)].reset_index(drop=True)
        print("test_df: ", test_df)
        def get_xy(df):
            x = np.zeros((len(df.index), num_patterns))
            y = ["unassigned"]*(len(df.index))
            for game_index, i in enumerate(df.index):
                pid = self.body.at[i, 'pid']
                code_shape = self.__get_code_shape_from_pid(pid, self.code_state)
                for pattern_index, p in enumerate(significant_patterns):
                    try:
                        occurance = code_shape[p]
                    except KeyError:
                        occurance = 0
                    x[game_index][pattern_index] = occurance
                if df.at[i, "code"] == "yes":
                    y[game_index] = "yes"
                elif df.at[i, "code"] == "no" or df.at[i, "code"] == "undetermined":
                    y[game_index] = "no"
            return x, y

        X_train, y_train = get_xy(train_df)
        X_test, y_test = get_xy(test_df)
        all_X, all_y = get_xy(self.body)

        return all_X, all_y, X_train, X_test, y_train, y_test




    def get_numbers(self):
        total = len(self.body["code"]) - self.last_pos - self.last_neg
        pos = Counter(self.body["code"])["yes"] - self.last_pos
        neg = Counter(self.body["code"])["no"] - self.last_neg
        try:
            tmp=self.record['x'][-1]
        except:
            tmp=-1
        if int(pos+neg)>tmp:
            self.record['x'].append(int(pos+neg))
            self.record['pos'].append(int(pos))
        self.pool = np.where(np.array(self.body['code']) == "undetermined")[0]
        self.labeled = list(set(range(len(self.body['code']))) - set(self.pool))
        return pos, neg, total

    def get_opposite(self, l, ind):
        opposite_list = []
        for i, e in enumerate(l):
            if i in ind:
                continue
            else:
                opposite_list.append(e)
        return opposite_list

    def no_pole_train(self, step, model):
        poses = np.where(np.array(self.body['code']) == "yes")[0]
        negs = np.where(np.array(self.body['code']) == "no")[0]
        left = poses
        decayed = list(left) + list(negs)
        unlabeled = np.where(np.array(self.body['code']) == "undetermined")[0]
        try:
            unlabeled = np.random.choice(unlabeled, size=np.max((len(decayed)//2, len(left), self.atleast)),
                                     replace=False)
        except:
            pass

        sample = list(decayed) + list(unlabeled)

        train_pid = self.body['pid'][sample].to_list()
        print("train_pid: ", train_pid)
        test_pid = self.get_opposite(self.body['pid'], sample)
        print("test_pid:", test_pid)

        all_X, all_y, X_train, X_test, y_train, y_test = self.get_x_y_train_test(train_pid, test_pid)
        print("y_train: ", y_train)
        model.fit(X_train, y_train)

        uncertain_id, uncertain_prob = self.uncertain(model, step, all_X)
        certain_id, certain_prob = self.certain(model, step, all_X)

        return uncertain_id, uncertain_prob, certain_id, certain_prob





    def alms_train(self, step, model):
        '''
        get X_train, y_train,
        train all models on tau_t, compute P_t(M)
        for all x, y that belong to P:
            compute x's estimated label y
            E(VOI) = computeVOI(t,v)
            E(VOI)model = computeVOI(T,V,x)
        '''
            model.fit(X_train, y_train)

        uncertain_id, uncertain_prob = self.uncertain(model, step, all_X)
        certain_id, certain_prob = self.certain(model, step, all_X)




    ## Get certain ##
    def certain(self,clf, step, all_X):
        if list(clf.classes_) == ['no']:
            print("attention, all classes are no")

            return self.random(step), [0.001]*step
        print( list(clf.classes_))
        pos_at = list(clf.classes_).index("yes")

        if len(self.pool)==0:
            return [],[]
        prob = clf.predict_proba(all_X[self.pool])[:,pos_at]
        order = np.argsort(prob)[::-1][:step]

        return np.array(self.pool)[order],np.array(prob)[order]

    ## Get uncertain ##
    def uncertain(self,clf, step, all_X):
        # print( list(clf.classes_))
        if list(clf.classes_) == ['no']:
            return self.random(step), [0.001]*step
        pos_at = list(clf.classes_).index("yes")

        if len(self.pool)==0:
            return [],[]
        prob = clf.predict_proba(all_X[self.pool])[:, pos_at]
        # print("all_X: ", all_X)
        # print("all_X[self.pool]: ", all_X[self.pool])
        # print("clf.predict_proba(all_X[self.pool]): ", clf.predict_proba(all_X[self.pool]))
        # print(("clf.predict_proba(all_X[self.pool])[:, pos_at] : ", clf.predict_proba(all_X[self.pool])[:, pos_at]))
        # print(prob)
        train_dist = clf.decision_function(all_X[self.pool])
        order = np.argsort(np.abs(train_dist))[:step]  ## uncertainty sampling by distance to decision plane
        return np.array(self.pool)[order], np.array(prob)[order]

    ## Get random ##
    def random(self, step):
        return np.random.choice(self.pool,size=np.min((step,len(self.pool))),replace=False)


    def start_as_1_pos(self):
        r = self.random(10)
        while True:
            for ele in r:
                if self.body.label[ele] == 'yes':
                    return [ele]
            r = self.random(step= 10)



    ## Code candidate studies ##
    def code(self,id,label):
        if self.body['code'][id] == label:
            self.body['fixed'][id] = 1
        self.body["code"][id] = label
        self.body["time"][id] = time.time()

