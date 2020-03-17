from __future__ import print_function, division
try:
    import cPickle as pickle
except:
    import pickle
import sys
sys.path.append("/Users/wwang33/Documents/IJAIED20/CuratingExamples/my_module")
import sys
sys.path.append("/Users/wwang33/Documents/IJAIED20/CuratingExamples/my_module")
from alms_helper import *
from ActionData import *
import numpy as np
from collections import Counter

from sklearn.model_selection import train_test_split
from tqdm import tqdm

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
model_list = [baseline, knn, lr, svm_c, svm_nu, svm_linear, dt, adaboost, bagging, rf, gaussian_nb,
                    bernoulli_nb, multi_nb, complement_nb, mlp]

model_list = [baseline, knn, lr]
root_dir = "/Users/wwang33/Documents/IJAIED20/CuratingExamples/"


class ActiveLearnActionData(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.body = self.get_body()
        self.last_pos = 0
        self.last_neg = 0
        self.record = {"x": [], "pos": []}




    def get_body(self):
        body = pd.DataFrame(columns=['X', 'label'])
        for i in range(len(self.y)):
            body.loc[i] = {"X": self.X[i], 'label': "yes" if int(self.y[i]) == 1 else "no"}
        n = len(self.y)
        body["code"] = ["undetermined"] * n
        body["time"] = [0] * n
        body["fixed"] = [0] * n
        return body

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

    def train(self, step):
        print("--------------train session---------")
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

        train_pid = self.body['X'][sample].to_list()
        # print("train_pid: ", train_pid)
        test_pid = self.get_opposite(self.body['X'], sample)
        # print("test_pid:", test_pid)

        # all_X, all_y, X_train, X_test, y_train, y_test = self.get_x_y_train_test(train_pid, test_pid)
        # print("y_train: ", y_train)
        best_model = get_best_model(self.X, self.y, model_list)
        current_model = best_model.model
        current_model.fit(self.X, self.y)

        uncertain_id, uncertain_prob = self.uncertain(current_model, step, self.X)
        certain_id, certain_prob = self.certain(current_model, step, self.X)
        # #
        # positive_id = self.get_positive_id()
        # # TODO: finish this get_positive_ID FUNCTION to get the positive id corresponding to X_val, y_val
        # # TODO: find all the positive_id
        # # TODO: use VOI calculation to use the one we need
        # return candidate_id

        return uncertain_id, uncertain_prob, certain_id, certain_prob

    ## Get certain ##
    def certain(self,clf, step, all_X):
        if list(clf.classes_) == ['no']:
            print("attention, all classes are no")

            return self.random(step), [0.001]*step
        print( list(clf.classes_))
        pos_at = list(clf.classes_).index(1)

        if len(self.pool)==0:
            return [],[]
        prob = clf.predict_proba(all_X[self.pool])[:,pos_at]
        order = np.argsort(prob)[::-1][:step]

        return np.array(self.pool)[order],np.array(prob)[order]

    ## Get uncertain ##
    def uncertain(self, clf, step, all_X):
        print( list(clf.classes_))
        print(len(np.unique(list(clf.classes_))))
        if len(np.unique(list(clf.classes_))) == 1:
            return self.random(step), [0.001]*step
        pos_at = list(clf.classes_).index(1)

        if len(self.pool)==0:
            return [],[]
        prob = clf.predict_proba(all_X[self.pool])[:, pos_at]
        print(prob)
        # train_dist = clf.decision_function(all_X[self.pool])
        order = np.argsort(np.abs(prob))[:step]  ## uncertainty sampling by distance to decision plane
        return np.array(self.pool)[order], np.array(prob)[order]

    ## Get random ##
    def random(self, step):
        return np.random.choice(self.pool,size=np.min((step,len(self.pool))),replace=False)


    def start_as_1_pos(self):
        r = self.random(10)
        while True:
            for ele in r:
                if self.body.label[ele] == "yes":
                    return [ele]
            r = self.random(step= 10)



    ## Code candidate studies ##
    def code(self,id,label):
        if self.body['code'][id] == label:
            self.body['fixed'][id] = 1
        self.body["code"][id] = label
        self.body["time"][id] = time.time()

