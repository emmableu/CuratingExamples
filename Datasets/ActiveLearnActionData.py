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
import operator
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
            body.loc[i] = {"X": self.X[i], 'label': int(self.y[i])}
        n = len(self.y)
        body["code"] = ["undetermined"] * n
        body["time"] = [0] * n
        body["fixed"] = [0] * n
        return body

    def get_numbers(self):
        total = len(self.body["code"]) - self.last_pos - self.last_neg
        pos = Counter(self.body["code"])[1] - self.last_pos
        neg = Counter(self.body["code"])[0] - self.last_neg
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



    def train(self, step):
        print("--------------train session---------")
        poses = np.where(np.array(self.body['code']) == 1)[0]
        negs = np.where(np.array(self.body['code']) == 0)[0]
        validation_ids = list(poses) + list(negs)

        unlabeled = np.where(np.array(self.body['code']) == "undetermined")[0]
        print("poses: ", poses)

        try:
            unlabeled_train = np.random.choice(unlabeled, size=len(poses))
            train_ids1 = list(poses) + list(negs) + list(unlabeled_train)

        except:
            train_ids1 = list(poses) + list(negs)

        if len(poses)==1:
            best_model = bernoulli_nb
        else:
            model_recall = get_model_recall(train_ids1, validation_ids, self.X, self.y, model_list)
            print(model_recall)
            best_model = max(model_recall.items(), key=operator.itemgetter(1))[0]

        current_model = best_model.model
        current_model.fit(self.X[validation_ids], self.y[validation_ids])
        rest_data_ids = self.get_opposite(range(len(self.y)), validation_ids)
        # pos_at = list(current_model.classes_).index(1)
        # prob = current_model.predict_proba(self.X[rest_data_ids])[:, pos_at]
        # prediction = current_model.predict(self.X[rest_data_ids])
        # print('prediction', prediction)
        candidate_id_voi_dict = {}
        for candidate_id  in rest_data_ids:
            train_ids2 = list([candidate_id]) + list(train_ids1)
            validation_ids2 = list(poses)
            voi = get_VOI(train_ids2, validation_ids2, self.X, self.y, model_list)
            candidate_id_voi_dict[candidate_id] = voi
        print("candidate_id_voi_dict: ", candidate_id_voi_dict)

        # uncertain_id, uncertain_prob = self.uncertain(current_model, step, self.X)
        # certain_id, certain_prob = self.certain(current_model, step, self.X)
        # #
        # positive_id = self.get_positive_id()
        # # TODO: finish this get_positive_ID FUNCTION to get the positive id corresponding to X_val, y_val
        # # TODO: find all the positive_id
        # # TODO: use VOI calculation to use the one we need
        # return candidate_id

        return uncertain_id, uncertain_prob, certain_id, certain_prob

    ## Get certain ##
    def certain(self,clf, step, all_X):
        if list(clf.classes_) == [0]:
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
                if self.body.label[ele] == 1:
                    return [ele]
            r = self.random(step= 10)

    def start_as_1_neg(self):
        r = self.random(10)
        while True:
            for ele in r:
                if self.body.label[ele] == 0:
                    return [ele]
            r = self.random(step= 10)

    ## Code candidate studies ##
    def code(self,id,label):
        if self.body['code'][id] == label:
            self.body['fixed'][id] = 1
        self.body["code"][id] = label
        self.body["time"][id] = time.time()

