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

model_list = [bernoulli_nb, knn, lr]
root_dir = "/Users/wwang33/Documents/IJAIED20/CuratingExamples/"


class ActiveLearnActionData(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.body = self.get_body()
        self.last_pos = 0
        self.last_neg = 0
        self.record = {"x": [], "pos": []}
        self.session = 0




    def get_body(self):
        body = pd.DataFrame(columns=['X', 'label'])
        for i in range(len(self.y)):
            body.loc[i] = {"X": self.X[i], 'label': int(self.y[i])}
        n = len(self.y)
        body["code"] = ["undetermined"] * n
        body["session"] = [-1] * n
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



    def train(self):
        self.session += 1
        print("--------------train session ", self.session, "-----------------")
        print("body: ")
        print(self.body)
        poses = np.where(np.array(self.body['code']) == 1)[0]
        negs = np.where(np.array(self.body['code']) == 0)[0]
        validation_ids = list(poses) + list(negs)

        unlabeled = np.where(np.array(self.body['code']) == "undetermined")[0]
        print("poses: ", poses)

        try:
            unlabeled_train = np.random.choice(unlabeled, size=len(poses))
            train_ids1 = list(poses) + list(negs) + list(unlabeled_train)
            code_array = np.array(self.body.code.to_list())
            code_array[unlabeled_train] = '0'
            assert bool(
                set(code_array[validation_ids]) & set(['undetermined'])) == False, "train set includes un-coded data!"

        except:
            train_ids1 = list(poses) + list(negs)

        if len(poses)==1:
            best_model = bernoulli_nb
        else:
            model_f1 = get_model_f1(train_ids1, validation_ids, self.X, self.y, model_list)
            print(model_f1)
            best_model = max(model_f1.items(), key=operator.itemgetter(1))[0]

        current_model = best_model.model
        code_array = np.array(self.body.code.to_list())
        assert bool(set(code_array[validation_ids]) & set(['undetermined'])) == False, "validation set includes un-coded data!"
        current_model.fit(self.X[validation_ids], code_array[validation_ids])
        rest_data_ids = get_opposite(range(len(self.y)), validation_ids)
        # print(list(current_model.classes_))
        try:
            pos_at = list(current_model.classes_).index('1')
        except:
            pos_at = list(current_model.classes_).index(1)

        prob = current_model.predict_proba(self.X)[:, pos_at]

        if len(rest_data_ids) == 1:
            return rest_data_ids[0]

        candidate_id_voi_dict = {}
        for candidate_id in tqdm(rest_data_ids):
            code_array = np.array(self.body.code.to_list())
            train_ids2 = list([candidate_id]) + list(validation_ids)
            code_array[candidate_id] = '1'
            assert bool(set(code_array[validation_ids]) & set(['undetermined'])) == False, "train set includes un-coded data!"
            voi = get_VOI(train_ids2, validation_ids, self.X, code_array, model_list)
            candidate_id_voi_dict[candidate_id] = voi
        determine_dict = candidate_id_voi_dict
        if not determine_dict:
            return -1
        for i in determine_dict:
            determine_dict[i]  = determine_dict[i] + (prob[i])
        print("candidate_id_voi_dict: ", candidate_id_voi_dict)
        print("determine_dict: ", determine_dict)
        best_candidate = max(determine_dict.items(), key=operator.itemgetter(1))[0]
        return best_candidate

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

    def start_as_3_pos(self):
        r = self.random(10)
        while True:
            ele_list = []
            for ele in r:
                if self.body.label[ele] == 1:
                    ele_list.append(ele)
                if len(ele_list) == 3:
                    return ele_list
            r = self.random(step= 10)



    def start_as_1_neg(self):
        r = self.random(10)
        while True:
            for ele in r:
                if self.body.label[ele] == 0:
                    return [ele]
            r = self.random(step= 10)

    ## Code candidate studies ##
    def code(self,id):
        self.body['code'][id] = self.body['label'][id]
        self.body["session"][id] = self.session

