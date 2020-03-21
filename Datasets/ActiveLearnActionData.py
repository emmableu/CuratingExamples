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

model_list = [bernoulli_nb, multi_nb, complement_nb,gaussian_nb, adaboost, svm_linear, lr ]
root_dir = "/Users/wwang33/Documents/IJAIED20/CuratingExamples/"

step = 5
no_model_selection = False

class ActiveLearnActionData(object):

    def __init__(self, X, y):
        self.X = np.digitize(X, bins = [1])
        self.y = y
        self.body = self.get_body()
        self.last_pos = 0
        self.last_neg = 0
        self.record = {"x": [], "pos": []}
        self.session = 0
        self.best_feature_next = list(range(len(self.X[0])))
        self.last_time_best_feature = (0, False, 0, 0)
        self.feature_coded_correct_dict = {}
        self.feature_coded_times_dict= {}





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
        poses = np.where(np.array(self.body['code']) == 1)[0]
        negs = np.where(np.array(self.body['code']) == 0)[0]
        validation_ids = list(poses) + list(negs)

        unlabeled = np.where(np.array(self.body['code']) == "undetermined")[0]
        # print("poses: ", poses)
        print("number of poses: ", len(poses), "/ ", end = "")
        print("total poses: ", Counter(self.y)[1], "/ ", end = "")
        print("total coded till now: ", len(validation_ids))

        # print("coded correctly: ", len(poses)-start_data)
        try:
            unlabeled_train = np.random.choice(unlabeled, size=len(poses))
            train_ids1 = list(poses) + list(negs) + list(unlabeled_train)
            code_array = np.array(self.body.code.to_list())
            code_array[unlabeled_train] = '0'
            assert bool(
                set(code_array[validation_ids]) & set(['undetermined'])) == False, "train set includes un-coded data!"

        except:
            train_ids1 = list(poses) + list(negs)
        if no_model_selection:
            best_model = svm_linear
        else:
            if len(poses)==1:
                best_model = bernoulli_nb
            else:
                model_f1 = get_model_f1(train_ids1, validation_ids, self.X, self.y, model_list)
                print_model(model_f1)
                itemMaxValue = max(model_f1.items(), key=lambda x: x[1])
                listOfKeys = list()
                # Iterate over all the items in dictionary to find keys with max value
                for key, value in model_f1.items():
                    if value == itemMaxValue[1]:
                        listOfKeys.append(key)
                best_model = np.random.choice(listOfKeys[:4], 1)[0]
            print("best model for this session is: ", best_model.name)
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

        prob = current_model.predict_proba(self.X[rest_data_ids])[:, pos_at]
        order = np.argsort(np.abs(prob))[::-1]  ## uncertainty sampling by distance to decision plane

        most_certain = order[:step]

        return np.array(rest_data_ids)[most_certain]





    def train_model_feature_selection(self):
        self.session += 1
        print("--------------train session ", self.session, "-----------------")
        poses = np.where(np.array(self.body['code']) == 1)[0]
        negs = np.where(np.array(self.body['code']) == 0)[0]
        validation_ids = list(poses) + list(negs)
        rest_data_ids = get_opposite(range(len(self.y)), validation_ids)
        unlabeled = np.where(np.array(self.body['code']) == "undetermined")[0]
        # print("poses: ", poses)
        print("number of poses: ", len(poses), "/ ", end = "")
        print("total poses: ", Counter(self.y)[1], "/ ", end = "")
        print("total coded till now: ", len(validation_ids))

        # print("coded correctly: ", len(poses)-start_data)
        try:
            unlabeled_train = np.random.choice(unlabeled, size=len(poses))
            train_ids1 = list(poses) + list(negs) + list(unlabeled_train)
            code_array = np.array(self.body.code.to_list())
            code_array[unlabeled_train] = '0'
            assert bool(
                set(code_array[validation_ids]) & set(['undetermined'])) == False, "train set includes un-coded data!"

        except:
            train_ids1 = list(poses) + list(negs)


        if len(poses)< 3 or len(negs) < 3:
            print("small sample, naive train")
            best_model = bernoulli_nb

            best_model.model.fit(self.X[train_ids1], code_array[train_ids1])
            try:
                pos_at = list(best_model.model.classes_).index('1')
            except:
                pos_at = list(best_model.model.classes_).index(1)

            prob = best_model.model.predict_proba(self.X[rest_data_ids])[:, pos_at]
            prob = sorted(prob, reverse= True)
            candidate_id_voi_dict = {}
            for candidate_id in tqdm(rest_data_ids):
                candidate_id_voi_dict[candidate_id] = 0
            print("candidate_id_voi_dict: ", sorted(candidate_id_voi_dict.items(), key=lambda x: x[1],reverse=True)[:5])
            determine_dict = candidate_id_voi_dict
            for i in range(len(rest_data_ids)):
                determine_dict[rest_data_ids[i]] += (prob[i])
            print("determine_dict: ", sorted(candidate_id_voi_dict.items(), key=lambda x: x[1],reverse=True)[:5])
            print("prob: ", sorted(prob, reverse=True))
            sorted_candidate = sorted(determine_dict.items(), key=operator.itemgetter(1), reverse=True)
            best_index = [ind[0] for ind in sorted_candidate][:step]
            print("best_index: ", best_index)
            return best_index



        else:
            print("bigger sample, complex train")
            model_f1 = get_model_f1(train_ids1, validation_ids, self.X[:, self.best_feature_next], self.y, model_list)
            # print_model(model_f1)
            itemMaxValue = max(model_f1.items(), key=lambda x: x[1])
            listOfKeys = list()
            # Iterate over all the items in dictionary to find keys with max value
            for key, value in model_f1.items():
                if value == itemMaxValue[1]:
                    listOfKeys.append(key)
            best_model = np.random.choice(listOfKeys[:4], 1)[0]
            print("best model for this session is: ", best_model.name)
        current_model = best_model

        feature_f1, feature_dict = get_feature_f1(train_ids1, validation_ids, self.X, self.y, current_model.model)
        print( sorted(feature_f1.items(), key=lambda x: x[1],reverse=True)[:5])
        print("---  additive-----")
        additive_dict = {}
        for key, value in feature_f1.items():
            if key in self.feature_coded_correct_dict.keys() and self.session > 20:
                # if self.feature_coded_correct_dict[key] - (len(poses)-3)/(len(validation_ids)-4) > 0.05 :
                additive_dict[key] = value +self.feature_coded_correct_dict[key] - (len(poses)-1)/(len(validation_ids)-1)
            else:
                additive_dict[key] = value
        print(sorted(additive_dict.items(), key=lambda x: x[1],reverse=True)[:5])
        sorted_feature_f1 = [key for key, item in sorted(additive_dict.items(), key=lambda x: x[1],reverse=True)]
        # print(sorted_feature_f1)
        code_array = np.array(self.body.code.to_list())
        assert bool(set(code_array[validation_ids]) & set(['undetermined'])) == False, "validation set includes un-coded data!"


        for best_feature in sorted_feature_f1:
            print('using best_feature: ', best_feature, "  to predict")
            selected_feature =  feature_dict[best_feature]
            # print("selected_feature")
            # current_model.fit(self.X[validation_ids], code_array[validation_ids])
            try:
                current_model.model.fit(self.X[validation_ids][:, selected_feature], code_array[validation_ids])
                rest_data_ids = get_opposite(range(len(self.y)), validation_ids)
            except:
                continue
            try:
                pos_at = list(current_model.model.classes_).index('1')
            except:
                pos_at = list(current_model.model.classes_).index(1)
            # try:
            prob = current_model.model.predict_proba(self.X[rest_data_ids][:, selected_feature])[:, pos_at]
            order = np.argsort(np.abs(prob))[::-1]  ## certainty sampling by distance to decision plane
            # most_certain = order[:step]
            self.best_feature_next = selected_feature
            # save_obj(poses, "poses", base_dir + "temp/experiment_feature/session" + str(self.session), "")
            # save_obj(negs, "negs", base_dir + "temp/experiment_feature/session" + str(self.session), "")
            # orig_dir = base_dir + "/cv/test_size0/fold0/code_state[[1, 0], [1, 1], [1, 2], [1, 3]]baseline/costopall"
            # patterns = load_obj("significant_patterns", orig_dir, "")
            # pattern_list = np.array([pattern for pattern in patterns]
            # pattern_save = pd.DataFrame(columns = ['patterns', 'idf_all', 'idf_pos', 'idf_neg'])
            # pattern_save['patterns'] = pattern_list[selected_feature]
            # save_obj(pattern_list[selected_feature], "selected_feature" + str(len(selected_feature)), base_dir + "temp/experiment_feature/session" + str(self.session), "")
            self.last_time_best_feature = best_feature
            # return np.array(rest_data_ids)[most_certain]

            prob = current_model.model.predict_proba(self.X[rest_data_ids][:, selected_feature])[:, pos_at]
            prob = sorted(prob, reverse= True)
            if prob[0] == 0:
                print('last feature could not get pos value, change feature')
                continue

            if len(rest_data_ids) == 1:
                return rest_data_ids[0]

            candidate_id_voi_dict = {}
            if len(negs) == 1:
                for candidate_id in tqdm(rest_data_ids):
                    candidate_id_voi_dict[candidate_id] = 0
            else:
                for candidate_id in tqdm(rest_data_ids):
                    code_array = np.array(self.body.code.to_list())
                    train_ids2 = list([candidate_id]) + list(validation_ids)
                    code_array[candidate_id] = 1
                    assert bool(set(code_array[validation_ids]) & set(['undetermined'])) == False, "train set includes un-coded data!"
                    voi = get_VOI(train_ids2, validation_ids, self.X[:, selected_feature], code_array,
                                current_model)
                    candidate_id_voi_dict[candidate_id] = voi
            print("candidate_id_voi_dict: ", sorted(candidate_id_voi_dict.items(), key=lambda x: x[1],reverse=True)[:5])
            determine_dict = candidate_id_voi_dict
            if not determine_dict:
                return -1
            for i in range(len(rest_data_ids)):
                determine_dict[rest_data_ids[i]] += (prob[i])
            print("determine_dict: ", sorted(candidate_id_voi_dict.items(), key=lambda x: x[1],reverse=True)[:5])
            # prob.sort(reversed = True)
            print("prob: ", sorted(prob, reverse=True))
            sorted_candidate = sorted(determine_dict.items(), key=operator.itemgetter(1), reverse = True)
            best_index = [ind[0] for ind in sorted_candidate][:step]
            print("best_index: ", best_index)
            return best_index


















    def best_train(self, label_name):
        self.session += 1
        print("--------------train session ", self.session, "-----------------")
        poses = np.where(np.array(self.body['code']) == 1)[0]
        negs = np.where(np.array(self.body['code']) == 0)[0]
        validation_ids = list(poses) + list(negs)

        unlabeled = np.where(np.array(self.body['code']) == "undetermined")[0]
        # print("poses: ", poses)
        print("number of poses: ", len(poses), "/ ", end = "")
        print("total poses: ", Counter(self.y)[1], "/ ", end = "")
        print("total coded till now: ", len(validation_ids))

        # print("coded correctly: ", len(poses)-start_data)
        try:
            unlabeled_train = np.random.choice(unlabeled, size=len(poses))
            train_ids1 = list(poses) + list(negs) + list(unlabeled_train)
            code_array = np.array(self.body.code.to_list())
            code_array[unlabeled_train] = '0'
            assert bool(
                set(code_array[validation_ids]) & set(['undetermined'])) == False, "train set includes un-coded data!"

        except:
            train_ids1 = list(poses) + list(negs)
        if label_name == "jump":
            best_model = bernoulli_nb

        if label_name == 'costopall':
            best_model = multi_nb

        if label_name == 'keymove':
            best_model = bernoulli_nb

        if label_name == 'moveanimate':
            best_model = multi_nb

        if label_name == 'cochangescore':
            best_model = bernoulli_nb


        if label_name == 'movetomouse':
            best_model = bernoulli_nb


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

        prob = current_model.predict_proba(self.X[rest_data_ids])[:, pos_at]
        order = np.argsort(np.abs(prob))[::-1]  ## uncertainty sampling by distance to decision plane

        most_certain = order[:step]

        return np.array(rest_data_ids)[most_certain]





    def selection_train_no_voi(self):
        self.session += 1
        print("--------------train session ", self.session, "-----------------")
        poses = np.where(np.array(self.body['code']) == 1)[0]
        negs = np.where(np.array(self.body['code']) == 0)[0]
        validation_ids = list(poses) + list(negs)
        rest_data_ids = get_opposite(range(len(self.y)), validation_ids)
        unlabeled = np.where(np.array(self.body['code']) == "undetermined")[0]
        # print("poses: ", poses)
        print("number of poses: ", len(poses), "/ ", end="")
        print("total poses: ", Counter(self.y)[1], "/ ", end="")
        print("total coded till now: ", len(validation_ids))

        # print("coded correctly: ", len(poses)-start_data)
        try:
            unlabeled_train = np.random.choice(unlabeled, size=len(poses))
            train_ids1 = list(poses) + list(negs) + list(unlabeled_train)
            code_array = np.array(self.body.code.to_list())
            code_array[unlabeled_train] = '0'
            assert bool(
                set(code_array[validation_ids]) & set(['undetermined'])) == False, "train set includes un-coded data!"

        except:
            train_ids1 = list(poses) + list(negs)

        if len(poses) < 3 or len(negs) < 3:
            print("small sample, naive train")
            best_model = bernoulli_nb

            best_model.model.fit(self.X[train_ids1], code_array[train_ids1])
            try:
                pos_at = list(best_model.model.classes_).index('1')
            except:
                pos_at = list(best_model.model.classes_).index(1)

            prob = best_model.model.predict_proba(self.X[rest_data_ids])[:, pos_at]
            prob = sorted(prob, reverse=True)
            order = np.argsort(np.abs(prob))[::-1]  ## certainty sampling by distance to decision plane
            most_certain = order[:step]
            return np.array(rest_data_ids)[most_certain]



        else:
            print("bigger sample, complex train")
            model_f1 = get_model_f1(train_ids1, validation_ids, self.X[:, self.best_feature_next], self.y, model_list)
            # print_model(model_f1)
            itemMaxValue = max(model_f1.items(), key=lambda x: x[1])
            listOfKeys = list()
            # Iterate over all the items in dictionary to find keys with max value
            for key, value in model_f1.items():
                if value == itemMaxValue[1]:
                    listOfKeys.append(key)
            best_model = np.random.choice(listOfKeys[:4], 1)[0]
            print("best model for this session is: ", best_model.name)
        current_model = best_model

        feature_f1, feature_dict = get_feature_f1(train_ids1, validation_ids, self.X, self.y, current_model.model)
        print(sorted(feature_f1.items(), key=lambda x: x[1], reverse=True)[:5])
        print("---  additive-----")
        additive_dict = {}
        for key, value in feature_f1.items():
            if key in self.feature_coded_correct_dict.keys() and self.session > 20:
                # if self.feature_coded_correct_dict[key] - (len(poses)-3)/(len(validation_ids)-4) > 0.05 :
                additive_dict[key] = value + self.feature_coded_correct_dict[key] - (len(poses) - 1) / (
                            len(validation_ids) - 1)
            else:
                additive_dict[key] = value
        print(sorted(additive_dict.items(), key=lambda x: x[1], reverse=True)[:5])
        sorted_feature_f1 = [key for key, item in sorted(additive_dict.items(), key=lambda x: x[1], reverse=True)]
        # print(sorted_feature_f1)
        code_array = np.array(self.body.code.to_list())
        assert bool(
            set(code_array[validation_ids]) & set(['undetermined'])) == False, "validation set includes un-coded data!"

        for best_feature in sorted_feature_f1:
            print('using best_feature: ', best_feature, "  to predict")
            selected_feature = feature_dict[best_feature]
            try:
                current_model.model.fit(self.X[validation_ids][:, selected_feature], code_array[validation_ids])
                rest_data_ids = get_opposite(range(len(self.y)), validation_ids)
            except:
                continue
            try:
                pos_at = list(current_model.model.classes_).index('1')
            except:
                pos_at = list(current_model.model.classes_).index(1)
            # try:
            prob = current_model.model.predict_proba(self.X[rest_data_ids][:, selected_feature])[:, pos_at]
            order = np.argsort(np.abs(prob))[::-1]  ## certainty sampling by distance to decision plane
            most_certain = order[:step]
            self.best_feature_next = selected_feature
            self.last_time_best_feature = best_feature
            return np.array(rest_data_ids)[most_certain]







    ## Get random ##
    def random(self, step):
        return np.random.choice(self.pool,size=np.min((step,len(self.pool))),replace=False)


    def start_as_1_pos(self):
        r = self.random(10)
        while True:
            for ele in r:
                if self.body.label[ele] == 1:
                    return [ele]
            r = self.random(step = 10)

    def start_as_3_pos(self):
        r = self.random(10)
        while True:
            ele_list = []
            for ele in r:
                if self.body.label[ele] == 1:
                    ele_list.append(ele)
                if len(ele_list) == 3:
                    return ele_list
            r = self.random(step = 10)



    def start_as_1_neg(self):
        r = self.random(10)
        while True:
            for ele in r:
                if self.body.label[ele] == 0:
                    return [ele]
            r = self.random(step= 10)

    ## Code candidate studies ##
    def code(self,id):
        id = np.array(id)
        print_data = []
        print("id: ", id)
        # try:
        for bla in np.nditer(id):
            print_data.append(self.body['label'][bla])
        # except:
        #     print_data.append(id)
        # print("bla: ", print_data)
        print("coded correct: ", Counter(print_data)[1], "among 10")
        self.body['code'][id] = self.body['label'][id]
        self.body["session"][id] = self.session
        if self.last_time_best_feature in self.feature_coded_correct_dict.keys():
            existing_number = self.feature_coded_correct_dict[self.last_time_best_feature]
            existing_times = self.feature_coded_times_dict[self.last_time_best_feature]

            precision1 = Counter(print_data)[1] / step
            precision2 = (existing_number * step * existing_times )/ (step * existing_times)
            if precision1 + precision2 == 0:
                weighted_precision = 0
            else:
                weighted_precision = (precision1 * 0.7 + precision2 * 0.3)

            self.feature_coded_correct_dict[self.last_time_best_feature] = weighted_precision
            self.feature_coded_times_dict[self.last_time_best_feature] += 1

        elif self.session >= 1:
            self.feature_coded_correct_dict[self.last_time_best_feature] = Counter(print_data)[1] / step
            self.feature_coded_times_dict[self.last_time_best_feature] = 1
        # if self.session >=1:
        #     self.feature_coded_correct_dict[self.last_time_best_feature] = Counter(print_data)[1] / step
        #     # self.feature_coded_times_dict[self.last_time_best_feature] = 1
        print("self.feature_coded_times_dict",
              sorted(self.feature_coded_times_dict.items(), key=lambda x: x[1], reverse=True))
        print("self.feature_coded_correct_dict",
              sorted(self.feature_coded_correct_dict.items(), key=lambda x: x[1], reverse=True))
