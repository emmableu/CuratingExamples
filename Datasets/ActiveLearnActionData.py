from __future__ import print_function, division
from sklearn import linear_model

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
# model_list = [baseline, knn, lr, svm_c, svm_nu, svm_linear, dt, adaboost, bagging, rf, gaussian_nb,
#                     bernoulli_nb, multi_nb, complement_nb, mlp]

model_list = [bernoulli_nb, multi_nb, complement_nb, gaussian_nb, adaboost, svm_linear, lr]
root_dir = "/Users/wwang33/Documents/IJAIED20/CuratingExamples/"

step = 5
no_model_selection = False


class ActiveLearnActionData(object):

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.body = self.get_body()
        self.session = 0
        self.best_feature_next = list(range(len(self.X[0])))
        self.last_time_best_feature = (0, False, 0, 0)
        self.feature_coded_correct_dict = {}
        self.feature_coded_times_dict = {}
        self.step = 10
        self.get_numbers()
        self.last_pos = 0
        self.train_prevalence = 0
        self.uncertainty = True
        self.turn_point_arrival = False
        self.post_turn_point = False
        self.steady_growth = False

    def get_body(self):
        body = pd.DataFrame(columns=['X', 'label'])
        for i in range(len(self.y)):
            body.loc[i] = {"X": self.X[i], 'label': int(self.y[i])}
        n = len(self.y)
        body["code"] = [-1] * n
        body["session"] = [-1] * n
        return body

    def get_numbers(self):
        total = len(self.body["code"])
        pos = Counter(self.body["code"])[1]
        neg = Counter(self.body["code"])[0]
        self.poses = np.where(np.array(self.body['code']) == 1)[0]
        self.negs = np.where(np.array(self.body['code']) == 0)[0]
        code_array = np.array(self.body.code.to_list())
        self.code_array = code_array.astype(int)

        self.train_ids1 = list(self.poses) + list(self.negs)
        self.rest_data_ids = get_opposite(range(len(self.y)), self.train_ids1)

        self.pool = np.where(np.array(self.body['code']) == -1)[0]
        self.labeled = list(set(range(len(self.body['code']))) - set(self.pool))
        assert bool(
            set(self.code_array[self.train_ids1]) & set(['undetermined'])) == False, "train set includes un-coded data!"
        self.session_data = self.get_session_data()
        f = np.array(self.session_data, dtype=np.float)

        def moving_average(x, w):
            return np.convolve(x, np.ones(w), 'valid') / w

        if len(self.session_data)>10:
            self.session_data_ma = moving_average(f, 3)
            f = moving_average(f, 3)
            g = np.gradient(f)
            g = moving_average(g, 3)


            if not self.steady_growth and not self.turn_point_arrival: #deterimine whether this point is steady growth point
                g_order = np.argsort(abs(g))
                print(g)
                print(g_order)
                xsorted = np.argsort(g_order)
                ypos = np.searchsorted(g_order[xsorted], [len(g) - 1, len(g) - 2, len(g) - 3])
                indices2 = xsorted[ypos]
                if np.sum(indices2) < 8:
                    self.steady_growth = True
                    print("steady growth point arrival!")
                    print(self.session)

            if not self.turn_point_arrival: #determine whether this point is turn point
                g_order = np.argsort(g)
                print(g)
                print(g_order)

                xsorted = np.argsort(g_order)
                ypos = np.searchsorted(g_order[xsorted], [len(g) - 1, len(g) - 2, len(g) - 3])
                indices2 = xsorted[ypos]

                print("indices2", indices2)
                if 0 in indices2:
                    if g[len(g)-1] < -0.9:
                        self.turn_point_arrival = True
                        print("turn point arrival!")
                        print(self.session)


                    elif 1 in indices2 and g[len(g)-1]<-0.7:
                        self.turn_point_arrival = True
                        print("turn point arrival!")
                        print(self.session)


            print("session data and gradients")
            for f_i, f_d in enumerate(f):
                try:
                    print(f[f_i], g[f_i-1])
                except:
                    continue
            self.session_data_ma = moving_average(f, 3)
            # session_order = np.argsort(self.session_data_ma)
            # xsorted = np.argsort(session_order)
            # ypos = np.searchsorted(session_order[xsorted], [len(self.session_data_ma) - 1, len(self.session_data_ma) - 2, len(self.session_data_ma) - 3])
            # indices2 = xsorted[ypos]
            if self.turn_point_arrival:
                if self.session_data_ma[len(g) - 1] < 2:
                        self.post_turn_point = True
                        print("post_turn point")
                        print(self.session)
            else:
                if self.session_data_ma[len(g) - 1] < 1 and self.session_data_ma[len(g) - 2] < 1 and self.session_data_ma[len(g) - 3] < 1 and self.session_data_ma[len(g) - 4] < 1:
                        self.post_turn_point = True

                print("indices2", indices2)

        return pos, neg, total

    def get_model(self):
        model_f1 = submission_get_model_f1(self.train_ids1, self.X, self.y, model_list)
        print_model(model_f1)
        itemMaxValue = max(model_f1.items(), key=lambda x: x[1])
        listOfKeys = list()
        # Iterate over all the items in dictionary to find keys with max value
        for key, value in model_f1.items():
            if value == itemMaxValue[1]:
                listOfKeys.append(key)
        best_model = np.random.choice(listOfKeys[:4], 1)[0]
        print("best model for this session is: ", best_model.name)
        return best_model

    def active_uncertainty_train(self):
        self.session += 1
        self.get_numbers()
        print("--------------uncertainty train session ", self.session, "-----------------")
        best_model = svm_linear
        current_model = best_model
        input_x = np.insert(self.X, 0, 1, axis=1)
        current_model.model.fit(input_x[self.train_ids1], self.code_array[self.train_ids1])
        rest_data_ids = get_opposite(range(len(self.y)), self.train_ids1)
        if len(rest_data_ids) == 0:
            return current_model, []
        try:
            pos_at = list(current_model.model.classes_).index('1')
        except:
            pos_at = list(current_model.model.classes_).index(1)
        prob = current_model.model.predict_proba(input_x[rest_data_ids])[:, pos_at]
        order = np.argsort(np.abs(prob-0.5))[:self.step]    ## uncertainty sampling by prediction probability
        return current_model, np.array(rest_data_ids)[order]

    def passive_train(self, get_candidate=False):
        self.session += 1
        self.get_numbers()

        print("--------------vanila passive train session ", self.session, "-----------------")
        best_model = svm_linear
        current_model = best_model
        input_x = np.insert(self.X, 0, 1, axis=1)
        current_model.model.fit(input_x[self.train_ids1], self.code_array[self.train_ids1])
        if not get_candidate:
            return current_model
        else:
            rest_data_ids = get_opposite(range(len(self.y)), self.train_ids1)
            try:
                pos_at = list(current_model.model.classes_).index('1')
            except:
                pos_at = list(current_model.model.classes_).index(1)
            prob = current_model.model.predict_proba(input_x[rest_data_ids])[:, pos_at]
            order1 = np.argsort(np.abs(prob))[::-1]  ## uncertainty sampling by distance to decision plane

            most_certain = np.array(rest_data_ids)[order1[:self.step]]
            return best_model, most_certain

    def dpm_passive_train(self, jaccard):
        self.session += 1
        self.get_numbers()

        print("--------------train session ", self.session, "-----------------")
        best_model = svm_linear
        current_model = best_model
        selected_features = select_feature(self.X[self.train_ids1], self.code_array[self.train_ids1], jaccard)
        input_x = np.insert(self.X[:, selected_features], 0, 1, axis=1)
        # print("input x: ", input_x)
        current_model.model.fit(input_x[self.train_ids1], self.code_array[self.train_ids1])
        return current_model, selected_features

    def active_model_selection_train(self, all_uncertainty = False):
        self.session += 1
        self.get_numbers()
        print("--------------train session ", self.session, "-----------------")
        if len(self.poses) == 1:
            best_model = svm_linear
        else:
            best_model = self.get_model()
        current_model = best_model.model
        code_array = np.array(self.body.code.to_list())
        input_x = np.insert(self.X, 0, 1, axis=1)
        current_model.fit(input_x[self.train_ids1], code_array[self.train_ids1])
        rest_data_ids = get_opposite(range(len(self.y)), self.train_ids1)
        try:
            pos_at = list(current_model.classes_).index('1')
        except:
            pos_at = list(current_model.classes_).index(1)

        prob = current_model.predict_proba(input_x[rest_data_ids])[:, pos_at]
        order1 = np.argsort(np.abs(prob))[::-1]  ## uncertainty sampling by distance to decision plane

        most_certain = np.array(rest_data_ids)[order1[:self.step]]
        order2 = np.argsort(np.abs(prob-0.5))[:self.step]    ## uncertainty sampling by prediction probability
        most_uncertain =  np.array(rest_data_ids)[order2[:self.step]]
        if all_uncertainty:
            return best_model, most_uncertain
        if len(self.poses) <=10:
            self.uncertainty = True
            return best_model, most_uncertain
        else:
            self.uncertainty = False
            return best_model, most_certain








    def passive_model_selection_train(self):
        self.session += 1
        self.get_numbers()
        print("--------------train session ", self.session, "-----------------")
        best_model = self.get_model()
        code_array = np.array(self.body.code.to_list())
        input_x = np.insert(self.X, 0, 1, axis=1)
        best_model.model.fit(input_x[self.train_ids1], code_array[self.train_ids1])
        return best_model



    def estimate_curve_orig(self, clf_class, reuse=False, num_neg=0):
        clf = clf_class.model

        def prob_sample(probs):
            order = np.argsort(probs)[::-1]
            count = 0
            can = []
            sample = []
            for i, x in enumerate(probs[order]):
                count = count + x
                can.append(order[i])
                if count >= 1:
                    # sample.append(np.random.choice(can,1)[0])
                    sample.append(can[0])
                    count = 0
                    can = []
            return sample
        input_x = np.insert(self.X, 0, 1, axis=1)
        prob1 = clf.decision_function(input_x)
        prob = np.array([[x] for x in prob1])
        y = np.array([1 if x == 1 else 0 for x in self.body['code']])
        y0 = np.copy(y)

        if len(self.poses) and reuse:
            all = list(set(self.poses) | set(self.negs) | set(self.pool))
        else:
            all = range(len(self.y))
        pos_num_last = Counter(y0)[1]
        lifes = 1
        life = lifes
        while (True):
            # C = Counter(y0[all])[1]/ num_neg
            es = linear_model.LogisticRegression(penalty='l2', fit_intercept=True)
            es.fit(prob[all], y0[all])
            pos_at = list(es.classes_).index(1)
            pre = es.predict_proba(prob[self.pool])[:, pos_at]
            y = np.copy(y0)
            sample = prob_sample(pre)
            for x in self.pool[sample]:
                y[x] = 1
            pos_num = Counter(y)[1]
            if pos_num == pos_num_last:
                life = life - 1
                if life == 0:
                    break
            else:
                life = lifes
            pos_num_last = pos_num


        est_y = pos_num - self.last_pos
        pre = es.predict_proba(prob)[:, pos_at]

        return est_y, pre

    def estimate_curve(self, clf_class, reuse=False, num_neg=0):
        self.get_numbers()
        if (self.uncertainty):
            self.train_prevalence = len(self.poses) / len(self.labeled)
        est_y = len(self.y) * self.train_prevalence
        if self.steady_growth and not self.post_turn_point:
            clf = clf_class.model
            clf.fit(self.X[self.train_ids1], self.y[self.train_ids1])
            y_pred = clf.predict(self.X)
            # test_prevalence = (Counter(y_pred)[1])/len(self.y)
            est_y = (est_y + (Counter(y_pred)[1]))/2



        if self.post_turn_point:
            # est_y2 =len(self.poses)+self.session_data_ma[-1] * len(self.pool)/self.step
            est_y2 =len(self.poses)
            est_y = est_y2
        return est_y

    def get_session_data(self):
        session_data = []
        for i in range(self.session):
            body_data = self.body[self.body.session == i]
            pos_count = Counter(body_data['code'])[1]
            if i == 0:
                pos_count -= 1
            session_data.append(pos_count)
        return session_data



            # self.get_numbers()
        # clf = lr.model
        # train_prevalence = len(self.poses)/len(self.labeled)
        # input_x = np.insert(self.X, 0, 1, axis=1)
        # clf.fit(input_x[self.train_ids1], self.y[self.train_ids1])
        # rest_data_ids = get_opposite(range(len(self.y)), self.train_ids1)
        # # pos_at = list(clf.classes_).index(1)
        # y_pred =clf.predict(input_x[rest_data_ids])
        # print(y_pred)
        # # y_pred = clf.predict(input_x)
        # test_prevalence = (Counter(y_pred)[1])/len(rest_data_ids)
        # total_prevalence = ((Counter(y_pred)[1]) + len(self.poses))/(len(rest_data_ids) +len(self.labeled))
        # # prevalence = min(train_prevalence, total_prevalence)
        # est_y = len(self.y) *total_prevalence
        # return est_y

    def init_estimate_curve(self, clf_class, reuse=False, num_neg=0):
        self.get_numbers()
        clf = clf_class.model
        train_prevalence = len(self.poses)/len(self.labeled)
        est_y = len(self.y) *train_prevalence
        return est_y

    def train_model_feature_selection(self):
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
            candidate_id_voi_dict = {}
            for candidate_id in tqdm(rest_data_ids):
                candidate_id_voi_dict[candidate_id] = 0
            print("candidate_id_voi_dict: ",
                  sorted(candidate_id_voi_dict.items(), key=lambda x: x[1], reverse=True)[:5])
            determine_dict = candidate_id_voi_dict
            for i in range(len(rest_data_ids)):
                determine_dict[rest_data_ids[i]] += (prob[i])
            print("determine_dict: ", sorted(candidate_id_voi_dict.items(), key=lambda x: x[1], reverse=True)[:5])
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
            # print("selected_feature")
            # current_model.fit(self.X[validation_ids], code_array[validation_ids])
            try:
                current_model.model.fit(self.X[train_ids1][:, selected_feature], code_array[train_ids1])
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
            prob = sorted(prob, reverse=True)
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
                    assert bool(set(code_array[validation_ids]) & set(
                        ['undetermined'])) == False, "train set includes un-coded data!"
                    voi = get_VOI(train_ids2, validation_ids, self.X[:, selected_feature], code_array,
                                  current_model)
                    candidate_id_voi_dict[candidate_id] = voi
            print("candidate_id_voi_dict: ",
                  sorted(candidate_id_voi_dict.items(), key=lambda x: x[1], reverse=True)[:5])
            determine_dict = candidate_id_voi_dict
            if not determine_dict:
                return -1
            for i in range(len(rest_data_ids)):
                determine_dict[rest_data_ids[i]] += (prob[i])
            print("determine_dict: ", sorted(candidate_id_voi_dict.items(), key=lambda x: x[1], reverse=True)[:5])
            # prob.sort(reversed = True)
            print("prob: ", sorted(prob, reverse=True))
            sorted_candidate = sorted(determine_dict.items(), key=operator.itemgetter(1), reverse=True)
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
        assert bool(
            set(code_array[validation_ids]) & set(['undetermined'])) == False, "validation set includes un-coded data!"
        current_model.fit(self.X[train_ids1], code_array[train_ids1])
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
                # print("train_ids1: ", train_ids1)
                current_model.model.fit(self.X[train_ids1][:, selected_feature], code_array[train_ids1])
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
        # print(self.pool)
        return np.random.choice(self.pool, size=np.min((step, len(self.pool))), replace=False)

    def start_as_1_pos(self):
        r = self.random(10)
        while True:
            for ele in r:
                if self.body.label[ele] == 1:
                    return [ele]
            r = self.random(step=10)

    def start_as_3_pos(self):
        r = self.random(10)
        while True:
            ele_list = []
            for ele in r:
                if self.body.label[ele] == 1:
                    ele_list.append(ele)
                if len(ele_list) == 3:
                    return ele_list
            r = self.random(step=10)

    def start_as_1_neg(self):
        r = self.random(10)
        while True:
            for ele in r:
                if self.body.label[ele] == 0:
                    return [ele]
            r = self.random(step=10)

    def code(self, id):
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
            precision2 = (existing_number * step * existing_times) / (step * existing_times)
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

    def code_recall_curve(self, id):
        id = np.array(id)
        print_data = []
        print("id: ", id)
        # try:
        for bla in np.nditer(id):
            print_data.append(self.body['label'][bla])

        print("coded correct: ", Counter(print_data)[1], "among 10")
        self.body['code'][id] = self.body['label'][id]
        self.body["session"][id] = self.session
        if self.last_time_best_feature in self.feature_coded_correct_dict.keys():
            existing_number = self.feature_coded_correct_dict[self.last_time_best_feature]
            existing_times = self.feature_coded_times_dict[self.last_time_best_feature]

            precision1 = Counter(print_data)[1] / step
            precision2 = (existing_number * step * existing_times) / (step * existing_times)
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

        return Counter(print_data)[1]
