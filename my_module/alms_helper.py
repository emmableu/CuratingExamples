import numpy as np
import sklearn
import warnings

from classifiers.Baseline import BaselineModel
from classifiers.knn_classifiers.KNN import KNNModel
from classifiers.lr_classifiers.LogisticRegression import LRModel
import numpy as np
from sklearn.model_selection import LeavePOut, KFold, cross_val_predict, cross_validate, LeaveOneOut
import pandas as pd
import math
from sklearn.metrics import zero_one_loss, log_loss
from collections import Counter
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

baseline = BaselineModel()
knn = KNNModel()
lr = LRModel()



warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

baseline = BaselineModel()
knn = KNNModel()
lr = LRModel()



def get_VOI(train_ids, validation_ids, X, y, model_list):
    X_train_orig = X[train_ids]
    X_val_orig = X[validation_ids]
    y_train_orig  = y[train_ids]
    y_val_orig = y[validation_ids]
    v = len(y_val_orig)
    m = len(model_list)
    model_name_list = [model.name for model in model_list]
    iterables = [model_name_list, range(v), range(v)]
    s = pd.MultiIndex.from_product(iterables, names=['model', 'p_validation_index', 'l_validation_index'])

    for j in model_name_list:
        for i in range(v):
            s = s.drop((j, i, i))

    y_predict_proba_series = pd.Series([[-1, -1]] * m * v * (v - 1), index=s)
    y_pred_series = pd.Series([[-1, -1]] * m * v * (v - 1), index=s)

    l2o = LeavePOut(2)
    l2o.get_n_splits(X_val_orig)

    for train_index, val_index in l2o.split(X_val_orig):
        X_train, X_val = np.append(X_train_orig, X_val_orig[train_index], axis= 0), X_val_orig[val_index]
        y_train, y_val = np.append(y_train_orig, y_val_orig[train_index], axis= 0), y_val_orig[val_index]
        # print(X_train)
        # print(y_train)
        for mod in model_list:
            mod.model.fit(X_train, y_train)
            y_predict_proba = mod.model.predict_proba(X_val)
            y_pred = mod.model.predict(X_val)
            y_predict_proba_series[(mod.name, val_index[0], val_index[1])] = [y_predict_proba[0][0],
                                                                              y_predict_proba[0][1]]
            y_pred_series[(mod.name, val_index[0], val_index[1])] = y_pred[0]
            y_predict_proba_series[(mod.name, val_index[1], val_index[0])] = [y_predict_proba[1][0],
                                                                              y_predict_proba[1][1]]
            y_pred_series[(mod.name, val_index[1], val_index[0])] = y_pred[1]

    VOI = get_VOI_tau_m_hat(v, m, y_val_orig, y_pred_series, model_name_list)

    return VOI



# Removed baseline from models in case baseline result changes



# m*v*(v-1)次遍历

# print(y_predict_proba_series)

def get_error(j, i, y_val_orig, y_pred_series, v, model_name_list):
    # j: model name, e.g., "Baseline"
    # i: predicted_holdout_index
    compared = []
    for p_validation_index in range(v):
        if p_validation_index == i:
            continue
        compared.append(y_pred_series[(model_name_list[j], p_validation_index, i)])
    assert len(compared) == v-1, "length is not v-1!"
    # print("compared: ", compared)
    error = zero_one_loss([y_val_orig[i]]*(v-1), compared, normalize=True)
    # print("error: ", error)
    return error

def get_all_model_sum_exp_error(i,y_val_orig, y_pred_series, v, model_name_list):
    # j: model name, e.g., "Baseline"
    # i: predicted_holdout_index
    sum_exp_error = 0
    for j in model_name_list:
        compared = []
        for trained_holdout_index in range(v):
            if trained_holdout_index == i:
                continue
            compared.append(y_pred_series[(j, trained_holdout_index, i)])
        # print("compared: ", compared)
        error = zero_one_loss([y_val_orig[i]]*(v-1), compared, normalize=True)
        sum_exp_error += math.exp(error)
    return sum_exp_error


def get_P(v, m, y_val_orig, y_pred_series, model_name_list):
    '''
    P ∈ R(v×m) has Pij set to Pit (Mj) the probability of model Mj computed on only the left-out point (xi, yi) ∈ Vt,
    m = # models
    v = # validation data
    '''
    P = np.full((v,m), -1.0)
    for i in range(v):
        all_model_sum_exp_error = get_all_model_sum_exp_error(i,y_val_orig, y_pred_series, v, model_name_list)
        # print("all_model_sum_exp_error", all_model_sum_exp_error)
        for j in range(m):
            error = get_error(j, i,y_val_orig, y_pred_series, v, model_name_list)
            P[i,j] = 1 - math.exp(error)/all_model_sum_exp_error
    return P

def get_L(v, m,y_val_orig, y_pred_series, model_name_list):
    '''
    L ∈ R(m×v) has Lij set to the loss of model Mi on the other left-out point (xj , yj ) ∈ Vt.
    '''
    L = np.full((m,v), -1.0)
    for i in range(v):
        for j in range(m):
            error = get_error(j, i,y_val_orig, y_pred_series, v, model_name_list)
            L[j,i] = error
    return L


def get_s(v):
    '''
    s ∈ R(v) = (1/v, . . . , 1/v),
    v = # validation data
    '''
    return np.full((v, 1), 1/v)





def get_VOI_tau_m_hat(v, m, y_val_orig, y_pred_series, model_name_list):
    s = get_s(v)
    P = get_P(v, m, y_val_orig, y_pred_series, model_name_list)
    L = get_L(v, m, y_val_orig, y_pred_series, model_name_list)
    # print(P)
    # print("s@P", s.transpose()@P@L@s)
    # VOI_tau_m_hat = s.transpose() @P@L@s
    return s.transpose()@P@L@s




def get_model_f1(train_ids, validation_ids, X, y, model_list):
    X_train_orig = X[train_ids]
    X_val_orig = X[validation_ids]
    y_train_orig  = y[train_ids]
    y_val_orig = y[validation_ids]
    l1o = LeaveOneOut()
    l1o.get_n_splits(X_val_orig)
    actual_pos = Counter(y_val_orig)[1]
    recall_dict = {}
    precision_dict = {}
    f1_dict = {}
    for mod in model_list:
        tp = 0
        predict_pos = 0
        for train_index, val_index in l1o.split(X_val_orig):
            X_train, X_val = np.append(X_train_orig, X_val_orig[train_index], axis=0), X_val_orig[val_index]
            y_train, y_val = np.append(y_train_orig, y_val_orig[train_index], axis=0), y_val_orig[val_index]
            mod.model.fit(X_train, y_train)
            y_pred = mod.model.predict(X_val)
            if y_val == 1:
                predict_pos += 1
            if y_pred == y_val == 1:
                tp += 1
        recall = tp/actual_pos
        precision = tp/predict_pos
        recall_dict[mod] = recall
        precision_dict[mod] = precision
        if recall == precision == 0:
            f1 = 0
        else:
            f1 = 2*recall*precision/(recall + precision)
        f1_dict[mod] = f1



    return f1_dict


model_list = [baseline, knn, lr]
#
#
X_train_orig = np.arange(20).reshape(10,2)
y_train_orig = np.array([0,0,0,0,0,0,1,0,0,1])
X_val_orig = np.arange(20).reshape(10,2)
y_val_orig = np.array([1,1,1,1,1,1,1,1,1,1])
# voi = get_VOI(X_train_orig, X_val_orig, y_train_orig, y_val_orig, model_list)
#
#
# best_model = get_best_model(X_train_orig, y_train_orig, model_list)
# print(best_model)
# print(voi)







# from sklearn.model_selection import StratifiedKFold
# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4], [1, 2], [3, 4]])
# y = np.array([0, 0, 1, 1, 0, 0])
# skf = StratifiedKFold(n_splits=3)
# skf.get_n_splits(X, y)
#
# print(skf)
#
# for train_index, test_index in skf.split(X, y):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#





# class ALMS:
#     def __init__(self, X, y, model_list):
#         self.X = X
#         self.y = y
#         self.y_pred = [(-1)]*len(y)
#         self.pred_time = [-1]*len(y)
#         self.model_list = model_list


def recall_error(compared):
    return(Counter(compared)[1]/len(compared))
