import numpy as np
import sklearn
import warnings

from classifiers.Baseline import BaselineModel
from classifiers.knn_classifiers.KNN import KNNModel
from classifiers.lr_classifiers.LogisticRegression import LRModel
import numpy as np
from sklearn.model_selection import LeavePOut
import pandas as pd
import math
from sklearn.metrics import zero_one_loss, log_loss


warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

baseline = BaselineModel()
knn = KNNModel()
lr = LRModel()



def get_VOI(X_train, X_val, y_train, y_val, model_list):
    v = len(X_val)
    m = len(model_list)

    iterables = [M, range(v), range(v)]

    s = pd.MultiIndex.from_product(iterables, names=['model', 'p_validation_index', 'l_validation_index'])

    for j in M:
        for i in range(v):
            s = s.drop((j, i, i))

    y_predict_proba_series = pd.Series([[-1, -1]] * m * v * (v - 1), index=s)
    y_pred_series = pd.Series([[-1, -1]] * m * v * (v - 1), index=s)

    for train_index, val_index in l2o.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train = y[train_index]
        for mod in no_tuning_models:
            mod.model.fit(X_train, y_train)
            y_predict_proba = mod.model.predict_proba(X_val)
            y_pred = mod.model.predict(X_val)
            y_predict_proba_series[(mod.name, val_index[0], val_index[1])] = [y_predict_proba[0][0],
                                                                              y_predict_proba[0][1]]
            y_pred_series[(mod.name, val_index[0], val_index[1])] = y_pred[0]
            y_predict_proba_series[(mod.name, val_index[1], val_index[0])] = [y_predict_proba[1][0],
                                                                              y_predict_proba[1][1]]
            y_pred_series[(mod.name, val_index[1], val_index[0])] = y_pred[1]


class ALMS:
    def __init__(self, X, y, model_list):
        self.X = X
        self.y = y
        self.y_pred = [(-1)]*len(y)
        self.pred_time = [-1]*len(y)
        self.model_list = model_list
# Removed baseline from models in case baseline result changes

no_tuning_models = [baseline, knn, lr]


X = np.arange(20).reshape(10,2)
y = np.array([0,1,0,1,0,0,1,1,1,0])

v = 10
M = [model.name for model in no_tuning_models]
m = 3

l2o = LeavePOut(2)
l2o.get_n_splits(X)

# m*v*(v-1)次遍历

# print(y_predict_proba_series)

def get_error(j, i):
    # j: model name, e.g., "Baseline"
    # i: predicted_holdout_index
    compared = []
    for trained_holdout_index in range(v):
        if trained_holdout_index == i:
            continue
        compared.append(y_pred_series[(M[j], trained_holdout_index, i)])
    assert len(compared) == v-1, "length is not v-1!"
    print("compared: ", compared)
    # print([y[i]]*(v-1))
    # error = log_loss([y[i]]*(v-1), compared, labels = [0, 1],normalize=True)
    error = zero_one_loss([y[i]]*(v-1), compared, normalize=True)
    print("error: ", error)
    return error

def get_all_model_sum_exp_error(i):
    # j: model name, e.g., "Baseline"
    # i: predicted_holdout_index
    sum_exp_error = 0
    for j in M:
        compared = []
        for trained_holdout_index in range(v):
            if trained_holdout_index == i:
                continue
            compared.append(y_pred_series[(j, trained_holdout_index, i)])
        print("compared: ", compared)
        error = zero_one_loss([y[i]]*(v-1), compared, normalize=True)
        sum_exp_error += math.exp(error)
    return sum_exp_error

# print("y: ", y)
# get_error('Baseline', 3)

def get_P():
    '''
    P ∈ R(v×m) has Pij set to Pit (Mj) the probability of model Mj computed on only the left-out point (xi, yi) ∈ Vt,
    m = # models
    v = # validation data
    '''
    P = np.full((v,m), -1.0)
    for i in range(v):
        all_model_sum_exp_error = get_all_model_sum_exp_error(i)
        print("all_model_sum_exp_error", all_model_sum_exp_error)
        for j in range(m):
            error = get_error(j, i)
            P[i,j] = 1 - math.exp(error)/all_model_sum_exp_error
    return P

def get_L():
    '''
    L ∈ R(m×v) has Lij set to the loss of model Mi on the other left-out point (xj , yj ) ∈ Vt.
    '''
    L = np.full((m,v), -1.0)
    for i in range(v):
        for j in range(m):
            error = get_error(j, i)
            L[j,i] = error
    return L


def get_s():
    '''
    s ∈ R(v) = (1/v, . . . , 1/v),
    v = # validation data
    '''
    return np.full((v, 1), 1/v)





def get_VOI_tau_m_hat():
    s = get_s()
    P = get_P()
    L = get_L()
    print(P)
    print("s@P", s.transpose()@P@L@s)
    # VOI_tau_m_hat = s.transpose() @P@L@s
    return s.transpose()@P@L@s


VOI_tau_m_hat = get_VOI_tau_m_hat()

print(VOI_tau_m_hat)


