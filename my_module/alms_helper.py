import numpy as np
import sklearn
import warnings
import sys
from tqdm import tqdm
from itertools import combinations
sys.path.append("/Users/wwang33/Documents/IJAIED20/CuratingExamples/my_module")
from save_load_pickle import *
sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets')
from Dataset import *
sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples')
from classifiers.Baseline import BaselineModel
from classifiers.knn_classifiers.KNN import KNNModel
from classifiers.lr_classifiers.LogisticRegression import LRModel
import numpy as np
from sklearn.model_selection import LeavePOut, StratifiedKFold, cross_val_predict, cross_validate, LeaveOneOut
import pandas as pd
import math
from sklearn.metrics import zero_one_loss, log_loss
from collections import Counter
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.feature_extraction.text import TfidfTransformer
baseline = BaselineModel()
knn = KNNModel()
lr = LRModel()



warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

baseline = BaselineModel()
knn = KNNModel()
lr = LRModel()



# feature_method_list = ['yes', 'diff', 'all']
yes_params = [3, 2, 1, 0, 4]
yes_exist_params = [False, True]
diff_params = [1, 0]
all_params = [3, 0, 4, 5, 6, 7]

feature_grids = [(x, y, z, g) for x in yes_params for y in yes_exist_params for z in diff_params for g in all_params]


def get_weights(x, y):
    y_yes_index = np.where(y == 1)[0]
    yes_x = x[y_yes_index]
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf_all = transformer.fit_transform(x).toarray()
    all_weights = transformer.idf_
    tfidf_yes = transformer.fit_transform(yes_x).toarray()
    yes_weights = transformer.idf_
    y_no_index = np.where(y == 0)[0]
    no_x = x[y_no_index]
    tfidf_no = transformer.fit_transform(no_x).toarray()
    no_weights = transformer.idf_

    return all_weights,yes_weights,no_weights






def select_feature(x, y, jaccard):
    x = np.digitize(x, bins=[1])

    y = y.astype(int)
    y_yes_index = np.where(y == 1)[0]
    yes_x = x[y_yes_index]
    # print('y_yes_index:', y_yes_index)
    y_no_index = np.where(y == 0)[0]
    no_x = x[y_no_index]
    feature_freq1 =  np.mean(yes_x, axis = 0)
    feature_freq2 =  np.mean(no_x, axis = 0)
    feature_sd1 = np.std(yes_x, axis=0)
    feature_sd2 = np.std(no_x, axis=0)
    # print(feature_freq1)
    n1 = len(yes_x)
    n2 = len(no_x)
    print("n1, n2", n1, n2)
    selected_patterns = []

    for i in tqdm(range(len(x[0]))):
        z,p = twoSampZ(feature_freq1[i], feature_freq2[i], feature_sd1[i], feature_sd2[i], n1, n2)
        # print(p)
        if p<0.01:
            selected_patterns.append(i)
    print("pattern selected with length:" ,len(selected_patterns))

    if not jaccard:
        return selected_patterns
    else:
        keep = jaccard_select(selected_patterns, x)
        print("jaccard similarity returns feature with length: ", len(keep))
        return keep



def jaccard_select(selected_patterns, x):
    # print("selected_patterns: ", selected_patterns)
    code_shape_p_q_list =  [[1, 0], [1, 1], [1, 2], [1, 3], [2, 3]]
    # code_shape_p_q_list =  [[1, 0]]
    #ATTENTION! Below IS by default from the full codestate, not the one hot!
    pattern_dir = base_dir + "/xy_0.3heldout/code_state" + str(code_shape_p_q_list)
    patterns = load_obj("full_patterns", pattern_dir, "")
    pattern_orig = np.array(patterns)
    print("start jaccard reduction")
    # p_pair_list = list(combinations(selected_patterns, 2))
    # for p_pair in tqdm(p_pair_list):
    reduced_list = [c1 for c1 in tqdm(selected_patterns) if jaccard_keep(c1, selected_patterns, pattern_orig, x)]
    print("jaccard selected a reduced list of length: ", len(reduced_list))
    return reduced_list

def jaccard_keep(c1, selected_patterns, pattern_orig, x):
    for c2 in (selected_patterns):
        if c2 == c1:
            continue
        c1_name, c2_name = pattern_orig[c1], pattern_orig[c2]
        s_c1, s_c2 = set(), set()
        for i in range(len(x)):
            if x[i, c1] == 1:
                s_c1.add(i)
            if x[i, c2] == 1:
                s_c2.add(i)
        union_set = s_c1.union(s_c2)
        joint_set = s_c1.intersection(s_c2)
        if len(union_set) == 0:
            continue
        j = len(joint_set)/len(union_set)
        if j >= 0.975*0.975:
            # print("j >= 0.95^2, c1, c2 name are: ", c1_name, c2_name)
            # print('union_set: ', union_set)
            # print('joint_set: ', joint_set)
            len_c1_name = len(c1_name.split("|"))
            len_c2_name = len(c2_name.split("|"))
            if len_c1_name <= len_c2_name:
                return False
    return True










def twoSampZ(X1, X2, sd1, sd2, n1, n2, mudiff=0):
    from numpy import sqrt, abs, round
    from scipy.stats import norm
    pooledSE = sqrt(sd1**2/n1 + sd2**2/n2)
    z = ((X1 - X2) - mudiff)/pooledSE
    pval = 2*(1 - norm.cdf(abs(z)))
    return round(z, 3), round(pval, 4)


def get_VOI(train_ids, validation_ids, X, y, cur_model):
    train_subset_index  = []

    for id in train_ids:
        if id not in validation_ids:
            train_subset_index.append(id)

    X_train_orig = X[train_subset_index]
    X_val_orig = X[validation_ids]
    y_train_orig  = y[train_subset_index]
    y_val_orig = y[validation_ids]

    actual_pos =  np.count_nonzero((y_val_orig) == str(1))


    split_strategy = StratifiedKFold(3)
    tp = 0
    predict_pos = 0

    # print("y_train_org: ", y_train_orig)
    # print('y_val_org: ', y_val_orig)
    for train_index, val_index in split_strategy.split(X_val_orig, y_val_orig):
        X_train, X_val = np.append(X_train_orig, X_val_orig[train_index], axis=0), X_val_orig[val_index]
        y_train, y_val = np.append(y_train_orig, y_val_orig[train_index], axis=0), y_val_orig[val_index]
        cur_model.model.fit(X_train, y_train)
        y_pred = cur_model.model.predict(X_val)
        # print("y_val: ", y_val, "y_train: ", y_train)
        # print("y_pred: ", y_pred)
        for index in range(len(y_pred)):
            if y_pred[index] == str(1):
                predict_pos += 1
                if y_val[index] == str(1):
                    tp += 1
    recall = tp / actual_pos
    # print("in model: ","tp: ", tp, "actual_pos: ", actual_pos, "predict_pos: ", predict_pos)
    if predict_pos == 0:
        precision = 0
    else:
        precision = tp / predict_pos
    if recall == precision == 0:
        f1 = 0
    else:
        f1 = 2 * recall * precision / (recall + precision)
    return f1



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

    train_subset_index  = []
    for id in train_ids:
        if id not in validation_ids:
            train_subset_index.append(id)

    X_train_orig = X[train_subset_index]
    X_val_orig = X[validation_ids]
    y_train_orig  = y[train_subset_index]
    y_val_orig = y[validation_ids]


    actual_pos = Counter(y_val_orig)[1]
    recall_dict = {}
    precision_dict = {}
    f1_dict = {}

    if len(validation_ids) < 10:
        split_strategy = LeaveOneOut()
        for mod in model_list:
            tp = 0
            predict_pos = 0
            for train_index, val_index in split_strategy.split(X_val_orig):
                X_train, X_val = np.append(X_train_orig, X_val_orig[train_index], axis=0), X_val_orig[val_index]
                y_train, y_val = np.append(y_train_orig, y_val_orig[train_index], axis=0), y_val_orig[val_index]
                mod.model.fit(X_train, y_train)
                y_pred = mod.model.predict(X_val)
                if y_pred == 1:
                    predict_pos += 1
                    if y_val == 1:
                        tp += 1
            recall = tp / actual_pos
            if predict_pos == 0:
                precision = 0
            else:
                precision = tp / predict_pos
            recall_dict[mod] = recall
            precision_dict[mod] = precision
            if recall == precision == 0:
                f1 = 0
            else:
                f1 = 2 * recall * precision / (recall + precision)
            f1_dict[mod] = f1
    else:
        split_strategy = StratifiedKFold(n_splits=5)
        for mod in model_list:
            tp = 0
            predict_pos = 0
            # print("y_train_org: ", y_train_orig)
            # print('y_val_org: ', y_val_orig)
            for train_index, val_index in split_strategy.split(X_val_orig, y_val_orig):
                X_train, X_val = np.append(X_train_orig, X_val_orig[train_index], axis=0), X_val_orig[val_index]
                y_train, y_val = np.append(y_train_orig, y_val_orig[train_index], axis=0), y_val_orig[val_index]
                mod.model.fit(X_train, y_train)
                y_pred = mod.model.predict(X_val)
                # print("y_val: ", y_val, "y_train: ", y_train)
                # print("y_pred: ", y_pred)
                for index in range(len(y_pred)):
                    if y_pred[index] == 1:
                        predict_pos += 1
                        if y_val[index] == 1:
                            tp += 1
            recall = tp / actual_pos
            # print("in model: ","tp: ", tp, "actual_pos: ", actual_pos, "predict_pos: ", predict_pos)
            if predict_pos == 0:
                precision = 0
            else:
                precision = tp / predict_pos
            recall_dict[mod] = recall
            precision_dict[mod] = precision
            if recall == precision == 0:
                f1 = 0
            else:
                f1 = 2 * recall * precision / (recall + precision)
            f1_dict[mod] = f1


    return f1_dict

def submission_get_model_f1(train_ids, X, y, model_list):
    X = X[train_ids]
    y = y[train_ids]
    f1_dict = {}

    for mod in model_list:
        return_dict = mod.naive_cross_val_predict(X, y, cv = 3)
        f1_dict[mod] = return_dict['f1']


    return f1_dict




def get_feature_list_dict(all_weights,yes_weights,no_weights):
    feature_dict = {}
    # np.random.shuffle(feature_grids)
    for feature_grid in feature_grids:
        feature_left = []
        x, y, z, g = feature_grid[0], feature_grid[1], feature_grid[2], feature_grid[3]
        if y:
            for i in range(len(yes_weights)):
                if yes_weights[i] > x and yes_weights[i] < 100 and abs(yes_weights[i] - no_weights[i]) > z and \
                        all_weights[i] > g:
                    feature_left.append(i)
        elif not y:
            for i in range(len(yes_weights)):
                if yes_weights[i] > x and abs(yes_weights[i] - no_weights[i]) > z and all_weights[i] > g:
                    feature_left.append(i)
        feature_dict[feature_grid] =feature_left
    return feature_dict

def get_feature_f1(train_ids, validation_ids, X, y, model):
    train_subset_index  = []
    for id in train_ids:
        if id not in validation_ids:
            train_subset_index.append(id)

    X_train_orig = X[train_subset_index]
    X_val_orig = X[validation_ids]
    y_train_orig  = y[train_subset_index]
    y_val_orig = y[validation_ids]
    all_weights, yes_weights, no_weights = get_weights(X_val_orig, y_val_orig)
    feature_dict = get_feature_list_dict(all_weights, yes_weights, no_weights)
    actual_pos = Counter(y_val_orig)[1]
    # recall_dict = {}
    # precision_dict = {}
    f1_dict = {}
    mod = model
    np.random.shuffle(feature_grids)
    for feature_grid in feature_grids:
        feature_left = feature_dict[feature_grid]
        X_train_trans = X_train_orig[:, feature_left]
        X_val_trans = X_val_orig[:, feature_left]

        exist_index1 = np.where(X_train_trans == 1)[0]
        exist_index2 = np.where(X_val_trans == 1)[0]

        if len(exist_index1) ==0 or len(exist_index2) == 0:
            f1_dict[feature_grid] = -1
            continue


        if len(validation_ids) < 5:
            split_strategy = LeaveOneOut()
            tp = 0
            predict_pos = 0
            for train_index, val_index in split_strategy.split(X_val_trans):
                X_train, X_val = np.append(X_train_trans, X_val_trans[train_index], axis=0), X_val_trans[val_index]
                y_train, y_val = np.append(y_train_orig, y_val_orig[train_index], axis=0), y_val_orig[val_index]
                mod.fit(X_train, y_train)
                y_pred = mod.predict(X_val)
                if y_pred == 1:
                    predict_pos += 1
                    if y_val ==  1:
                        tp += 1
            recall = tp / actual_pos
            if predict_pos == 0:
                precision = 0
            else:
                precision = tp / predict_pos

            # recall_dict[feature_grid] = recall
            # precision_dict[feature_grid] = precision
            if recall == precision == 0:
                f1 = 0
            else:
                f1 = 2 * recall * precision / (recall + precision)
            f1_dict[feature_grid] = f1
        else:
            split_strategy = StratifiedKFold(n_splits=3)

            tp = 0
            predict_pos = 0
            for train_index, val_index in split_strategy.split(X_val_trans, y_val_orig):
                X_train, X_val = np.append(X_train_trans, X_val_trans[train_index], axis=0), X_val_trans[val_index]
                y_train, y_val = np.append(y_train_orig, y_val_orig[train_index], axis=0), y_val_orig[val_index]
                mod.fit(X_train, y_train)
                y_pred = mod.predict(X_val)
                for index in range(len(y_pred)):
                    if y_pred[index] == 1:
                        predict_pos += 1
                        if y_val[index] == 1:
                            tp += 1
            recall = tp / actual_pos
            # print("tp: ", tp, "actual_pos: ", actual_pos, "predict_pos: ", predict_pos)




            if predict_pos == 0:
                precision = 0
            else:
                precision = tp / predict_pos

            # recall_dict[feature_method] = recall
            # precision_dict[feature_method] = precision
            if recall == precision == 0:
                f1 = 0
            else:
                f1 = 2 * recall * precision / (recall + precision)
            f1_dict[feature_grid] = f1


    return f1_dict, feature_dict


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
