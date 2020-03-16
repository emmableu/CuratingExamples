# import numpy as np
# import sklearn
# import warnings
#
# from classifiers.Baseline import BaselineModel
# from classifiers.knn_classifiers.KNN import KNNModel
# from classifiers.lr_classifiers.LogisticRegression import LRModel
# import numpy as np
# from sklearn.model_selection import LeavePOut
# import pandas as pd
#
# warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)
#
# baseline = BaselineModel()
# knn = KNNModel()
# lr = LRModel()
#
# # Removed baseline from models in case baseline result changes
# no_tuning_models = [baseline, knn, lr]
#
#
# X = np.arange(20).reshape(10,2)
# y = np.array([0,1,0,1,0,0,1,1,1,0])
#
# v = 10
# M = [model.name for model in no_tuning_models]
# m = 3
#
# knn.model.fit(X, y)
# y_prob = knn.model.predict_proba(X)
# from sklearn.metrics import log_loss
#
# i = 1
# error = y[i] - y_prob[i][y[i]]
#
#
# l2o = LeavePOut(2)
# l2o.get_n_splits(X)
#
# # m*v*(v-1)次遍历
#
#
# iterables = [M, range(v), range(v)]
#
# s = pd.MultiIndex.from_product(iterables, names=['model', 'trained_holdout_index', 'predicted_holdout_index'])
#
# for j in M:
#     for i in range(v):
#         s = s.drop((j, i, i))
#
# y_predict_proba_series = pd.Series([-1]*m*v*(v-1), index = s)
#
#
# for train_index, val_index_orig in l2o.split(X):
#     val_index_list = [val_index_orig, val_index_orig[::-1]]
#     print("val_index_list: ", val_index_list)
#     for val_index in val_index_list:
#         X_train, X_val = X[train_index] + X[val_index[0]], X[val_index[1]]
#         print("x_train: ", X_train)
#         y_train = y[train_index] + y[val_index[0]]
#         for j in no_tuning_models:
#             j.model.fit(X_train, y_train)
#             y_predict_proba = j.model.predict_proba([X_val])
#             y_predict_proba_series[(j.name, val_index[0], val_index[1])] = y_predict_proba
#
#
# print(y_predict_proba_series)
#
#
#
#
#
#
# #
# # def get_error(i, j):
# #
# #     error = log_loss([y_val]*(v-1), knn.model.predict_proba(X))
# #     return error
# #
# #
# #
# # def get_P(v, m):
# #     '''
# #     P ∈ R(v×m) has Pij set to Pit (Mj) the probability of model Mj computed on only the left-out point (xi, yi) ∈ Vt,
# #     m = # models
# #     v = # validation data
# #     '''
# #     P = np.zeros(v, m)
# #     for i in range(v):
# #
# #         for j in range(m):
# #             error = get_error(i)
# #             sum_exp_error_M = get_sum_exp_error_M(label,)
# #             P[i,j] = 1 - exp(error)/sum_exp_error_M
# #     return 0
# #
# # def get_L():
# #     '''
# #     L ∈ R(m×v) has Lij set to the loss of model Mi on the other left-out point (xj , yj ) ∈ Vt.
# #     '''
# #
# #
# # def get_s(v):
# #     '''
# #     s ∈ R(v) = (1/v, . . . , 1/v),
# #     v = # validation data
# #     '''
# #     return np.array([1/v]*v)
# #
# #
# #
# #
# #
# # def get_VOI_tau_m_hat():
# #     s = get_s(validation_set_size)
# #     P = get_P()
# #     L = get_L()
# #     VOI_tau_m_hat = s.transpose()*P*L*s
# #     return VOI_tau_m_hat
# #
# #
# # VOI_tau_m_hat