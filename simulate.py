from __future__ import print_function, division
import sys, os
root = os.getcwd().split("src")[0] + "src/src/util"
sys.path.append(root)
sys.path.append("/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets")

from ActiveLearnActionData import *
import warnings
warnings.filterwarnings('ignore')
sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets')
from Dataset import *

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
no_tuning_models = [baseline, knn, lr, svm_c, svm_linear, dt, adaboost, bagging, rf, gaussian_nb,
                    bernoulli_nb, multi_nb, complement_nb, mlp]

label_name = "move_to_mouse"
#
# X = np.arange(0, 500).reshape(250, 2)
#
# y = np.random.randint(2, size=250)
# # array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0])
#
start_data = 3
# # the amount of positive data to start with
# # the total amount of data to start with is start_data +1, because it also starts with a negative data
#
X = load_obj("X_train",base_dir + "/cv/test_size0/fold0/code_state[[1, 0]]baseline/keymove/","")
y = load_obj("y_train",base_dir + "/cv/test_size0/fold0/code_state[[1, 0]]baseline/keymove/","")
# #
# X = X[:50]
# y = y[:50]
# #

total_data = len(y)


def simulate(X,y,label_name):
    all_simulation = {}
    all_simulation["y"] = y
    for i in range(1):
        read = ActiveLearnActionData(X, y)
        total = total_data
        for j in (range((total-start_data)//step)):
            pos, neg, total_real = read.get_numbers()
            print(pos, neg, total_real)
            real_true = Counter(y)[1]
            # print("actual positive before training: ", real_true)
            # if pos >= real_true:
            #     break
            if pos <= 1:
                if start_data == 3:
                    for id in read.start_as_3_pos():
                        read.code(id)
                for id in read.start_as_1_neg():
                    read.code(id)
            else:
                candidate = read.train()
                read.code(candidate)
        all_simulation[i] = (read.body['session'])
    save_pickle(all_simulation,'all_simulation_' + label_name, base_dir, "simulation")


simulate(X, y, 'keymove')