from __future__ import print_function, division
import sys, os
root = os.getcwd().split("src")[0] + "src/src/util"
sys.path.append(root)
from ActiveLearnActionData import *
import pickle
import os
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


def simulate_5_times_to_get_all(action_name,total, thres = 0, model = bernoulli_nb):
    # data_path='game_labels_'+ str(total) + '/' + label_name + '.csv'
    # target_recall = 0.7
    all_repetitions = 1
    all_simulation = []
    step = 10
    code_shape_p_q_list = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0]]
    allow_gap = True
    dataset = Dataset(total = 46, code_shape_p_q_list = code_shape_p_q_list, allow_gap = allow_gap)
    code_state = load_obj("code_state" + str(dataset.code_shape_p_q_list), dataset.root_dir + "Datasets/data",
                          "game_labels_" + str(415))


    for i in range(all_repetitions):
        read = ActiveLearnActionData(code_state, dataset.data, action_name)
        if thres == -1:
            real_thres = read.est_num//2
        else:
            real_thres = thres
        all_simulation.append(read)
        count = 0
        for j in range(total//step + 1):
            pos, neg, total_real = read.get_numbers()
            if total_real != total:
                print("wrong! total_real != total")
                break
            if pos + neg < total:
                count += 1
            if pos < 1:
                for id in read.start_as_1_pos():
                    read.code(id, read.body["label"][id])
            else:
                print("body: ", read.body)
                if j == total//step:
                    uncertain, uncertain_proba, certain, certain_proba = read.no_pole_train(total%step, model)
                else:
                    uncertain, uncertain_proba, certain, certain_proba_ = read.no_pole_train(step, model)

                if pos <= real_thres:
                    for id in uncertain:
                        read.code(id, read.body["label"][id])
                else:
                    for id in certain:
                        read.code(id, read.body["label"][id])
        read.count = count
        save_pickle(all_simulation, '/all_simulation_' + label_name,
                    "/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets/data/game_labels_" + str(415),
                    'simulation_' + str(thres) + "_" + "all/no_pole")





if __name__ == "__main__":
    total = 46
    thres = 0

    label_name_s = ['keymove', 'jump', 'costopall', 'wrap', 'cochangescore', 'movetomouse','moveanimate']
    # label_name_s = ['cochangescore']
    for label_name in label_name_s:
        # label_name = "cochangescore"
        model = svm_c.model
        simulate_5_times_to_get_all(label_name,total, thres, model)
        # simulate_10_times_using_weighted_train(label_name,total, thres)



    # all_simulation_wrap = load_obj('all_simulation_'+behavior,'/Users/wwang33/Documents/IJAIED20/src/workspace/data/game_labels_'+str(total), 'simulation_'+ str(thres) +"_"+ str(target_recall))
    # count_s = []
    # for simulation in all_simulation_wrap:
    #     count_s.append(simulation.count)
    # median_index = np.argsort(count_s)[len(count_s) // 2]
    # all_simulation_wrap[median_index].plot()
    # print(all_simulation_wrap[median_index].count)




