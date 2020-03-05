import pandas as pd
import warnings
import sklearn
from classifiers.Baseline import BaselineModel
from classifiers.knn_classifiers.KNN import TunedKNNModel
from classifiers.lr_classifiers.LogisticRegression import TunedLRModel
from classifiers.svm_classifiers.SVM_C import TunedSVMCModel
from classifiers.svm_classifiers.SVM_Nu import TunedSVMNuModel
from classifiers.svm_classifiers.SVM_Linear import TunedSVMLinearModel
from classifiers.decision_tree_classifiers.DecisionTree import TunedDecisionTreeModel
from classifiers.emsemble_classifiers.AdaBoost import TunedAdaBoostModel
from classifiers.emsemble_classifiers.RandomForest import TunedRandomForestModel
from classifiers.neural_network_classifiers.MLPModel import TunedMLPModel

warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)

baseline = BaselineModel()
knn = TunedKNNModel()
lr = TunedLRModel()
svm_c = TunedSVMCModel()
svm_nu = TunedSVMNuModel()
svm_linear = TunedSVMLinearModel()
dt = TunedDecisionTreeModel()
adaboost = TunedAdaBoostModel()
rf = TunedRandomForestModel()
mlp = TunedMLPModel()

tuning_models = [baseline, knn, lr, svm_c, svm_nu, svm_linear, dt, adaboost, rf, mlp]
# adaboost, bagging, rf, gaussian_nb, bernoulli_nb,
# multi_nb, complement_nb, mlp]


def get_xy_from_csv(filename):
    data = pd.read_csv(filename)
    x = data.iloc[:, 3:-1]
    y = data.iloc[:, -1]
    return x, y


def main():
    print("1. Dropping columns with missing data (1976-2004)")
    x1, y1 = get_xy_from_csv("preprocessed_data/team_seasons_classified_1.csv")
    for model in tuning_models:
        model.get_and_save_performance(x1, y1, "results/model_evaluation_1.csv", "results/figures1/")
        print("--------------" + model.get_name() + "--------------")
        print(model.get_confusion_matrix())
        print(model.get_performance())

    print("\n2. Dropping rows with missing data (1976-2004)")
    x2, y2 = get_xy_from_csv("preprocessed_data/team_seasons_classified_2.csv")
    for model in tuning_models:
        model.get_and_save_performance(x2, y2, "results/model_evaluation_2.csv", "results/figures2/")
        print("--------------" + model.get_name() + "--------------")
        print(model.get_confusion_matrix())
        print(model.get_performance())

    print("\n3. Replace missing data with mean values(1976-2004)")
    x3, y3 = get_xy_from_csv("preprocessed_data/team_seasons_classified_3.csv")
    for model in tuning_models:
        model.get_and_save_performance(x3, y3, "results/model_evaluation_3.csv", "results/figures3/")
        print("--------------" + model.get_name() + "--------------")
        print(model.get_confusion_matrix())
        print(model.get_performance())

    print("\n4. Replace missing data with median values(1976-2004)")
    x4, y4 = get_xy_from_csv("preprocessed_data/team_seasons_classified_4.csv")
    for model in tuning_models:
        model.get_and_save_performance(x4, y4, "results/model_evaluation_4.csv", "results/figures4/")
        print("--------------" + model.get_name() + "--------------")
        print(model.get_confusion_matrix())
        print(model.get_performance())


main()
