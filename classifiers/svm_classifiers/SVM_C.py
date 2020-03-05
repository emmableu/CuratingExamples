import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from classifiers.Model import Model

dir_name = os.path.dirname(__file__)
data_path = os.path.join(dir_name, '../../preprocessed_data/team_seasons_classified_1.csv')

C = [0.001, 0.01, 0.1, 0.2, 0.3, 1, 5, 10, 20, 100, 200, 1000]
degree = [1, 2, 3, 4, 5]
coef0 = [0.0001, 0.001, 0.002, 0.01, 0.02, 0.1, 0.2, 0.3, 1, 2, 5, 10]
gamma = [0.0001,0.001, 0.002, 0.01, 0.02, 0.03, 0.1, 0.2, 1, 2, 3, 5, 10, 100, 1000]
kernels = ["linear", "poly", "rbf", "sigmoid"]


def get_tuned_parameters():
    data = pd.read_csv(data_path)
    x = data.iloc[:, 3:-1]
    y = data.iloc[:, -1]
    param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 100]}
    grid_search = GridSearchCV(SVC(), param_grid, cv=10)
    grid_search.fit(x, y)
    print("Best parameters for rbf kernel:{}".format(grid_search.best_params_))


class SVMCModel(Model):
    def __init__(self):
        self.name = "C-Support SVC"
        self.model = SVC(gamma="scale")


class TunedSVMCModel(Model):
    def __init__(self):
        self.name = "C-Support SVC Tuned"
        self.model = SVC(gamma=0.001, C=0.001)


# get_tuned_parameters()
