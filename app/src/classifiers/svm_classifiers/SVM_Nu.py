import os
import pandas as pd
from sklearn.svm import NuSVC
from sklearn.model_selection import GridSearchCV
from classifiers.Model import Model

dir_name = os.path.dirname(__file__)
data_path = os.path.join(dir_name, '../../preprocessed_data/team_seasons_classified_1.csv')


def get_tuned_parameters():
    data = pd.read_csv(data_path)
    x = data.iloc[:, 3:-1]
    y = data.iloc[:, -1]
    param_grid = {"nu": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                  "gamma": ["scale", "auto"]}
    grid_search = GridSearchCV(NuSVC(), param_grid, cv=10)
    grid_search.fit(x, y)
    print("Best parameters for NuSVC:{}".format(grid_search.best_params_))


class SVMNuModel(Model):
    def __init__(self):
        self.name = "Nu-Support SVC"
        self.model = NuSVC(gamma="scale")


class TunedSVMNuModel(Model):
    def __init__(self):
        self.name = "Nu-Support SVC Tuned"
        # gamma = 1 / (n_features * X.var())
        self.model = NuSVC(gamma="scale", nu=0.5)


# get_tuned_parameters()
