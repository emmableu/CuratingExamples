from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV
from classifiers.Model import Model

dir_name = os.path.dirname(__file__)
data_path = os.path.join(dir_name, '../../preprocessed_data/team_seasons_classified_1.csv')


def get_tuned_parameters():
    param_grid = {'activation': ["identity", "logistic", "tanh", "relu"],
                  'solver': ["lbfgs", "sgd", "adam"],
                  'learning_rate': ["constant", "invscaling", "adaptive"]}
    grid_search = GridSearchCV(MLPClassifier(), param_grid, cv=10)
    data = pd.read_csv(data_path)
    x = data.iloc[:, 3:-1]
    y = data.iloc[:, -1]
    grid_search.fit(x, y)
    print("Best parameters:{}".format(grid_search.best_params_))
    return grid_search.best_params_


class MLPModel(Model):
    def __init__(self):
        self.name = "MLP"
        self.model = MLPClassifier()


class TunedMLPModel(Model):
    def __init__(self):
        self.name = "MLPTuned"
        self.model = MLPClassifier(activation="relu", learning_rate="adaptive", solver="adam")


# get_tuned_parameters()
