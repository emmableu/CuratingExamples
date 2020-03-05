import pandas as pd
import os
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from classifiers.Model import Model

dir_name = os.path.dirname(__file__)
data_path = os.path.join(dir_name, '../../preprocessed_data/team_seasons_classified_1.csv')


def get_tuned_parameters():
    param_grid = {"learning_rate": [0.001, 0.01, 0.1, 0.2, 0.5, 1],
                  "n_estimators": [25, 50, 75, 100, 200, 300, 500, 1000]}
    grid_search = GridSearchCV(AdaBoostClassifier(), param_grid, cv=10)
    data = pd.read_csv(data_path)
    x = data.iloc[:, 3:-1]
    y = data.iloc[:, -1]
    grid_search.fit(x, y)
    print("Best parameters:{}".format(grid_search.best_params_))
    return grid_search.best_params_


class AdaBoostModel(Model):
    def __init__(self):
        self.name = "AdaBoost"
        self.model = AdaBoostClassifier()


class TunedAdaBoostModel(Model):
    def __init__(self):
        self.name = "TunedAdaBoost"
        self.model = AdaBoostClassifier(learning_rate=0.5, n_estimators=500)

# get_tuned_parameters()
