from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV
from classifiers.Model import Model

dir_name = os.path.dirname(__file__)
data_path = os.path.join(dir_name, '../../preprocessed_data/team_seasons_classified_1.csv')


def get_tuned_parameters():
    param_grid = {'bootstrap': [True, False],
                  'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
                  'max_features': ['auto', 'sqrt'],
                  'min_samples_leaf': [1, 2, 4],
                  'min_samples_split': [2, 5, 10]}
                  # 'n_estimators': [10, 25, 50, 75, 100, 200, 300, 500, 1000]}
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=10)
    data = pd.read_csv(data_path)
    x = data.iloc[:, 3:-1]
    y = data.iloc[:, -1]
    grid_search.fit(x, y)
    print("Best parameters:{}".format(grid_search.best_params_))
    return grid_search.best_params_


class RandomForestModel(Model):
    def __init__(self):
        self.name = "RandomForest"
        self.model = RandomForestClassifier(n_estimators=100)


class TunedRandomForestModel(Model):
    def __init__(self):
        self.name = "RandomForestTuned"
        self.model = RandomForestClassifier(bootstrap=True, max_depth=30, max_features="auto", min_samples_leaf=2, min_samples_split=10)


# get_tuned_parameters()
