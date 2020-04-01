import os
import pandas as pd
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from classifiers.Model import Model


dir_name = os.path.dirname(__file__)
data_path = os.path.join(dir_name, '../../preprocessed_data/team_seasons_classified_1.csv')


def get_tuned_parameters():
    param_grid = {"criterion": ["gini", "entropy"]}
    grid_search = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv=10)
    data = pd.read_csv(data_path)
    x = data.iloc[:, 3:-1]
    y = data.iloc[:, -1]
    grid_search.fit(x, y)
    print("Best parameters:{}".format(grid_search.best_params_))
    return grid_search.best_params_


class DecisionTreeModel(Model):
    def __init__(self):
        self.name = "DT"
        self.model = tree.DecisionTreeClassifier()


class TunedDecisionTreeModel(Model):
    def __init__(self):
        self.name = "DTTuned"
        self.model = tree.DecisionTreeClassifier(criterion="entropy")
