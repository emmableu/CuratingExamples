import os
import pandas as pd
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.svm import NuSVC
from sklearn.model_selection import GridSearchCV
from classifiers.Model import Model

dir_name = os.path.dirname(__file__)
data_path = os.path.join(dir_name, '../../preprocessed_data/team_seasons_classified_1.csv')


def get_tuned_parameters():
    data = pd.read_csv(data_path)
    x = data.iloc[:, 3:-1]
    y = data.iloc[:, -1]
    param_grid = {"C": [0.001, 0.01, 0.1, 1, 10, 20, 30, 40, 50, 100]}
    grid_search = GridSearchCV(LinearSVC(), param_grid, cv=10)
    grid_search.fit(x, y)
    print("Best parameters for LinearSVC:{}".format(grid_search.best_params_))


class SVMLinearModel(Model):
    def __init__(self):
        self.name = "Linear-Support SVC"
        self.model = LinearSVC()


class TunedSVMLinearModel(Model):
    def __init__(self):
        self.name = "Linear-Support SVC Tuned"
        self.model = LinearSVC(C=20)


# get_tuned_parameters()
