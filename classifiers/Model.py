import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_predict
import sys
sys.path.append("/home/wwang33/IJAIED20/CuratingExamples/my_module")
from save_load_pickle import *

class Model:
    def __init__(self):
        self.name = "Model"
        self.model = None
        self.confusion_matrix = None
        self.performance = None

    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.confusion_matrix = None
        self.performance = None

    def get_name(self):
        return self.name

    def get_model(self):
        return self.model

    def get_confusion_matrix(self):
        return self.confusion_matrix

    def get_performance(self):
        return self.performance

    def save_performance(self, save_dir, test_size):
        if is_obj( "results_ysize"+str(test_size), save_dir, ""):
            evaluation_metrics = load_obj("results_ysize"+str(test_size), save_dir, "")
            new_row = self.performance
            self.performance = new_row
            evaluation_metrics.loc[self.name] = new_row
            save_obj(evaluation_metrics,  "results_ysize"+str(test_size), save_dir, "")
        else:
            new_row = self.performance
            self.performance = new_row
            df = pd.DataFrame.from_dict({self.name: self.performance}, orient="index")
            save_obj(df, "results_ysize"+str(test_size), save_dir, "")

    def get_and_save_performance(self,X_train, X_test, y_train, y_test, save_dir, test_size):

            self.model.fit(X_train,y_train)
            y_pred = self.model.predict(X_test)
            self.confusion_matrix = confusion_matrix(y_test, y_pred)
            fpr, tpr, threshold = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)

            tn, fp, fn, tp = self.confusion_matrix.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            if tp == 0 and fp == 0 and fn == 0:
                precision = 1
                recall = 1
                f1 = 1
            elif tp == 0 and (fp > 0 or fn > 0):
                precision = 0
                recall = 0
                f1 = 0
            else:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                f1 = 2 * (precision) * (recall) / (precision + recall)
            perf = {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": accuracy, "precision": precision, "recall": recall,
                    "f1": f1, "auc": roc_auc}
            self.performance = perf
            self.save_performance( save_dir, test_size)
            return perf
