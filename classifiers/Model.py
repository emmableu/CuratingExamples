import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_predict
import sys
sys.path.append("/Users/wwang33/Documents/IJAIED20/CuratingExamples/my_module")
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
        if is_obj("results_ysize"+str(test_size), dataset.root_dir + "Datasets/data",
                        "game_labels_" + str(dataset.total) + "/performance" + str(dataset.code_shape_p_q_list)):
            evaluation_metrics = pd.read_csv(filename, index_col=0)
            new_row = self.performance
            self.performance = new_row
            evaluation_metrics.loc[self.name] = new_row
            save_obj(evaluation_metrics,  "results_ysize"+str(test_size), save_dir, "")
        else:
            new_row = self.performance
            self.performance = new_row
            df = pd.DataFrame.from_dict({self.name: self.performance}, orient="index")
            save_obj(df, "results_ysize"+str(test_size), save_dir, "")

    def get_and_save_performance(self,x, y, save_dir, test_size):
            X_train, X_test, y_train, y_test = train_test_split(
             x, y, test_size= test_size, random_state=0)
            r = 0
            while len(set(y_train)) == 1:
                r += 1
                X_train, X_test, y_train, y_test = train_test_split(
                    x, y, test_size=test_size, random_state=r)
            self.model.fit(X_train,y_train)
            y_pred = self.model.predict(X_test)
            self.confusion_matrix = confusion_matrix(y_test, y_pred)
            fpr, tpr, threshold = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)

            def tn(y_test, y_pred): return confusion_matrix(y_test, y_pred)[0, 0]

            def fp(y_test, y_pred): return confusion_matrix(y_test, y_pred)[0, 1]

            def fn(y_test, y_pred): return confusion_matrix(y_test, y_pred)[1, 0]

            def tp(y_test, y_pred): return confusion_matrix(y_test, y_pred)[1, 1]

            tp, tn, fp, fn = int(tn(y_test, y_pred)), int(tp(y_test, y_pred)), int(fn(y_test, y_pred)), int(fp(y_test, y_pred))
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
