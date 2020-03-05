import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_predict


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

    def save_performance(self, filename):
        if os.path.isfile(filename):
            evaluation_metrics = pd.read_csv(filename, index_col=0)
            new_row = self.performance
            self.performance = new_row
            evaluation_metrics.loc[self.name] = new_row
            evaluation_metrics.to_csv(filename)
        else:
            new_row = self.performance
            self.performance = new_row
            df = pd.DataFrame.from_dict({self.name: self.performance}, orient="index")
            df.to_csv(filename)

    def draw_and_save_roc(self, fpr, tpr, roc_auc, figure_path):
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        figure_name = self.get_name().replace(" ", "_")
        path = figure_path+figure_name+"_ROC.png"
        plt.savefig(path)
        plt.clf()

    def get_and_save_performance(self, x, y, filename, figure_path):
        y_pred = cross_val_predict(self.model, x, y, cv=10)
        self.confusion_matrix = confusion_matrix(y, y_pred)
        fpr, tpr, threshold = roc_curve(y, y_pred)
        roc_auc = auc(fpr, tpr)

        def tn(y, y_pred): return confusion_matrix(y, y_pred)[0, 0]

        def fp(y, y_pred): return confusion_matrix(y, y_pred)[0, 1]

        def fn(y, y_pred): return confusion_matrix(y, y_pred)[1, 0]

        def tp(y, y_pred): return confusion_matrix(y, y_pred)[1, 1]

        tp, tn, fp, fn = int(tn(y, y_pred)), int(tp(y, y_pred)), int(fn(y, y_pred)), int(fp(y, y_pred))
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
        # precision = tp / (tp + fp)
        # recall = tp / (tp + fn)
        # f1 = 2 * (precision) * (recall) / (precision + recall)
        perf = {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": accuracy, "precision": precision, "recall": recall,
                "f1": f1, "auc": roc_auc}
        self.performance = perf
        self.save_performance(filename)
        self.draw_and_save_roc(fpr, tpr, roc_auc, figure_path)
        return perf
