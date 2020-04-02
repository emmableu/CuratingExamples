from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_predict
import sys
sys.path.append("/Users/wwang33/Documents/IJAIED20/CuratingExamples/my_module")
from save_load_pickle import *
from preprocessors.PreProcessor import *
from preprocessors.DPMPreProcessor import *
from sklearn.model_selection import KFold, cross_val_score, LeaveOneOut

class Trainer:
    def __init__(self, X, y):
        self.comment = "Parent class trainer"
        self.X = X
        self.y = y

    def trainer_preprocess(self, pre_processor):
        self.X = pre_processor.preprocess(self.X, self.y)

    def train(self):
        pass

    def get_sample(self, sampler):
        pass




x_train, y_train, pattern_orig = get_data([[1, 0]], digit01= True, action_name = 'keymove', datafolder = "xy_0heldout")
trainer = Trainer(x_train, y_train)
pre_processer = DPMPreProcessor()
trainer.trainer_preprocess(pre_processer)
print(trainer.X)