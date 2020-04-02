import sys
sys.path.append("/Users/wwang33/Documents/IJAIED20/CuratingExamples/my_module")
from save_load_pickle import *

class PreProcessor():
    def __init__(self):
        self.comment = "pre-process data, the default case is not preprocess"

    def preprocess(self, X, y):
        return X


