import sys
sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples/PaperSubmission/PassiveLearner')
from PatternMining import *
sys.setrecursionlimit(10**8)
from ActiveLearnActionData import *
import warnings
warnings.filterwarnings('ignore')

class InteractiveActiveLearnActionData(ActiveLearnActionData):
    def __init__(self, X, y, scratch_id_list):
        super().__init__(X, y)
        self.scratch_id_list = scratch_id_list


