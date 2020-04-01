import sys
sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples/PaperSubmission/PassiveLearner')
from PatternMining import *
sys.setrecursionlimit(10**8)
from ActiveLearnActionData import *
import warnings
warnings.filterwarnings('ignore')

class FrontEndActiveLearnActionData:
    def __init__(self, behavior_list):
        self.behavior_list = behavior_list
        self.action_data_list = []
        for label_name in behavior_list:
            code_shape_p_q_list = [[1, 0]]
            # label_name = 'keymove'
            x_train, y_train, pattern_orig = get_data(code_shape_p_q_list, digit01=True, action_name=label_name,
                                                      datafolder="xy_0heldout")
            action_data = ActiveLearnActionData(x_train, y_train)
            self.action_data_list.append(action_data)
