import sys
sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples/PaperSubmission/PassiveLearner')
from PatternMining import *
sys.setrecursionlimit(10**8)
from ActiveLearnActionData import *
import warnings
warnings.filterwarnings('ignore')
# from pattern_mining_util import *


def active_sampling(label_name, code_shape_p_q_list):
    x_train, y_train, x_test, y_test, pattern_orig = get_data(code_shape_p_q_list, digit01= True, action_name = label_name)
    total_data = len(y_train)
    all_simulation = {}
    all_simulation["y"] = y_train
    for repetition in range(10):
        new_row = {}
        read = ActiveLearnActionData(x_train, y_train)
        total = total_data
        for id in read.start_as_1_pos():
            read.code([id])
            candidate = read.random(read.step)
            read.code(candidate)

        for j in (range((total-1)//read.step+1)):
            new_row_key = min((j)*read.step + 11, total)
            new_row_key = new_row_key
            model, candidate = read.active_uncertainty_train()
            if len(candidate) > 0:
                read.code(candidate)
            input_test = np.insert(x_test, 0, 1, axis=1)
            perf_dict = model.naive_predict(input_test, y_test)
            new_row_value = (perf_dict['f1'])
            new_row[new_row_key] = new_row_value

        print("new_row: ", new_row)
        save_dir = base_dir + "/Simulation/PatternMining/SessionTable/ActiveLearning/Uncertainty/" + label_name
        if repetition == 0:
            atomic_save_performance_for_one_repetition(new_row, save_dir, code_shape_p_q_list,repetition, dpm = False)
        else:
            save_performance_for_one_repetition(new_row, save_dir,code_shape_p_q_list, repetition, dpm = False)

def encapsulated_simulate():
    label_name_s = ['movetomouse']
    code_shape_p_q_list_s = [[[1, 0], [1, 1], [1, 2], [1, 3], [2, 3]]]
    for label_name in label_name_s:
        for code_shape_p_q_list in code_shape_p_q_list_s:
                active_sampling(label_name, code_shape_p_q_list)


encapsulated_simulate()