import sys
sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples/PaperSubmission/PassiveLearner')
from PatternMining import *
sys.setrecursionlimit(10**8)
from ActiveLearnActionData import *
import warnings
warnings.filterwarnings('ignore')
# from pattern_mining_util import *


def active_sampling(label_name, code_shape_p_q_list):
    x_train, y_train, x_test, y_test, pattern_orig = get_data(code_shape_p_q_list, digit01= True, action_name = label_name,  datafolder = "xy_0.3heldout")
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





def model_selection_and_uncertainty_sampling(label_name, code_shape_p_q_list):
    x_train, y_train, pattern_orig = get_data(code_shape_p_q_list, digit01= True, action_name = label_name, datafolder = "xy_0heldout")
    total_data = len(y_train)
    all_simulation = {}
    all_simulation["y"] = y_train

    for repetition in range(10):
        new_row = {}
        read = ActiveLearnActionData(x_train, y_train)
        print("read.session: ", read.session)
        total = total_data
        for id in read.start_as_1_pos():
            read.code([id])
            candidate = read.random(read.step)
            read.code(candidate)
            read.get_numbers()
            input_x = np.insert(x_train[read.train_ids1], 0, 1, axis=1)
            perf_dict = svm_linear.model_cross_val_predict(input_x, y_train[read.train_ids1])
            new_row_value = (perf_dict['f1'])
            new_row[read.step + 1] = new_row_value

        for j in (range(1, (total - 1) // read.step)):
            new_row_key = min((j) * read.step + 11, total)
            new_row_key = new_row_key

            read.get_numbers()
            if len(read.poses) == 1:
                candidate= read.random(read.step)
                model = read.passive_train()

            else:
                model, candidate = read.active_model_selection_train(all_uncertainty = True)

            if len(candidate) > 0:
                read.code(candidate)
            input_x = np.insert(x_train, 0, 1, axis=1)
            perf_dict = model.model_cross_val_predict(input_x, y_train)
            new_row_value = (perf_dict['f1'])
            new_row[new_row_key] = new_row_value

        print("new_row: ", new_row)
        save_dir = base_dir + "/Simulation/PatternMining/SessionTable/ActiveLearning/F1Curve/ModelSelectionAndUncertainty/" + label_name
        if repetition == 0:
            atomic_save_performance_for_one_repetition(new_row, save_dir, code_shape_p_q_list,repetition, dpm = False)
        else:
            save_performance_for_one_repetition(new_row, save_dir,code_shape_p_q_list, repetition, dpm = False)

def passive_sampling(label_name, code_shape_p_q_list):
    x_train, y_train, pattern_orig = get_data(code_shape_p_q_list, digit01= True, action_name = label_name, datafolder = "xy_0heldout")
    total_data = len(y_train)
    all_simulation = {}
    all_simulation["y"] = y_train

    for repetition in range(10):
        new_row = {}
        read = ActiveLearnActionData(x_train, y_train)
        print("read.session: ", read.session)
        total = total_data
        for id in read.start_as_1_pos():
            read.code([id])
            candidate = read.random(read.step)
            read.code(candidate)
            read.get_numbers()
            input_x = np.insert(x_train[read.train_ids1], 0, 1, axis=1)
            perf_dict = svm_linear.model_cross_val_predict(input_x, y_train[read.train_ids1])
            new_row_value = (perf_dict['f1'])
            new_row[read.step + 1] = new_row_value

        for j in (range(1, (total - 1) // read.step)):
            new_row_key = min((j) * read.step + 11, total)
            new_row_key = new_row_key
            candidate = read.random(read.step)
            read.code(candidate)
            read.get_numbers()
            input_x = np.insert(x_train[read.train_ids1], 0, 1, axis=1)
            perf_dict = svm_linear.model_cross_val_predict(input_x, y_train[read.train_ids1])
            new_row_value = (perf_dict['f1'])
            new_row[new_row_key] = new_row_value

        print("new_row: ", new_row)
        save_dir = base_dir + "/Simulation/PatternMining/SessionTable/ActiveLearning/F1Curve/PassiveSampling/" + label_name
        if repetition == 0:
            atomic_save_performance_for_one_repetition(new_row, save_dir, code_shape_p_q_list,repetition, dpm = False)
        else:
            save_performance_for_one_repetition(new_row, save_dir,code_shape_p_q_list, repetition, dpm = False)


def encapsulated_simulate():
    label_name_s = ['keymove', 'jump', 'cochangescore', 'movetomouse','moveanimate']
    # label_name_s = ['moveanimate']
    code_shape_p_q_list_s = [[[1, 0], [1, 1], [1, 2], [1, 3], [2, 3]]]
    code_shape_p_q_list_s = [[[1, 0]]]
    for label_name in label_name_s:
        for code_shape_p_q_list in code_shape_p_q_list_s:
                passive_sampling(label_name, code_shape_p_q_list)


encapsulated_simulate()