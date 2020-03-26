import sys
sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples/PaperSubmission/PassiveLearner')
from PatternMining import *
sys.setrecursionlimit(10**8)
from ActiveLearnActionData import *
import warnings
warnings.filterwarnings('ignore')
# from pattern_mining_util import *

def save_hit_for_one_repetition(new_row, save_dir, code_shape_p_q_list, repetition):
    file_name = "code_state" + str(code_shape_p_q_list)
    if is_obj(file_name, save_dir, ""):
        evaluation_metrics = load_obj(file_name, save_dir, "")
        evaluation_metrics.loc[repetition] = new_row
        save_obj(evaluation_metrics, file_name, save_dir, "")
    else:
        atomic_save_hit_for_one_repetition(new_row, save_dir,code_shape_p_q_list, repetition)

def atomic_save_hit_for_one_repetition(new_row, save_dir, code_shape_p_q_list,repetition):
    file_name = "code_state" + str(code_shape_p_q_list)
    df = pd.DataFrame.from_dict({repetition: new_row}, orient="index")
    save_obj(df, file_name, save_dir, "")

def recall_curve(label_name, code_shape_p_q_list):
    x_train, y_train, pattern_orig = get_data(code_shape_p_q_list, digit01= True, action_name = label_name, datafolder = "xy_0heldout")
    total_data = len(y_train)
    all_simulation = {}
    all_simulation["y"] = y_train
    y_pos = Counter(y_train)[1]
    for repetition in range(10):
        new_row = {}
        read = ActiveLearnActionData(x_train, y_train)
        print("read.session: ", read.session)
        total = total_data
        for id in read.start_as_1_pos():
            read.code([id])
            candidate = read.random(read.step)
            hit_count = read.code_recall_curve(candidate)
            cum_hit_count = hit_count
            new_row[read.step + 1] = cum_hit_count
        for j in (range(1, (total-1)//read.step)):
            new_row_key = min((j)*read.step + 11, total)
            if cum_hit_count == y_pos:
                new_row[new_row_key] = cum_hit_count
                continue
            new_row_key = new_row_key
            new_row[new_row_key] = cum_hit_count
            read.get_numbers()
            model, candidate = read.active_model_selection_train()
            hit_count = read.code_recall_curve(candidate)
            cum_hit_count += hit_count
            new_row[new_row_key] = cum_hit_count
        print("new_row: ", new_row)
        save_dir = base_dir + "/Simulation/PatternMining/SessionTable/ActiveLearning/OneHotModelSelectionUncertainty_10_RecallCurve/" + label_name
        if repetition == 0:
            atomic_save_hit_for_one_repetition(new_row, save_dir, code_shape_p_q_list, repetition)
        else:
            save_hit_for_one_repetition(new_row, save_dir, code_shape_p_q_list, repetition)



def baseline_recall_curve(label_name, code_shape_p_q_list):
    x_train, y_train, pattern_orig = get_data(code_shape_p_q_list, digit01= True, action_name = label_name, datafolder = "xy_0heldout")
    total_data = len(y_train)
    all_simulation = {}
    all_simulation["y"] = y_train
    y_pos = Counter(y_train)[1]
    for repetition in range(10):
        new_row = {}
        read = ActiveLearnActionData(x_train, y_train)
        print("read.session: ", read.session)
        total = total_data
        for id in read.start_as_1_pos():
            read.code([id])
            candidate = read.random(read.step)
            hit_count = read.code_recall_curve(candidate)
            cum_hit_count = hit_count
            new_row[read.step + 1] = cum_hit_count
        for j in (range(1, (total-1)//read.step)):
            new_row_key = min((j)*read.step + 11, total)
            new_row_key = new_row_key
            new_row[new_row_key] = cum_hit_count
            read.get_numbers()
            model, candidate = read.passive_train(get_candidate = True)
            hit_count = read.code_recall_curve(candidate)
            cum_hit_count += hit_count
            # if cum_hit_count > y_pos*0.9:
            #     break
            new_row[new_row_key] = cum_hit_count
        print("new_row: ", new_row)
        save_dir = base_dir + "/Simulation/PatternMining/SessionTable/ActiveLearning/BaselineCertainty_RecallCurve/" + label_name
        if repetition == 0:
            atomic_save_hit_for_one_repetition(new_row, save_dir, code_shape_p_q_list, repetition)
        else:
            save_hit_for_one_repetition(new_row, save_dir, code_shape_p_q_list, repetition)



def encapsulated_simulate():
    label_name_s = ['moveanimate']
    code_shape_p_q_list2 = [[1, 0], [1, 1], [1, 2], [1, 3], [2, 3]]
    code_shape_p_q_list1 = [[1, 0]]
    # for code_shape_p_q_list in [code_shape_p_q_list1]:
    #     for label_name in label_name_s:
    #         baseline_recall_curve(label_name, code_shape_p_q_list)
    for code_shape_p_q_list in [code_shape_p_q_list1]:
        for label_name in label_name_s:
            recall_curve(label_name, code_shape_p_q_list)


encapsulated_simulate()