import sys
sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets')
from Dataset import *
sys.setrecursionlimit(10**8)
from ActiveLearnActionData import *
import warnings
warnings.filterwarnings('ignore')
from pattern_mining_util import *


action_name_s = ['keymove', 'jump', 'cochangescore', 'movetomouse', 'moveanimate']
# code_shape_p_q_list = [[1, 0], [1, 1], [1, 2], [1, 3], [2, 3]]
code_shape_p_q_list = [[1, 0]]
action_name = 'keymove'
orig_dir = base_dir + "/xy_0.3heldout/code_state" + str(code_shape_p_q_list) +  "/" + action_name

x_train = load_obj('X_train', orig_dir, "")
x_train = np.digitize(x_train, bins=[1])

y_train = load_obj('y_train', orig_dir, "")
x_test = load_obj('X_test', orig_dir, "")
x_test= np.digitize(x_test, bins=[1])

y_test = load_obj('y_test', orig_dir, "")

patterns = load_obj("significant_patterns", orig_dir, "")
pattern_orig = np.array([pattern for pattern in patterns])
model_list = [ adaboost, gaussian_nb,
                        bernoulli_nb, multi_nb, complement_nb, mlp, svm_linear]


model = svm_linear

def save_performance_for_one_repetition(new_row, save_dir, repetition):
    file_name = "0.05_dpm_code_state" + str(code_shape_p_q_list)
    if is_obj(file_name, save_dir, ""):
        evaluation_metrics = load_obj(file_name, save_dir, "")
        evaluation_metrics.loc[repetition] = new_row
        save_obj(evaluation_metrics, file_name, save_dir, "")
    else:
        atomic_save_performance_for_one_repetition(new_row, save_dir, repetition)

def atomic_save_performance_for_one_repetition(new_row, save_dir, repetition):
    file_name = "0.05_dpm_code_state" + str(code_shape_p_q_list)
    df = pd.DataFrame.from_dict({repetition: new_row}, orient="index")
    save_obj(df, file_name, save_dir, "")


def pattern_mining(label_name):
    total_data = len(y_train)
    all_simulation = {}
    all_simulation["y"] = y_train
    for repetition in range(10):
        new_row = {}
        read = ActiveLearnActionData(x_train, y_train)
        total = total_data
        for id in read.start_as_1_pos():
            read.code([id])

        for j in (range((total-1)//read.step+1)):
            new_row_key = min((j)*read.step + 11, total)
            new_row_key = new_row_key
            candidate = read.random(read.step)
            read.code(candidate)
            model, selected_feature = read.dpm_passive_train()
            input_test = np.insert(x_test[:, selected_feature], 0, 1, axis=1)
            perf_dict = model.naive_predict(input_test, y_test)

            new_row_value = (perf_dict['f1'])
            new_row[new_row_key] = new_row_value

        print("new_row: ", new_row)
        # break
        save_dir = base_dir + "/Simulation/PatternMining/SessionTable/" + label_name
        if repetition == 0:
            atomic_save_performance_for_one_repetition(new_row, save_dir, repetition)
        else:
            save_performance_for_one_repetition(new_row, save_dir, repetition)


def encapsulated_simulate():
    label_name_s = action_name_s
    for label_name in label_name_s:
        pattern_mining(label_name)



# encapsulated_simulate()

# save_dir = base_dir + "/Simulation/PatternMining/SessionTable"
# cs = load_obj("code_state[[1, 0]]", save_dir)
# print(cs)

def pattern_verification():
    code_shape_p_q_list2 = [[1, 0], [1, 1], [1, 2], [1, 3], [2, 3]]
    code_shape_p_q_list1 = [[1, 0]]
    # action_name = 'keymove'
    orig_dir = base_dir + "/xy_0.3heldout/code_state" + str(code_shape_p_q_list1) + "/" + action_name
    pattern_dir = base_dir + "/xy_0.3heldout/code_state" + str(code_shape_p_q_list1)
    patterns = load_obj("full_patterns", pattern_dir, "")
    # print(patterns)
    pattern_orig1 = np.array([pattern for pattern in patterns])
    x_train1 = load_obj('X_train', orig_dir, "")
    # x_train1 = np.digitize(x_train1, bins=[1])



    orig_dir = base_dir + "/xy_0.3heldout/code_state" + str(code_shape_p_q_list2) + "/" + action_name
    pattern_dir = base_dir + "/xy_0.3heldout/code_state" + str(code_shape_p_q_list2)
    patterns = load_obj("full_patterns", pattern_dir, "")
    pattern_orig2 = np.array([pattern for pattern in patterns])
    # print(pattern_orig2)
    x_train2 = load_obj('X_train', orig_dir, "")
    # x_train2 = np.digitize(x_train2, bins=[1])

    print("onehot length: ", len(pattern_orig1))
    print("pqgram length: ", len(pattern_orig2))


    def get_pattern_index():
        # Actually preform the operation...
        xsorted = np.argsort(pattern_orig2)
        ypos = np.searchsorted(pattern_orig2[xsorted], pattern_orig1)
        indices = xsorted[ypos]


        xsorted = np.argsort(pattern_orig1)
        ypos = np.searchsorted(pattern_orig1[xsorted], pattern_orig1)
        indices2 = xsorted[ypos]
        # one_hot_index =np.where(pattern_orig2 in pattern_orig1)[0]
        # indices = np.where(np.in1d(pattern_orig2, pattern_orig1))[0]

        # one_hot_index = []
        # for i, e in enumerate(pattern_orig1):
        #     if e in pattern_orig2:
        #         one_hot_index.append(np.where(pattern_orig2==e)[0])
        return indices, indices2

    one_hot_index, indices2 = get_pattern_index()
    print(one_hot_index)

    print(sorted(one_hot_index))
    print(pattern_orig2[one_hot_index])
    x2 = (x_train2[:,one_hot_index][0]).astype(int)
    x1 = (x_train1[: ,indices2][0].astype(int))
    print(x2)
    print(x1)
    # code_shape_p_q_list2 = [[1, 0], [1, 1], [1, 2], [1, 3], [2, 3]]
    # code_shape_p_q_list1 = [[1, 0]]
    # orig_dir = base_dir + "/CodeState"
    # patterns = load_obj("code_state" + str(code_shape_p_q_list1), orig_dir, "")
    # pid = load_obj('pid', base_dir)
    # pattern_orig1 = patterns.at[pid[0], 'code_state[1, 0]']
    # orig_dir = base_dir + "/CodeState"
    # patterns = load_obj("code_state" + str(code_shape_p_q_list2), orig_dir, "")
    # pattern_orig2 =  patterns.at[pid[0], 'code_state[1, 0]']
    # # one_hot_patterns = load_obj("pattern0", base_dir + "temp")
    # print("onehot length: ", len(pattern_orig1))
    # print("pqgram length: ", len(pattern_orig2))


def














