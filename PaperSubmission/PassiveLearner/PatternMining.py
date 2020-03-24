import sys
sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets')
from Dataset import *
sys.setrecursionlimit(10**8)
from ActiveLearnActionData import *
import warnings
warnings.filterwarnings('ignore')
from pattern_mining_util import *


action_name_s = ['keymove', 'jump', 'cochangescore', 'movetomouse', 'moveanimate']
model_list = [adaboost, gaussian_nb,
              bernoulli_nb, multi_nb, complement_nb, mlp, svm_linear]
model = svm_linear

def median_digitize(x):
    medium = np.median(x, axis=0)
    # print("medium: ", medium)

    medium[medium == 0] = 1
    # print("medium: ", medium)
    maxi = np.max(x, axis=0)
    maxi[maxi == 0] = 2

    for i in range(len(x[0])):
        bins = np.array([0,medium[i], maxi[i]])
        x[:,i] = np.digitize(x[:,i], bins, right=True)
    return x

def get_data(code_shape_p_q_list, digit01):
    # code_shape_p_q_list = [[1, 0], [1, 1], [1, 2], [1, 3], [2, 3]]
    # code_shape_p_q_list = [[1, 0]]
    action_name = 'cochangescore'
    orig_dir = base_dir + "/xy_0.3heldout/code_state" + str(code_shape_p_q_list) +  "/" + action_name

    x_train = load_obj('X_train', orig_dir, "")
    x_test = load_obj('X_test', orig_dir, "")

    if digit01:
        x_train = np.digitize(x_train, bins=[1])
        x_test = np.digitize(x_test, bins=[1])
    else:
        # print("x_train_before: ", x_train[0][:30])
        x_train = median_digitize(x_train)
        x_test = median_digitize(x_test)
        # print("x_train_after: ", x_train[0][:30])


    y_train = load_obj('y_train', orig_dir, "")

    y_test = load_obj('y_test', orig_dir, "")
    pattern_dir = base_dir + "/xy_0.3heldout/code_state" + str(code_shape_p_q_list)
    patterns = load_obj("full_patterns", pattern_dir, "")
    pattern_orig = np.array([pattern for pattern in patterns])

    return x_train, y_train, x_test, y_test, pattern_orig

def save_performance_for_one_repetition(new_row, save_dir, code_shape_p_q_list, repetition, dpm):
    if dpm:
        file_name = "0.05_dpm_code_state" + str(code_shape_p_q_list)
    else:
        file_name = "code_state" + str(code_shape_p_q_list)
    if is_obj(file_name, save_dir, ""):
        evaluation_metrics = load_obj(file_name, save_dir, "")
        evaluation_metrics.loc[repetition] = new_row
        save_obj(evaluation_metrics, file_name, save_dir, "")
    else:
        atomic_save_performance_for_one_repetition(new_row, save_dir,code_shape_p_q_list, repetition, dpm)

def atomic_save_performance_for_one_repetition(new_row, save_dir, code_shape_p_q_list,repetition, dpm):
    if dpm:
        file_name = "0.05_dpm_code_state" + str(code_shape_p_q_list)
    else:
        file_name = "code_state" + str(code_shape_p_q_list)
    df = pd.DataFrame.from_dict({repetition: new_row}, orient="index")
    save_obj(df, file_name, save_dir, "")

def pattern_mining(label_name, dpm, code_shape_p_q_list, digit01):
    x_train, y_train, x_test, y_test, pattern_orig = get_data(code_shape_p_q_list, digit01)
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
            if dpm:
                if code_shape_p_q_list == [[1, 0]]:
                    model, selected_feature = read.dpm_passive_train(jaccard=False)
                else:
                    model, selected_feature = read.dpm_passive_train(jaccard=True)
                input_test = np.insert(x_test[:, selected_feature], 0, 1, axis=1)
            else:
                model = read.passive_train()
                input_test = np.insert(x_test, 0, 1, axis=1)

            perf_dict = model.naive_predict(input_test, y_test)

            new_row_value = (perf_dict['f1'])
            new_row[new_row_key] = new_row_value

        print("new_row: ", new_row)
        if digit01:
            save_dir = base_dir + "/Simulation/PatternMining/SessionTable/0_1_Digitalized_Jaccard/" + label_name
        else:
            save_dir = base_dir + "/Simulation/PatternMining/SessionTable/0_1_2_Digitalized_Jaccard/" + label_name
        if repetition == 0:
            atomic_save_performance_for_one_repetition(new_row, save_dir, code_shape_p_q_list,repetition, dpm)
        else:
            save_performance_for_one_repetition(new_row, save_dir,code_shape_p_q_list, repetition, dpm)

def encapsulated_simulate():
    label_name_s = action_name_s
    dpm_s = [True, False]
    code_shape_p_q_list_s = [[[1, 0]], [[1, 0], [1, 1], [1, 2], [1, 3], [2, 3]]]
    # digit01_s = [True, False]
    digit01_s = [True]

    # label_name_s = ['keymove']
    # dpm_s = [True]
    # code_shape_p_q_list_s = [[[1, 0]]]
    # digit01_s = [False]

    for label_name in label_name_s:
        for dpm in dpm_s:
            for code_shape_p_q_list in code_shape_p_q_list_s:
                for digit01 in digit01_s:
                    pattern_mining(label_name, dpm, code_shape_p_q_list, digit01)

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

def pattern_examination():
    # code_shape_p_q_list = [[1, 0], [1, 1], [1, 2], [1, 3], [2, 3]]
    # code_shape_p_q_list = [[1, 0]]
    # x_train, y_train, x_test, y_test, pattern_orig = get_data(code_shape_p_q_list)
    # selected_features = select_feature(x_train, y_train)
    # selected_patterns1 = pattern_orig[selected_features]

    code_shape_p_q_list = [[1, 0], [1, 1], [1, 2], [1, 3], [2, 3]]
    x_train, y_train, x_test, y_test, pattern_orig = get_data(code_shape_p_q_list, digit01 = True)
    selected_features = select_feature(x_train, y_train, jaccard = False)
    train_x = np.insert(x_train[:, selected_features], 0, 1, axis=1)
    # print("input x: ", input_x)
    model.model.fit(train_x, y_train)
    input_test = np.insert(x_test[:, selected_features], 0, 1, axis=1)
    perf_dict = model.naive_predict(input_test, y_test)
    print(perf_dict)


    selected_features = select_feature(x_train, y_train, jaccard = True)
    selected_patterns2 = pattern_orig[selected_features]
    save_obj(selected_patterns2, 'cochangescore_jaccard', base_dir + 'temp')
    # selected_patterns2 = pattern_orig[selected_features]
    train_x = np.insert(x_train[:, selected_features], 0, 1, axis=1)
    # print("input x: ", input_x)
    model.model.fit(train_x, y_train)
    input_test = np.insert(x_test[:, selected_features], 0, 1, axis=1)
    perf_dict = model.naive_predict(input_test, y_test)
    print(perf_dict)



    #
    # xsorted = np.argsort(selected_patterns2)
    # ypos = np.searchsorted(selected_patterns2[xsorted], selected_patterns1)
    # indices = xsorted[ypos]
    # print(selected_patterns2[indices])
    # print(len(indices))

# encapsulated_simulate()
pattern_examination()




