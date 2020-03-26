import pickle
import pandas as pd
import os
import numpy as np
import time
import inspect
from tqdm import *
from translation_dict import *
root_dir = "/Users/wwang33/Documents/IJAIED20/CuratingExamples/"
base_dir = root_dir + "Datasets/data/PaperSubmission/"
# base_dir = root_dir + "Datasets/data/ScratchASTData/"
test_size_list = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0]


def save_pickle(obj, name, dir, sub_dir = ""):
    if sub_dir:
        csv_dir = dir +"/"+ sub_dir
        pickle_dir = dir +"/"+ sub_dir + '/pickle_files'
    else:
        csv_dir = dir
        pickle_dir = dir + '/pickle_files'
    atom_mkdir(csv_dir)
    atom_mkdir(pickle_dir)
    # print("pickle_dir: ", pickle_dir)
    with open(pickle_dir + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# for filename in os.listdir(directory):
def save_obj(obj, name, dir, sub_dir = ""):
    if sub_dir:
        csv_dir = dir +"/"+ sub_dir
        pickle_dir = dir +"/"+ sub_dir + '/pickle_files'
    else:
        csv_dir = dir
        pickle_dir = dir + '/pickle_files'
    atom_mkdir(csv_dir)
    save_csv_or_txt(obj, csv_dir + '/' + name)


    atom_mkdir(pickle_dir)
    with open(pickle_dir + '/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def is_obj(name, dir, sub_dir = ""):
    if sub_dir:
        csv_dir = dir +"/"+ sub_dir
        pickle_dir = dir +"/"+ sub_dir + '/pickle_files'
    else:
        csv_dir = dir
        pickle_dir = dir + '/pickle_files'
    atom_mkdir(pickle_dir)
    return os.path.isfile(pickle_dir+ "/" + name + ".pkl")


def load_obj(name, dir, sub_dir = ""):
    if sub_dir:
        pickle_dir = dir +"/"+ sub_dir + '/pickle_files'
    else:
        pickle_dir = dir + '/pickle_files'   

    with open(pickle_dir+ "/"+ name + '.pkl', 'rb') as f:
        pickle_load = pickle.load(f)
        return pickle_load


def save_csv_or_txt(obj, dir_plus_name):
    try:
        obj.to_csv(dir_plus_name + '.csv')
    except:
        with open(dir_plus_name + '.txt', 'w') as f:
            for item in obj:
                f.write("%s\n" % item)

def list2df(list_of_input_list, list_of_input_colnames):
    df = pd.DataFrame(list_of_input_colnames)
    for i in range(len(list_of_input_list)):
        new_row = {}
        for j, colname in enumerate(list_of_input_colnames):
            new_row[colname] = list_of_input_list[j][i]
        df.loc[len(df)] = new_row
    return df

def df2list(df,body):
    from ast import literal_eval
    columns = df.columns
    for column in columns:
        content = df[column].to_list()
        content = [literal_eval(content[index]) for index in range(len(content))]
        body[column] = np.array(content)
    return body


def save_figure(figure, dir = "plots", file_name ="", has_time =True):
    if dir == "plots":
        dir = ("/Users/wwang33/Documents/IJAIED20/CuratingExamples/PaperSubmission/Plots")
    t = time.time()
    if has_time:
        file_name = "" + str(int(t)) + ".png"
    figure.savefig(dir+"/" + file_name)




def atom_mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def hard_drive_get_code_shape_from_pid(pid, code_shape_p_q_list):
    # start = time.time()
    code_shape = {}
    loop_total = [i[0] for i in code_shape_p_q_list]
    for i in loop_total:
        folder =  "/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets/data/SnapASTData/game_labels_415/code_state[[" + str(i) + ", 0]]/pickle_files/"
        file = folder +  "code_state|0|414.pkl"
        with open(file, 'rb') as f:
            pickle_load = pickle.load(f)
            d = (pickle_load)
        code_shape.update(d.loc[pid])

    # folder4 =  "/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets/data/SnapASTData/game_labels_415/code_state[[4, 0]]/pickle_files/"
    # series = pd.Series()
    # for file in os.listdir(folder4):
    #     if file == ".DS_Store":
    #         continue
    #     f_cont = folder4 + file
    #
    #     with open(f_cont, 'rb') as f:
    #         d = pickle.load(f)
    #         series = series.append(d)
    # # print(series)
    # code_shape.update(series[(pid)])
    # end = time.time()
    # print("Time elapsed for: " + inspect.stack()[0][3]+ " is: ", end-start,  " seconds" )
    # print(code_shape)
    return code_shape

def get_all_pid_s():
    pid_s = []
    file_path = root_dir + "/Datasets/data/SnapJSON_413"
    file_list = os.listdir(file_path)
    for file_name in file_list:
        if file_name.endswith(".json"):
            pid = file_name.split(".")[0]
            pid_s.append(pid)
    # print(len(pid_s))
    assert len(pid_s) == 413, "pid length is not 413"
    pid_s.sort()
    save_obj(pid_s, "pid", base_dir, "")





import os
import pandas as pd
import pickle


def get_opposite(l, ind):
    opposite_list = []
    for i, e in enumerate(l):
        if i in ind:
            continue
        else:
            opposite_list.append(e)
    return opposite_list

def rotation(test_size):
    start = list(range(int(test_size*10)))
    rotation_list = []
    for i in range(10):
        new = [0]*len(start)
        rotation_list += [start]
        for j in range(len(start)):
            new[j] = start[j] + 1
            if new[j] >=10:
                new[j] = new[j]%10
        start = new
    return rotation_list

def generate_cv(all_pid_s):
    # all_pid_s = get_all_pid_s()
    pid_length = len(all_pid_s)

    len_test = int(0.1 * pid_length)
    base_data = []
    for i in range(10):
        test_start = i * len_test
        test_end = test_start + len_test
        test_pid = all_pid_s[test_start:test_end]
        base_data.append( test_pid)


    for test_size in test_size_list:
    # for test_size in [0.9]:
        if test_size == 0:
            cv_total = 1
        else:
            cv_total = 10
            rotation_list = rotation(test_size)
            for rotn in rotation_list:
                test_pid = []
                for i in rotn:
                    test_pid = test_pid+ base_data[i]
                compliment_rotn = get_opposite(list(range(10)),rotn)
                train_pid = []
                for j in compliment_rotn:
                    train_pid = train_pid + base_data[j]
                assert_train_test_mutual_exclusive(train_pid, test_pid)
                save_obj(train_pid, "train_pid", base_dir, "cv/test_size" + str(test_size) + "/fold" + str(i))
                save_obj(test_pid, "test_pid", base_dir, "cv/test_size" + str(test_size) + "/fold" + str(i))
    # #

def get_train_test_pid(test_size, fold):
    train_pid = load_obj("train_pid", base_dir, "xy_0heldout")
    test_pid = load_obj("test_pid", base_dir, "xy_0heldout")
    return train_pid, test_pid

def add_by_ele(orig_dict, add_dict):
    for i in orig_dict.keys():
        orig_dict[i] += add_dict[i]
    return orig_dict

def assert_list_equals(l1, l2):
    assert len(l1) == len(l2), "x_axis should be the same as list(range(total-start_data))!"
    for i in range(len(l1)):
        assert l1[i] == l2[i], "x_axis should be the same as list(range(total-start_data))!"

def add_dict_to_set(orig_set, add_dict):
    for i in add_dict.keys():
        orig_set.add(i)
    return orig_set


def yes_no_convert(i):
    if i == 1 or i == "1":
        return "yes"
    else:
        return 'no'

def get_dict_average(dict_name, cv_total):
    for i in dict_name.keys():
        dict_name[i] = dict_name[i] / cv_total
    return dict_name

def atomic_add(old_pattern_set, new_pattern_s):
    for pattern in new_pattern_s:
        old_pattern_set.add(pattern)
    return old_pattern_set


def get_x_y_train_test(get_dir):
    X_train = load_obj("X_train", get_dir, "")
    y_train = load_obj("y_train", get_dir, "")
    X_test = load_obj("X_test", get_dir, "")
    y_test = load_obj("y_test", get_dir, "")
    return X_train, X_test, y_train, y_test





def combine_code_state():

    code_shape_p_q_list = [[1, 0], [1, 1], [1, 2], [1, 3], [2, 3]]
    code_state1 = load_obj("code_state[[1, 0], [1, 1], [1, 2], [1, 3]]", base_dir,'CodeState')
    code_state2 = load_obj("code_state[[2, 3]]", base_dir,'CodeState')
    pid = load_obj('pid', base_dir)
    data_columns = ["code_state" + str(i) for i in code_shape_p_q_list]
    code_state_df = pd.DataFrame(index=pid, columns=data_columns)

    for p in tqdm(pid):
        data = {}
        for e, i in enumerate(data_columns):
            if e == 4:
                data[i] = code_state2.at[p, i]
            else:
                data[i] = code_state1.at[p, i]

        code_state_df.loc[p] = data

    # print(code_state_df)
    save_pickle(code_state_df, "code_state" + str(code_shape_p_q_list), base_dir, 'CodeState')
    print(code_state_df.columns)


def view_code_state():
    pid = load_obj('pid', base_dir)
    code_shape_p_q_list = [[1, 0], [1, 1], [1, 2], [1, 3], [2, 3]]

    data_columns = ["code_state" + str(i) for i in code_shape_p_q_list]

    code_state1 = load_obj("code_state[[1, 0], [1, 1], [1, 2], [1, 3], [2, 3]]", base_dir,'CodeState')
    for p in tqdm(pid):
        data = {}
        for e, i in enumerate(data_columns):
            if e == 4:
                data[i] = code_state1.at[p, i]
                print(data)
                break





# all_pid_s =     load_obj( "pid", base_dir, "")
# generate_cv(all_pid_s)

# start = time.time()
#
# all_pid_s = get_all_pid_s()
# pid_code_shape = {}
# for pid in all_pid_s:
#     code_shape = get_code_shape_from_pid(pid)
#     pid_code_shape[pid] = code_shape
# code_state = pd.Series(pid_code_shape)
# save_pickle(code_state, "code_state_all", root_dir, "Datasets/data/SnapASTData/game_labels_415/code_state[[1, 0], [2, 0], [3, 0]]")
# end = time.time()
# print("Time elapsed for: " + inspect.stack()[0][3] + " is: ", end - start, " seconds")

#
# code_state = load_obj("code_state_all", root_dir, "Datasets/data/SnapASTData/game_labels_415/code_state[[1, 0], [2, 0], [3, 0]]")
# # print(code_state[104765718])

# code_shape_p_q_list = [[1, 0], [2, 0], [3, 0]]
# pattern_set = load_obj( "pattern_set", root_dir + "Datasets/data/SnapASTData",
#          "game_labels_" + str(415) + "/code_state" + str(code_shape_p_q_list))
# for i in pattern_set:
#     if len(i.split("|")) == 1:
#         print(i)

# get_json()

# get_all_pid_s()

# r = rotation(0.4)
# print(r)
# generate_cv()



def print_model(model_dict):
    for model in model_dict:
        print(model.name, ": ", model_dict[model], "; ", end = "")



def assert_train_test_mutual_exclusive(train_pid, test_pid):
    a = set(train_pid)
    b = set(test_pid)
    assert bool(a&b) == False,  "there're items that are both in train and test!"
    return

#
# all_pid_s = load_obj( "pid", base_dir, "")
# generate_cv(all_pid_s)

def create_baseline(start_data, total_pos, total, x_axis_length, step, stopping_rule):
    b = [start_data]
    total_pos = total_pos * stopping_rule
    length_of_data = x_axis_length
    for i in range(1, length_of_data):
        b.append((total_pos-start_data)/total*(step*(i+1)) + start_data)
    return b


def get_auc(baseline_y, average_y, best_y):
    fen_mu = 0
    fen_zi = 0
    for i in range(1, len(baseline_y)):
        fen_mu += best_y[i] - baseline_y[i]
        fen_zi +=  average_y[i] - baseline_y[i]
    if len(baseline_y) == 1:
        # print(baseline_y)
        # return  1
        return 0
    return fen_zi/fen_mu

# combine_code_state()
# view_code_state()
