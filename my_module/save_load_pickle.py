import pickle
import pandas as pd
import os
import numpy as np
import time
import inspect

root_dir = "/Users/wwang33/Documents/IJAIED20/CuratingExamples/"
def save_pickle(obj, name, dir, sub_dir):
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
def save_obj(obj, name, dir, sub_dir):
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


def is_obj(name, dir, sub_dir):
    if sub_dir:
        csv_dir = dir +"/"+ sub_dir
        pickle_dir = dir +"/"+ sub_dir + '/pickle_files'
    else:
        csv_dir = dir
        pickle_dir = dir + '/pickle_files'
    atom_mkdir(pickle_dir)
    return os.path.isfile(pickle_dir+ "/" + name + ".pkl")


def load_obj(name, dir, sub_dir):
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

def get_json(pid):
    import json
    file_path = root_dir + "Datasets/data/game_raw_jsons/"
    file_list = os.listdir(file_path)
    for file_name in file_list:
        with open(file_path + '/' + file_name, 'r') as project:  ##  "with" is Python's crash resistant file open
            if file_name == str(pid) + ".json":
                try:
                    json_obj = json.load(project)
                    return json_obj
                except:
                    print("could not find the json file to load!")
                    return



def atom_mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)



#
# d=d.set_index("pid")
# d = d['codeshape_count_dict']
# print(len(d))
# save_pickle(d,  "code_state|" + str(0) + "|" + str(414), "/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets/data/game_labels_415/code_state[[3, 0]]", "")
#




def hard_drive_get_code_shape_from_pid(pid, code_shape_p_q_list):
    # start = time.time()
    code_shape = {}
    loop_total = [i[0] for i in code_shape_p_q_list]
    for i in loop_total:
        folder =  "/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets/data/game_labels_415/code_state[[" + str(i) + ", 0]]/pickle_files/"
        file = folder +  "code_state|0|414.pkl"
        with open(file, 'rb') as f:
            pickle_load = pickle.load(f)
            d = (pickle_load)
        code_shape.update(d.loc[pid])

    # folder4 =  "/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets/data/game_labels_415/code_state[[4, 0]]/pickle_files/"
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
    # start = time.time()
    folder =  "/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets/data/game_labels_415/code_state[[1, 0]]/pickle_files/"
    file = folder +  "code_state|0|414.pkl"
    with open(file, 'rb') as f:
        pickle_load = pickle.load(f)
        d = (pickle_load)
    all_pid_s = d.keys()
    # print(all_pid_s)
    # end = time.time()
    # print("Time elapsed for: " + inspect.stack()[0][3] + " is: ", end - start, " seconds")
    return all_pid_s





import os
import pandas as pd
import pickle


def generate_cv():
    all_pid_s = get_all_pid_s()
    for test_size in [0.9, 3/4, 2/3, 1/2, 1/3]:
    # for test_size in [0.9]:
        cv_total = int(max(1/(1-test_size), 1/test_size))
        for fold in range(cv_total):
            if test_size <= 0.5:
                len_test = int(test_size*415)
                test_start = fold * len_test
                test_end = test_start + len_test
                test_pid = all_pid_s[test_start:test_end].to_list()
                train_pid = all_pid_s[:test_start].to_list() + all_pid_s[test_end:].to_list()
                save_obj(train_pid, "train_pid", root_dir, "Datasets/data/cv/test_size" + str(test_size) + "/fold" + str(fold))
                save_obj(test_pid, "test_pid", root_dir, "Datasets/data/cv/test_size" + str(test_size) + "/fold" + str(fold))
            elif test_size > 0.5:
                len_train = int((1-test_size)*415)
                train_start = fold * len_train
                train_end = train_start + len_train
                train_pid = all_pid_s[train_start:train_end].to_list()
                test_pid = all_pid_s[:train_start].to_list() + all_pid_s[train_end:].to_list()
                save_obj(train_pid, "train_pid", root_dir, "Datasets/data/cv/test_size" + str(test_size) + "/fold" + str(fold))
                save_obj(test_pid, "test_pid", root_dir, "Datasets/data/cv/test_size" + str(test_size) + "/fold" + str(fold))



def get_train_test_pid(test_size, fold):
    train_pid = load_obj("train_pid", root_dir, "Datasets/data/cv/test_size" + str(test_size)+ "/fold" + str(fold))
    test_pid = load_obj("test_pid", root_dir, "Datasets/data/cv/test_size" + str(test_size)+ "/fold" + str(fold))
    return train_pid, test_pid


def add_by_ele(orig_dict, add_dict):
    for i in orig_dict.keys():
        orig_dict[i] += add_dict[i]
    return orig_dict


def get_dict_average(dict_name, cv_total):
    for i in dict_name.keys():
        dict_name[i] = dict_name[i] / cv_total
    return dict_name

def atomic_add(new_pattern_s, old_pattern_set):
    for pattern in new_pattern_s:
        old_pattern_set.add(pattern)
    return old_pattern_set

def get_x_y_train_test(get_dir):
    X_train = load_obj("X_train", get_dir, "")
    y_train = load_obj("y_train", get_dir, "")
    X_test = load_obj("X_test", get_dir, "")
    y_test = load_obj("y_test", get_dir, "")
    return X_train, X_test, y_train, y_test



# generate_cv()


#
# start = time.time()
#
# all_pid_s = get_all_pid_s()
# pid_code_shape = {}
# for pid in all_pid_s:
#     code_shape = get_code_shape_from_pid(pid)
#     pid_code_shape[pid] = code_shape
# code_state = pd.Series(pid_code_shape)
# save_pickle(code_state, "code_state_all", root_dir, "Datasets/data/game_labels_415/code_state[[1, 0], [2, 0], [3, 0]]")
# end = time.time()
# print("Time elapsed for: " + inspect.stack()[0][3] + " is: ", end - start, " seconds")

#
# code_state = load_obj("code_state_all", root_dir, "Datasets/data/game_labels_415/code_state[[1, 0], [2, 0], [3, 0]]")
# # print(code_state[104765718])


# code_shape_p_q_list = [[1, 0], [2, 0], [3, 0]]
# pattern_set = load_obj( "pattern_set", root_dir + "Datasets/data",
#          "game_labels_" + str(415) + "/code_state" + str(code_shape_p_q_list))
# for i in pattern_set:
#     if len(i.split("|")) == 1:
#         print(i)
