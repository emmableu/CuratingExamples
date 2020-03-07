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


# file = ("/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets/data/game_labels_415/pickle_files/code_state[[3, 0]].pkl")
# with open(file, 'rb') as f:
#     pickle_load = pickle.load(f)
#     d = (pickle_load)
#
# d=d.set_index("pid")
# d = d['codeshape_count_dict']
# print(len(d))
# save_pickle(d,  "code_state|" + str(0) + "|" + str(414), "/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets/data/game_labels_415/code_state[[3, 0]]", "")
#




def get_code_shape_from_pid():
    start = time.time()
    file = "/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets/data/game_labels_415/code_state[[3, 0]]/pickle_files/code_state|0|414.pkl"
    with open(file, 'rb') as f:
        pickle_load = pickle.load(f)
        d = (pickle_load)

    print(type(d))
    print(d.loc[331929186])

    end = time.time()
    print("Time elapsed for: " + inspect.stack()[0][3]+ " is: ", end-start,  " seconds" )


get_code_shape_from_pid()