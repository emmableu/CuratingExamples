import sys
sys.path.append("/Users/wwang33/Documents/IJAIED20/CuratingExamples/my_module")
from save_load_pickle import *
import pandas as pd
from CodeShape import *
from Test import *
from ActionData import *
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm





class LabelData(object):
    """docstring for LabelData."""

    def __init__(self, action_name):
        self.labeldf = pd.DataFrame(columns=['pid', 'label'])
        self.label = action_name

    def populate(self, data, action_name):
        for i in data.index:
            if data.at[i, 'good'] and data.at[i, 'good'] == True:
                pid = data.at[i, 'pid']
                new_row = {
                    'pid': pid,
                    'label': ('yes' if data.at[i, action_name] == True else 'no')
                }
                self.labeldf.loc[len(self.labeldf)] = new_row
        return self.labeldf
        # save_obj(self.labeldf, label, cwd, 'game_labels_' + str(total_games))


class Dataset:

    def __init__(self, total, code_shape_p_q_list, embedding_param = None, allow_gap = True):
        self.root_dir = "/Users/wwang33/Documents/IJAIED20/CuratingExamples/"
        self.total = total
        self.code_shape_p_q_list = code_shape_p_q_list
        self.embedding_param = embedding_param
        self.file_path = self.root_dir + "Datasets/data/game_label_" + str(total) + ".csv"
        self.data = pd.read_csv(self.file_path)
        self.data = self.data[self.data.good == True].reset_index(drop = True)
        print(self.data)
        self.allow_gap = allow_gap


    def get_code_shape_from_code(self, json_code, code_shape_p_q_list, allow_gap=True):
        if allow_gap:
            test_shape = combination(json_code, code_shape_p_q_list)
        else:
            test_shape = get_code_shape(json_code, 'targets', code_shape_p_q_list)
        return test_shape


    def create_code_state(self):
        '''
        :return: pd DataFrame, columns = ['pid', 'codeshape_count_dict']
        example row: ['2312424', {'sprite|repeat': 3, 'sprite|repeat|else': 1}]
        '''
        code_state = pd.DataFrame(columns = ['pid', 'codeshape_count_dict'] )
        for i in tqdm(self.data.index):
            pid = self.data.at[i, 'pid']
            json_code = get_json(pid)
            a = self.get_code_shape_from_code(json_code, self.code_shape_p_q_list)
            # print(a)
            new_row = {"pid": pid, "codeshape_count_dict": a}
            code_state.loc[len(code_state)] = new_row
            # print("pid: ", pid)

        save_pickle(code_state, "code_state" + str(self.code_shape_p_q_list), self.root_dir+"Datasets/data", "game_labels_" + str(self.total))

        return code_state

    def get_train_test_pid(self, test_size, action_name):
        pid = self.data['pid'].to_list()
        train_pid, test_pid = train_test_split(
            pid, test_size=test_size, random_state=0)
        y_train = self.data[self.data.pid.isin(train_pid)][action_name].to_list()
        r = 0
        while len(set(y_train)) == 1:
            r += 1
            train_pid, test_pid = train_test_split(
                pid, test_size=test_size, random_state=r)
            y_train = self.data[self.data.pid.isin(train_pid)][action_name].to_list()

        return train_pid, test_pid

    def get_all_pattern_keys(self):
        code_state = load_obj( "code_state" + str(self.code_shape_p_q_list), self.root_dir+"Datasets/data", "game_labels_" + str(415))
        pool = self.data
        pattern_set = set()
        for i in (pool.index):
            # print(i)
            pid = pool.at[i, 'pid']
            code_shape = self.__get_code_shape_from_pid(pid, code_state)
            new_pattern_s = code_shape.keys()
            self.__atomic_add(new_pattern_s, pattern_set)
        save_obj(pattern_set, "pattern_set", self.root_dir+"Datasets/data", "game_labels_" + str(415))

    def __atomic_add(self, new_pattern_s, old_pattern_set):
        for pattern in new_pattern_s:
            old_pattern_set.add(pattern)

    def __get_code_shape_from_pid(self, pid, code_state):
        for i in code_state.index:
            if code_state.at[i, 'pid'] == pid:
                return code_state.at[i, 'codeshape_count_dict']

    def get_result(self):

        code_state = load_obj( "code_state" + str(self.code_shape_p_q_list), self.root_dir+"Datasets/data", "game_labels_" + str(415))
        action_name_s = ['keymove', 'jump', 'costopall', 'wrap', 'cochangescore', 'movetomouse', 'moveanimate']
        # action_name_s = ['cochangescore']
        for action_name in tqdm(action_name_s):
            print("action_name: ", action_name)
            self.action_data = ActionData(code_state = code_state, game_label = self.data , action_name = action_name)

            save_dir = self.root_dir + "Datasets/data/" + "game_labels_" \
                       + str(self.total) + str(self.code_shape_p_q_list) + "/" + action_name
            for model in tqdm(no_tuning_models):
                for test_size in [3/4, 2/3, 1/2, 1/3]:
                    train_pid, test_pid = self.get_train_test_pid(test_size, action_name)
                    # self.action_data.get_yes_patterns(train_pid)
                    # self.action_data.get_pattern_statistics(train_pid)
                    X_train, X_test, y_train, y_test = self.action_data.get_x_y_train_test(train_pid, test_pid)
                    model.get_and_save_performance(X_train, X_test, y_train, y_test, save_dir, test_size)
                    print("--------------"+  action_name + model.get_name()  +  str(test_size)+ "--------------")
                    print(model.get_confusion_matrix())
                    print(model.get_performance())















