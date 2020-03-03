import sys
sys.path.append("/Users/wwang33/Documents/IJAIED20/CuratingExamples/my_module")
from save_load_pickle import *
import pandas as pd
from CodeShape import *
import os


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
        self.allow_gap = allow_gap

    def get_code_shape_from_code(self, json_code, code_shape_p_q_list, allow_gap=True):
        if allow_gap:
            test_shape = combination(json_code, code_shape_p_q_list)
        else:
            test_shape = get_code_shape(json_code, 'targets', code_shape_p_q_list)
        return test_shape
    def create_input_data(self):
        self.action = {}
        action_name_s = ['keymove', 'jump', 'costopall', 'wrap', 'cochangescore', 'movetomouse', 'moveanimate']
        for action_name in action_name_s:
            d = LabelData(action_name)
            labeldf = d.populate(self.data, action_name)
            pid, abstract, label = [],[],[]
            for i in (labeldf.index):
                p = labeldf.at[i, 'pid']
                pid.append(p)
                json_code = get_json(p)
                print("json_code: ", json_code)
                a = self.get_code_shape_from_code(json_code, self.code_shape_p_q_list)
                abstract.append(a)
                label.append(labeldf.at[i, 'label'])
            self.action[action_name]["pid"] = pid
            self.action[action_name]["abstract"] = abstract
            self.action[action_name]["label"] = label
            self.action[action_name]["df"] = list2df([pid, abstract, label], ['pid', 'abstract', 'label'])












