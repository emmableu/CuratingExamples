# from sklearn.feature_extraction.text import CountVectorizer
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from trainers.CrossValidationTrainer import *
# import matplotlib.pyplot as plt
# from sklearn.feature_extraction.text import CountVectorizer
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os
# import shutil
# sns.set_style('whitegrid')
# #
# # y = [0]*34
# # y[23] = 1
# # y[26] = 1
# # y[29] = 1
# # y[31] = 1
# # y[32] = 1
# # y[33] = 1
# #
# #
# # y = np.array(y)
# # save_obj(y, 'y_train', save_dir)
#
#
#
# from save_load_pickle import *
# load_dir_y = base_dir + "xy_0heldout/code_state_orig/jump"
# load_dir_x = base_dir + "xy_0heldout/digitized_01"
# snaphints_data_dir = "/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/csedm20_tr_val_te_split/"
#
#
#
#
# # def get_tp_fn(pattern_name):
# #     x = load_obj("X_train", load_dir_x)
# #     print(x.shape)
# #     patterns = load_obj("full_pattern", base_dir)
# #     pattern_list = list(patterns)
# #     p_id = pattern_list.index(pattern_name)
# #     y_pred = x[:, p_id]
# #     print(y_pred)
# #     y_true_index = np.where(y_pred ==1)
# #     # y_true = np.
# #
#
#
#
# # game_label = pd.read_csv(submission_dir + "/game_label_413_original.csv")
# # pid_list = load_obj("pid", submission_dir)
# # pid_list = [str(pid) for pid in pid_list]
# #
# # count = 0
# # print(game_label.columns)
# #
# # new_game_label = pd.DataFrame(columns = ['pid', 'keymove', 'jump', 'costopall', 'wrap',  'cochangescore', 'movetomouse', 'moveanimate'])
# # for id in game_label.index:
# #     data_pid = str(game_label.at[id, 'pid'])
# #     if game_label.ix[id, "good"] == True:
# #         count += 1
# #         if data_pid not in pid_list:
# #             continue
# #         else:
# #             new_row = {
# #                         "pid": data_pid,
# #                         "keymove": int(game_label.at[id, 'keymove']),
# #                        "jump": int(game_label.at[id, 'jump']),
# #                        "costopall": int(game_label.at[id, 'costopall']),
# #                        "wrap": int(game_label.at[id, 'wrap']),
# #                        "cochangescore": int(game_label.at[id, 'cochangescore']),
# #                        "movetomouse": int(game_label.at[id, 'movetomouse']),
# #                        "moveanimate": int(game_label.at[id, "moveanimate"])}
# #
# #             new_game_label.loc[len(new_game_label)] = new_row
# #
# #
# # save_obj(new_game_label, "game_label_413", submission_dir)
# #
# #
#
# game_label_csv = pd.read_csv(submission_dir + "/game_label_413.csv")
# pid_list = load_obj("pid", submission_dir)
# pid_list = [str(pid) for pid in pid_list]
#
#
#
#
# game_label_names = ['keymove', 'jump', 'costopall', 'wrap',  'cochangescore', 'movetomouse', 'moveanimate']
#
# def generate_train_test_pid():
#     for game_label in game_label_names:
#         pid_folder = root_dir + "/Datasets/data/SnapASTData/Data_413/cv/test_size0.1/"
#         write_folder = "/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/csedm20/" + game_label + "/cv/"
#         for i in range(10):
#             pid_fold = pid_folder + "fold" + str(i)
#             write_fold = write_folder + "fold" + str(i)
#             train_pid = load_obj("train_pid", pid_fold)
#             test_pid = load_obj("test_pid", pid_fold)
#             yes_no_train_test_pid = pd.DataFrame(columns = ["yes", "no"], index=["train", "test"])
#             yes_no_train_test_pid.at['train', 'yes'] = []
#             yes_no_train_test_pid.at['train', 'no'] = []
#             yes_no_train_test_pid.at['test', 'yes'] = []
#             yes_no_train_test_pid.at['test', 'no'] = []
#             for id in game_label_csv.index:
#                 data_pid = str(game_label_csv.at[id, 'pid'])
#                 if game_label_csv.at[id, game_label] == 1:
#                     if data_pid in train_pid:
#                         yes_no_train_test_pid.at['train', 'yes'].append(data_pid)
#                     else:
#                         yes_no_train_test_pid.at['test', 'yes'].append(data_pid)
#                 else:
#                     if data_pid in train_pid:
#                         yes_no_train_test_pid.at['train', 'no'].append(data_pid)
#                     else:
#                         yes_no_train_test_pid.at['test', 'no'].append(data_pid)
#             yes_no_train_test_pid.index.name = 'partition'
#             save_obj(yes_no_train_test_pid, "yes_no_train_test_pid", write_fold)
#
#
#
#
#
#
#
from shutil import *
import sys
# sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples/SnapHintsOutputAnalysis')
# from SnapHintsOutputAnalysis.evaluate_snaphints import *


# for behavior in behavior_labels:
#     for fold in range(10):
#         source = "/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/submitted/" \
#                         + behavior + "/cv/fold" + str(fold) + "/SnapHintsAllAllFinalSupportOver0/"
#         destination = '/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets/data/SnapHintsData/submitted/'\
#                         + behavior + "/cv/fold" + str(fold) + "/SnapHintsAllAllFinalSupportOver0/"
#         copytree(source, destination, ignore=ignore_patterns('*.pyc', 'tmp*','__pycache__'))


# game_label = pd.read_csv(submission_dir + "/game_label_413_original.csv")
