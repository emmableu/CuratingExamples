from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from trainers.CrossValidationTrainer import *
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
sns.set_style('whitegrid')
#
# y = [0]*34
# y[23] = 1
# y[26] = 1
# y[29] = 1
# y[31] = 1
# y[32] = 1
# y[33] = 1
#
#
# y = np.array(y)
# save_obj(y, 'y_train', save_dir)



from save_load_pickle import *
load_dir_y = base_dir + "xy_0heldout/code_state_orig/jump"
load_dir_x = base_dir + "xy_0heldout/digitized_01"
snaphints_data_dir = "/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/submitted/"




def get_tp_fn(pattern_name):
    x = load_obj("X_train", load_dir_x)
    print(x.shape)
    patterns = load_obj("full_pattern", base_dir)
    pattern_list = list(patterns)
    p_id = pattern_list.index(pattern_name)
    y_pred = x[:, p_id]
    print(y_pred)
    y_true_index = np.where(y_pred ==1)
    # y_true = np.


game_label_csv = pd.read_csv(submission_dir + "/game_label_413.csv")
pid_list = load_obj("pid", submission_dir)
pid_list = [str(pid) for pid in pid_list]

# count = 0
# print(game_label.columns)

# new_game_label = pd.DataFrame(columns = ['pid', 'keymove', 'jump', 'costopall', 'wrap',  'cochangescore', 'movetomouse', 'moveanimate'])
# game_label_names = ['keymove', 'jump', 'costopall', 'wrap',  'cochangescore', 'movetomouse', 'moveanimate']
# for id in game_label.index:
#     data_pid = str(game_label.at[id, 'pid'])
#     if data_pid not in pid_list:
#         continue
#     else:
#         for label_name in game_label_names:
#             file_name = data_pid + ".xml"
#             # source = os.path.join(snaphints_data_dir, "/project1/", file_name)
#             source = snaphints_data_dir + "/project1/" + file_name
#             if game_label.at[id, label_name] == 1:
#
#                 yes_dir  = snaphints_data_dir + "/" + label_name + "/yes_xml/"
#                 atom_mkdir(yes_dir)
#                 target = yes_dir + file_name
#                 shutil.copy(source, target)
#             else:
#                 no_dir = snaphints_data_dir + "/" + label_name + "/no_xml/"
#                 atom_mkdir(no_dir)
#                 target = no_dir + file_name
#                 shutil.copy(source, target)


game_label_names = ['keymove', 'jump', 'costopall', 'wrap',  'cochangescore', 'movetomouse', 'moveanimate']
for game_label in game_label_names:
    pid_folder = root_dir + "/Datasets/data/SnapASTData/Data_413/cv/test_size0.1/"
    write_folder = "/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/submitted/" + game_label + "/cv/"
    for i in range(10):
        pid_fold = pid_folder + "fold" + str(i)
        write_fold = write_folder + "fold" + str(i)
        train_pid = load_obj("train_pid", pid_fold)
        test_pid = load_obj("test_pid", pid_fold)
        yes_no_train_test_pid = pd.DataFrame(columns = ["yes", "no"], index=["train", "test"])
        yes_no_train_test_pid.at['train', 'yes'] = []
        yes_no_train_test_pid.at['train', 'no'] = []
        yes_no_train_test_pid.at['test', 'yes'] = []
        yes_no_train_test_pid.at['test', 'no'] = []
        for id in game_label_csv.index:
            data_pid = str(game_label_csv.at[id, 'pid'])
            if game_label_csv.at[id, game_label] == 1:
                if data_pid in train_pid:
                    yes_no_train_test_pid.at['train', 'yes'].append(data_pid)
                else:
                    yes_no_train_test_pid.at['test', 'yes'].append(data_pid)
            else:
                if data_pid in train_pid:
                    yes_no_train_test_pid.at['train', 'no'].append(data_pid)
                else:
                    yes_no_train_test_pid.at['test', 'no'].append(data_pid)
        yes_no_train_test_pid.index.name = 'partition'
        save_obj(yes_no_train_test_pid, "yes_no_train_test_pid", write_fold)






#
# save_obj(new_game_label, "game_label_413", base_dir)






# get_tp_fn("script|setYPosition|{reportSum}|var:userDef|yPosition")
# print(x.shape)
# print(x[:10])
# print(np.where(x[:10] > 2))

# x = np.zeros([5, 6])
# for i in range(5):
#     x[i, 0] = 1
# # x[1, 3] = 1
# for i in range(2, 5):
#     x[i, 1] = 1
#
# x[3, 1] = 1
# print(x)
# x = x[:,[1,2]]
#
# # x = np.array(x)
#
# y = np.zeros(5)
# y[2] = 1
# y[3] = 1
# y[4] = 1
# y = np.array(y).transpose()
# save_obj(x, 'X_train_manual', save_dir_x)
# save_obj(x, 'y_train_manual', save_dir_y)
#
# # trainer = DPMCrossValidationTrainer()
# # trainer = CrossValidationTrainer()
# # trainer.populate(x, y)
# # trainer.cross_val_get_score()
#
#
#
# pid = load_obj()
#



