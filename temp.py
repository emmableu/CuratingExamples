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


game_label = pd.read_csv(submission_dir + "/game_label_415.csv")
pid_list = load_obj("pid", submission_dir)
pid_list = [str(pid) for pid in pid_list]

count = 0
print(game_label.columns)

new_game_label = pd.DataFrame(columns = ['pid', 'keymove', 'jump', 'costopall', 'wrap',  'cochangescore', 'movetomouse', 'moveanimate'])
for id in game_label.index:
    data_pid = str(game_label.at[id, 'pid'])
    if game_label.ix[id, "good"] == True:
        count += 1
        if data_pid not in pid_list:
            continue
        else:
            new_row = {
                        "pid": data_pid,
                        "keymove": int(game_label.at[id, 'keymove']),
                       "jump": int(game_label.at[id, 'jump']),
                       "costopall": int(game_label.at[id, 'costopall']),
                       "wrap": int(game_label.at[id, 'wrap']),
                       "cochangescore": int(game_label.at[id, 'cochangescore']),
                       "movetomouse": int(game_label.at[id, 'movetomouse']),
                       "moveanimate": int(game_label.at[id, "moveanimate"])}

            new_game_label.loc[len(new_game_label)] = new_row


save_obj(new_game_label, "game_label_413", base_dir)






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



