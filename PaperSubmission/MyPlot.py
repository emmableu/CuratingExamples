import sys
sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets')
from Dataset import *
sys.setrecursionlimit(10**8)
from ActiveLearnActionData import *
import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt
import numpy as np


def get_xy_error_from_session_table(data):
    x = np.array(data.columns.to_list())
    repetition = len(data.index)
    y_array = np.empty([repetition, len(x)])
    for repe in (data.index):
        for x_i, x_e in enumerate(x):
            y_array[repe, x_i] = data.at[repe, x_e]


    y = np.mean(y_array, axis=0)
    error = np.std(y_array, axis=0)
    return x, y, error


plt.clf()

action_name_s =  ['keymove', 'jump', 'cochangescore', 'movetomouse', 'moveanimate']
action_name_s =  ['keymove', 'jump', 'cochangescore']



label_color_dict = {'one_hot':['#E88558', '#F2BCA3'],
                  'pq_gram': ['#9D82BC', '#AD9FBD'],
                  'one_hot_dpm':['#51C176', '#A0DDB4'],
                  'pq_gram_dpm':['#737371', '#878785']}

label_file_dict = {"one_hot": "dpm_code_state[[1, 0]]",
                   'pq_gram': "dpm_code_state[[1, 0], [1, 1], [1, 2], [1, 3], [2, 3]]",
                   'one_hot_dpm':  "0.05_dpm_code_state[[1, 0]]",
                   'pq_gram_dpm': "0.05_dpm_code_state[[1, 0], [1, 1], [1, 2], [1, 3], [2, 3]]"}


label_s = ['one_hot', 'pq_gram', 'one_hot_dpm', 'pq_gram_dpm']
# label_s = [ 'one_hot','one_hot_dpm',  'pq_gram_dpm']

fig = plt.figure(figsize=(24, 16))
gs = fig.add_gridspec(2, 3)
for action_index, action_name in enumerate(action_name_s):

    ax = fig.add_subplot(gs[action_index // 3, action_index % 3])
    for label in label_s:
        data = load_obj(label_file_dict[label], base_dir+"Simulation/PatternMining/SessionTable/0_1_Digitalized", action_name)
        x, y, error = get_xy_error_from_session_table(data)
        plt.plot(x, y, 'k', color=label_color_dict[label][0], label =label)
        plt.fill_between(x, y-error, y+error,
            alpha=0.2, linewidth=2, edgecolor=label_color_dict[label][0],
                        linestyle='dashdot', facecolor=label_color_dict[label][1], antialiased=True)
    ax.set_title(action_name)
    plt.legend()

# plt.show()


save_figure(plt)