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
# action_name_s =  ['keymove', 'jump', 'cochangescore']
# action_name_s =  ['keymove']



label_color_dict = {'one_hot':['#E88558', '#F2BCA3'],
                  'pq_gram': ['#9D82BC', '#AD9FBD'],
                  'one_hot_dpm':['#51C176', '#A0DDB4'],
                  'pq_gram_dpm':['#737371', '#878785']}

label_file_dict = {"one_hot": "code_state[[1, 0]]",
                   'pq_gram': "code_state[[1, 0], [1, 1], [1, 2], [1, 3], [2, 3]]",
                   'one_hot_dpm':  "0.05_dpm_code_state[[1, 0]]",
                   'pq_gram_dpm': "0.05_dpm_code_state[[1, 0], [1, 1], [1, 2], [1, 3], [2, 3]]"}

general_label_folder_dict = {"passive":base_dir+"Simulation/PatternMining/SessionTable/0_1_Digitalized",
                             "model_selection":base_dir+"Simulation/PatternMining/SessionTable/0_1_Digitalized/model_selection",
                             "uncertainty": base_dir + "Simulation/PatternMining/SessionTable/ActiveLearning/Uncertainty/"}



# general_label_s = ['passive', 'model_selection', 'uncertainty']
# label_s = [ 'one_hot','one_hot_dpm',  'pq_gram_dpm']
# general_label_s = ['passive', 'model_selection']
general_label_s = ['passive']
# label_s = ['pq_gram']

legend = 'label'
if legend == 'label':
    general_label_s = ['passive']
    label_s = ['one_hot', 'one_hot_dpm', 'pq_gram', 'pq_gram_dpm']

else:
    general_label_s = ['passive', 'model_selection', 'uncertainty']
    general_label_s = ['passive', 'model_selection']
    label_s = ['pq_gram']


general_label_color_dict = {'passive':['#9D82BC', '#AD9FBD'],
                  'model_selection': ['#9A7652', '#A88769'],
                  'uncertainty':['#A13034', '#E1666A']}


fig = plt.figure(figsize=(24, 16))
gs = fig.add_gridspec(2, 3)
for action_index, action_name in enumerate(action_name_s):

    ax = fig.add_subplot(gs[action_index // 3, action_index % 3])
    for general_label in general_label_s:
        for label in label_s:
            data = load_obj(label_file_dict[label], general_label_folder_dict[general_label], action_name)
            x, y, error = get_xy_error_from_session_table(data)

            if legend == "label":
                plt.plot(x, y, 'k', color=label_color_dict[label][0], label =label)
                plt.fill_between(x, y-error, y+error,
                    alpha=0.2, linewidth=2, edgecolor=label_color_dict[label][0],
                                linestyle='dashdot', facecolor=label_color_dict[label][1], antialiased=True)
            else:
                plt.plot(x, y, 'k', color=general_label_color_dict[general_label][0], label=general_label)
                plt.fill_between(x, y - error, y + error,
                                 alpha=0.2, linewidth=2, edgecolor=general_label_color_dict[general_label][0],
                                 linestyle='dashdot', facecolor=general_label_color_dict[general_label][1], antialiased=True)
        ax.set_title(action_name)
        plt.legend()





# plt.show()


save_figure(plt)