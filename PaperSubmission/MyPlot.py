import sys
sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets')
from Dataset import *
from translation_dict import *
sys.setrecursionlimit(10**8)
from ActiveLearnActionData import *
import warnings
warnings.filterwarnings('ignore')

from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 22})
import numpy as np
action_name_s = ['movetomouse', 'moveanimate', 'cochangescore', 'jump','keymove' ]
total_pos_dict = {
    'keymove': 190,
    'jump': 78,
    'cochangescore': 117,
    'movetomouse': 48,
    'moveanimate': 53
}

def get_xy_error_from_session_table(data):

    x = np.array(data.columns.to_list())
    repetition = len(data.index)
    y_array = np.empty([repetition, len(x)])
    for repe in (data.index):
        # print(repe)
        for x_i, x_e in enumerate(x):
            y_array[repe, x_i] = data.at[repe, x_e]
    # print("y_array")

    y = np.mean(y_array, axis=0)
    # y = y_array
    error = np.std(y_array, axis=0)
    # print(y)
    return x, y, error






def get_dict(recall_curve):
    if not recall_curve:
        label_color_dict = {'one_hot':['#E88558', '#F2BCA3'],
                          'pq_gram': ['#9D82BC', '#AD9FBD'],
                          'one_hot_dpm':['#51C176', '#A0DDB4'],
                          'pq_gram_dpm':['#737371', '#878785']}

        label_file_dict = {"one_hot": "code_state[[1, 0]]",
                           'pq_gram': "code_state[[1, 0], [1, 1], [1, 2], [1, 3], [2, 3]]",
                           'one_hot_dpm':  "0.05_dpm_code_state[[1, 0]]",
                           'pq_gram_dpm': "0.05_dpm_code_state[[1, 0], [1, 1], [1, 2], [1, 3], [2, 3]]"}

        general_label_folder_dict = {"passive":base_dir+"Simulation/PatternMining/SessionTable/ActiveLearning/F1Curve/PassiveSampling/",
                                     "model_selection":base_dir+"Simulation/PatternMining/SessionTable/0_1_Digitalized/model_selection",
                                     "uncertainty": base_dir + "Simulation/PatternMining/SessionTable/ActiveLearning/Uncertainty/",
                                     "active": base_dir + "Simulation/PatternMining/SessionTable/ActiveLearning/F1Curve/ModelSelectionAndUncertainty/"}
        general_label_color_dict = {'passive': ['#9D82BC', '#AD9FBD'],
                                    'model_selection': ['#9A7652', '#A88769'],
                                    'uncertainty': ['#A13034', '#E1666A'],
                                    'active': ['#A13034', '#E1666A']}
        return label_color_dict,label_file_dict, general_label_color_dict, general_label_folder_dict

    if recall_curve:

        label_file_dict = {"Baseline": "code_state[[1, 0]]",
            "Estimate": "code_state[[1, 0]]",
            "OneHotImproved": "code_state[[1, 0]]",
            "OneHotImprovedEstimate": "code_state[[1, 0]]",
            "3ConditionsEstimate": "code_state[[1, 0]]",
                           'Improved': "code_state[[1, 0], [1, 1], [1, 2], [1, 3], [2, 3]]"}

        label_folder_dict = {"Baseline": base_dir + "Simulation/PatternMining/SessionTable/ActiveLearning/BaselineCertainty_RecallCurve/",
                                "Estimate": base_dir + "Simulation/PatternMining/SessionTable/ActiveLearning/BaselineCertainty_RecallCurve/EstimateY/",
                             "OneHotImproved":base_dir + "Simulation/PatternMining/SessionTable/ActiveLearning/OneHotModelSelectionUncertainty_10_RecallCurve/",
                             "OneHotImprovedEstimate":base_dir + "Simulation/PatternMining/SessionTable/ActiveLearning/OneHotModelSelectionUncertainty_10_RecallCurve/EstimateY",
                             "3ConditionsEstimate":base_dir + "Simulation/PatternMining/SessionTable/ActiveLearning/OneHotModelSelectionUncertainty_10_RecallCurve/EstimateY_3Conditions",
                                     "Improved": base_dir + "Simulation/PatternMining/SessionTable/ActiveLearning/ModelSelectionUncertainty_10_RecallCurve/"}
        label_color_dict = {'Baseline': ['#9A7652', '#A88769'],
                                    'Estimate': ['#A13034', '#E1666A'],
                                    'OneHotImproved':['#9A7652', '#A88769'],
                                    'OneHotImprovedEstimate': ['#51C176', '#A0DDB4'],
                                    '3ConditionsEstimate': ['#A13034', '#E1666A'],
                                    'Improved': ['#A13034', '#E1666A']}

        formal_legend = { 'OneHotImproved':'Discovered',
                                    '3ConditionsEstimate': "#Positive Estimate",}

        return label_color_dict, label_file_dict, label_folder_dict, formal_legend

def f1_curve():
    label_color_dict, label_file_dict, general_label_color_dict, general_label_folder_dict = get_dict(recall_curve=False)
    plt.clf()
    legend = 'general_label'
    action_name_s = ['keymove', 'jump', 'cochangescore', 'movetomouse', 'moveanimate']
    # action_name_s = ['keymove', 'jump', 'cochangescore']

    # action_name_s = ['keymove']
    if legend == 'label':
        general_label_s = ['passive']
        label_s = ['one_hot', 'one_hot_dpm', 'pq_gram', 'pq_gram_dpm']

    else:
        general_label_s = ['passive', 'uncertainty']
        general_label_s = ['passive', 'model_selection', 'uncertainty', 'active']
        general_label_s = ['passive', 'active']
        # general_label_s = ['passive']
        label_s = ['one_hot']

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
    save_figure(plt)

def recall_curve():
    plt.clf()
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(2, 3)
    label_s = ['Baseline', "3ConditionsEstimate"]
    label_s = ['OneHotImproved', 'OneHotImprovedEstimate','3ConditionsEstimate']
    label_s = ['OneHotImproved','3ConditionsEstimate']
    # label_s = ['Baseline']
    action_name_s = ['keymove', 'jump', 'cochangescore', 'movetomouse', 'moveanimate']
    # action_name_s = ['keymove']
    # action_name_s = ['movetomouse']
    for action_index, action_name in enumerate(action_name_s):
        label_color_dict, label_file_dict, label_folder_dict, formal_legend = get_dict(recall_curve = True)
        ax = fig.add_subplot(gs[action_index // 3, action_index % 3])
        x_all = {}
        y_all = {}
        error_all = {}
        for label in label_s:
            data = load_obj(label_file_dict[label], label_folder_dict[label], action_name)
            x_all[label], y_all[label], error_all[label] = get_xy_error_from_session_table(data)

        shortest = 100
        shortest_label = ""

        total = 413
        total_pos = total_pos_dict[action_name]
        for label in label_s:
            if len(x_all[label]) < shortest:
                shortest_label = label
                shortest = len(x_all[label])
        for label in label_s:
            x = x_all[label][:(len(x_all[shortest_label]))]
            y = y_all[label][:(len(y_all[shortest_label]))]
            order_5 = np.argsort(np.abs(y - 0.5*total_pos))[0]  ## uncertainty sampling by prediction probability
            order_120 = np.argsort(np.abs(y - 1.25*total_pos))[0]
            if label == 'OneHotImproved':
                for order in [order_5]:
                    plt.text(x[order], y[order], '({:.0f}%, {:.0f}%)'.format(x[order]*100/total, y[order] * 100/total_pos))
            if label == '3ConditionsEstimate':
                for order in [order_120]:
                    plt.text(x[order], y[order], '({:.0f}%, {:.0f}%)'.format(x[order]*100/total, y[order] * 100/total_pos))

            error = error_all[label][:(len(error_all[shortest_label]))]
            plt.plot(x, y, 'k', color=label_color_dict[label][0], label=formal_legend[label])
            plt.fill_between(x, y - error, y + error,
                             alpha=0.2, linewidth=2, edgecolor=label_color_dict[label][0],
                             linestyle='dashdot', facecolor=label_color_dict[label][1],
                             antialiased=True)
            ax.set_title(action_name)
            plt.legend()

        plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100/total_pos) for x in plt.gca().get_yticks()])
        plt.gca().set_xticklabels(['{:.0f}%'.format(x * 100/total) for x in plt.gca().get_xticks()])
        ax.set_title(action_name  + " #P =" + str(total_pos))


        # plt.show()


    save_figure(plt)


f1_curve()

