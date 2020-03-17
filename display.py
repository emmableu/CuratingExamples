
import sys
sys.path.append("/Users/wwang33/Documents/IJAIED20/CuratingExamples/my_module")
from save_load_pickle import *
import matplotlib.colors as mcolors
label_name_dict =  {'keymove': "Keyboard-Triggered Move", 'jump': "Jump", 'costopall': "Collision-Triggered-Stop-All",
                    'wrap': "Wrap On Screen", 'cochangescore': "Collision-Triggered Change Score",
                    'movetomouse': "Move To or With Mouse",'moveanimate': "Move and Animate"}

import matplotlib.pyplot as plt
root_dir = "/Users/wwang33/Documents/IJAIED20/CuratingExamples/"
total = 415
action_name_s = ['keymove', 'jump', 'costopall', 'wrap', 'cochangescore', 'movetomouse', 'moveanimate']
action_name_s = [ 'cochangescore']

type_list = ['Baseline','KNN','LR','C-Support SVC','Linear-Support SVC','DT','AdaBoost',
        'BaggingClassifier','RandomForest','Gaussian NB','Bernoulli NB','Multinomial NB','Complement NB', 'MLP']

# plot_list  = [ 'code_state[[1, 0]]baseline', 'code_state[[1, 0]]','code_state[[1, 0], [2, 0], [3, 0]]baseline']
plot_list  = ['code_state[[1, 0]]baseline']


def get_all_df(action_name, plot):
    load_dir = base_dir + "/" + plot + "/result/" + action_name
    df = {}
    for i in test_size_list[:-1]:
        x = 1 - i
        df[x] = load_obj("results_test_size" + str(i), load_dir, "")
    return df






def plot(action_name):
    x_axis = [1 - i for i in test_size_list[:-1]]
    color_s = [i for i in mcolors.TABLEAU_COLORS.keys()] + ['blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue',
                                                            'chartreuse', 'chocolate']
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 14}
    fig = plt.figure(figsize=(36, 18))
    gs = fig.add_gridspec(1, 3)
    plt.rc('font', **font)
    paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
             'figure.autolayout': True, 'figure.figsize': (16, 8)}

    plt.rcParams.update(paras)
    for column,plot in enumerate(plot_list):
        all_df = get_all_df(action_name, plot)
        for i, type in enumerate(type_list):
            y_axis = []
            for x in x_axis:
                y_axis.append(all_df[x].recall[type])
            ax = fig.add_subplot(gs[0, column])
            plt.plot(x_axis, y_axis, marker='o', markerfacecolor=color_s[i],
                     markersize=9,
                     color=color_s[i], linewidth=3, label = type)

        plt.axis('tight')
        ax.set_title(label_name_dict[action_name] + " " + plot)
        if column == 0:
            plt.legend()
plt.savefig("/Users/wwang33/Desktop/" + 'figAabccf.png')
# plt.savefig("/Users/wwang33/Desktop/" + 'fig.png')



# plot('cochangescore')





#         for number in x_axis:
#             baseline_y.append(number * total_pos/total)
#             this_sum = 0
#             for iteration_item in range(all_repetitions):
#                 try:
#                     this_sum += get_x_y_for_plot(all_simulation[iteration_item])[1][number]
#                 except:
#                     print("error:  this_sum += get_x_y_for_plot(all_simulation[iteration_item])[1][number]")
#                     this_sum+= 1
#             average_y.append(this_sum/all_repetitions)
#             if number <= total_pos:
#                 best_y.append(number)
#             else:
#                 best_y.append(total_pos)
#
#         color_s = [i for i in mcolors.CSS4_COLORS.keys()]
#         font = {'family': 'normal',
#                 'weight': 'bold',
#                 'size': 14}
#
#         plt.rc('font', **font)
#         paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
#                  'figure.autolayout': True, 'figure.figsize': (16, 8)}
#
#         plt.rcParams.update(paras)
#         ax = fig.add_subplot(gs[label_index//3, label_index%3])
#         for i in (all_types):
#             plt.plot(x_axis, get_x_y_for_plot(all_simulation[i])[1], marker='o', markerfacecolor='blue', markersize=1,
#                      color=color_s[i], linewidth=1)
#         plt.plot(x_axis, baseline_y, marker='o', markerfacecolor='red', markersize=1,
#                  color='red', linewidth=2)
#         plt.plot(x_axis, best_y, marker='o', markerfacecolor='red', markersize=1,
#                  color='red', linewidth=2)
#         plt.plot(x_axis, average_y, marker='o', markerfacecolor='black', markersize=1,
#                  color='black', linewidth=2)
#         plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100/total_pos) for x in plt.gca().get_yticks()])
#         plt.gca().set_xticklabels(['{:.0f}%'.format(x * 100/total) for x in plt.gca().get_xticks()])
#         ax.set_title(label_name_dict[label_name] + " #Positive=" + str(total_pos))
#         plt.axis('tight')
#         plt.savefig("/Users/wwang33/Desktop/" + 'fig.png')
#     plt.savefig("/Users/wwang33/Desktop/" + 'figAll.png')
#
# plot_all(46, 0, "repetition_10_times_no_pole/")
# print("Kernel for this is : RBF")



