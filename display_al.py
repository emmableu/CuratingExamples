import sys
sys.path.append("/Users/wwang33/Documents/IJAIED20/CuratingExamples/my_module")
from save_load_pickle import *
from simulate import *
import matplotlib.colors as mcolors
label_name_dict =  {'keymove': "Keyboard-Triggered Move", 'jump': "Jump", 'costopall': "Collision-Triggered-Stop-All",
                    'wrap': "Wrap On Screen", 'cochangescore': "Collision-Triggered Change Score",
                    'movetomouse': "Move To or With Mouse",'moveanimate': "Move and Animate"}

import numpy as np
import matplotlib.pyplot as plt

label_name_s = ['keymove', 'jump', 'costopall', 'wrap', 'cochangescore', 'movetomouse', 'moveanimate']
label_name_s = ['keymove']
def plot_all():
    # print("total = "+  str(total)  +  ", thres = " + str(thres) + "  " + training_method + "  " + specified_info)
    all_repetitions = 3
    fig = plt.figure(figsize=(24, 24))
    gs = fig.add_gridspec(3, 3)

    for label_index, label_name in enumerate(label_name_s):
        all_simulation = load_obj( 'all_simulation_' + label_name,
                    base_dir, 'simulation')
        game_y = all_simulation["y"]
        total_pos = Counter(game_y)[1]
        total = len(game_y)
        def get_x_y_for_plot(session):
            # order = np.array(game_instance.body['session'])
            unique_order = np.unique(session)
            start = start_data
            counter = [3]
            for o in unique_order[1:]:
                ind = np.where(session == o)
                next = start + game_y[ind]
                counter.append(next)
                start = next

            return unique_order, counter
        x_axis = get_x_y_for_plot(all_simulation[0])[0]
        # print("x_axis: ", x_axis)
        # print(list(range(total-start_data)))
        assert_list_equals(x_axis, list(range(total-start_data)))
        baseline_y = [start_data]
        average_y = [start_data]
        best_y = [start_data]

        for number in x_axis[1:]:
            baseline_y.append((number) * (total_pos- start_data)/(total-start_data-1) +start_data)
            this_sum = 0
            for iteration_item in range(all_repetitions):
                try:
                    this_sum += get_x_y_for_plot(all_simulation[iteration_item])[1][number]
                except:
                    print("error:  this_sum += get_x_y_for_plot(all_simulation[iteration_item])[1][number]")
                    this_sum+= 1
            average_y.append(this_sum/all_repetitions)
            if number+start_data <= total_pos:
                best_y.append(number+start_data)
            else:
                best_y.append(total_pos)

        print("average_y: ", average_y)

        color_s = [i for i in mcolors.CSS4_COLORS.keys()]
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 14}

        plt.rc('font', **font)
        paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
                 'figure.autolayout': True, 'figure.figsize': (16, 8)}

        plt.rcParams.update(paras)
        ax = fig.add_subplot(gs[label_index//3, label_index%3])
        for i in range(all_repetitions):
            plt.plot(x_axis, get_x_y_for_plot(all_simulation[i])[1], marker='o', markerfacecolor='blue', markersize=1,
                     color=color_s[i], linewidth=1)
        plt.plot(x_axis, baseline_y, marker='o', markerfacecolor='red', markersize=1,
                 color='red', linewidth=2)
        plt.plot(x_axis, best_y, marker='o', markerfacecolor='red', markersize=1,
                 color='red', linewidth=2)
        plt.plot(x_axis, average_y, marker='o', markerfacecolor='black', markersize=1,
                 color='black', linewidth=2)
        plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100/total_pos) for x in plt.gca().get_yticks()])
        plt.gca().set_xticklabels(['{:.0f}%'.format((x+start_data+1) * 100/total) for x in plt.gca().get_xticks()])
        ax.set_title(label_name_dict[label_name] + " #Positive=" + str(total_pos))
        plt.axis('tight')
        plt.savefig("/Users/wwang33/Desktop/" + 'fig.png')
    plt.savefig("/Users/wwang33/Desktop/" + 'figAll.png')

# plot_all()
# print("Kernel for this is : RBF")



