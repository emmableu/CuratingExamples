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

label_name_s = ['keymove', 'cochangescore', 'jump',   'movetomouse', 'moveanimate', 'costopall']
# label_name_s = ['keymove', 'costopall', 'cochangescore', 'movetomouse']
# label_name_s = ['movetomouse[1, 0]baseline']
# label_name_s = ['cochangescore[1, 0]baseline']
# label_name_s = ['keymove[1, 0]baseline']

no_model_selection = False




color_s = [i for i in mcolors.TABLEAU_COLORS.keys()] + ['blue', 'blueviolet', 'brown', 'burlywood', 'cadetblue',
                                                        'chartreuse', 'chocolate']


all_data = {}
def get_all_data():
    # print("total = "+  str(total)  +  ", thres = " + str(thres) + "  " + training_method + "  " + specified_info)
    all_repetitions = 10

    for label_index, label_name in enumerate(label_name_s):
        all_data[label_name] = []
        all_simulation_1 = load_obj('all_simulation_' + label_name + "[1, 0]baseline",
                                    base_dir, 'simulation/no_model_selection')

        all_simulation_2 = load_obj('all_simulation_' + label_name + "[1, 0]baseline",
                                    base_dir, 'simulation')


        all_simulation_3 = load_obj('all_simulation_' + label_name,
                                    base_dir, 'simulation/best_train')
        #
        # all_simulation_3 = load_obj('all_simulation_' + label_name + "code_state[[1, 0], [1, 1], [1, 2], [1, 3]]",
        #                             base_dir, 'simulation')
        for all_simulation in [all_simulation_1, all_simulation_2, all_simulation_3]:

            game_y = all_simulation["y"]
            total_pos = Counter(game_y)[1]
            total = len(game_y)
            # print(all_simulation.keys())
            def get_x_y_for_plot(session):
                # order = np.array(game_instance.body['session'])
                unique_order = np.unique(session)
                unique_order.sort()
                x = [start_data]
                unique_order = list(filter((-1).__ne__, unique_order))
                start = start_data
                counter = [start_data]
                # print(unique_order)
                for o in unique_order[2:]:
                    # print("o: ", o, "ind: ")
                    ind = np.where(session == o)
                    # print(game_y[ind])
                    next = start + Counter(game_y[ind])[1]
                    counter.append(next)
                    start = next
                    x.append(start_data + o*step)
                # print("x: ", x)
                # print("y: ", counter)
                return x, counter
            x = {}
            y = {}
            min_len = 300
            for iteration_item in range(all_repetitions):
                x[iteration_item] = get_x_y_for_plot(all_simulation[0])[0]
                y[iteration_item] = get_x_y_for_plot(all_simulation[0])[1]
                if len(x[iteration_item])< min_len:
                    x_axis =  x[iteration_item]
                    # stopping_y =

            baseline_y = []
            average_y = []
            best_y = []



            for number_index, number in enumerate(x_axis):
                # print("number: ", number)

                baseline_y = create_baseline(start_data, total_pos, total, len(x_axis), step, 0.6)
                this_sum = 0
                for iteration_item in range(all_repetitions):
                    x_data, y_data = x[iteration_item], y[iteration_item]
                    try:
                        this_sum += y_data[number_index]
                    except:
                        print("this axis should not be printed")

                average_y.append(this_sum/all_repetitions)
                if number <= total_pos:
                    best_y.append(number)
                else:
                    best_y.append(total_pos)

            auc = get_auc(baseline_y, average_y, best_y)

            all_data[label_name].append([x_axis, baseline_y, average_y, best_y, auc, total, total_pos])
            # print(all_data)



def plot_all():
    # print("total = "+  str(total)  +  ", thres = " + str(thres) + "  " + training_method + "  " + specified_info)
    all_repetitions = 10
    fig = plt.figure(figsize=(24, 24))
    gs = fig.add_gridspec(3, 3)


    for label_index, label_name in enumerate(label_name_s):
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 14}

        plt.rc('font', **font)
        paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
                 'figure.autolayout': True, 'figure.figsize': (16, 8)}

        plt.rcParams.update(paras)
        ax = fig.add_subplot(gs[label_index//3, label_index%3])
        # for i in range(all_repetitions):
        #     plt.plot(x_axis, get_x_y_for_plot(all_simulation[i])[1], marker='o', markerfacecolor='blue', markersize=1,
        #              color=color_s[i], linewidth=1)

        min_index = -1
        min_len = 100
        for i in range(3):
            if len(all_data[label_name][i][0])<min_len:
                min_len = len(all_data[label_name][i][0])
                min_index = i
        # print(all_data['moveanimate'][2])

        plt.plot(all_data[label_name][min_index][0], all_data[label_name][min_index][1], marker='o', markerfacecolor='red', markersize=1,
                 color='red', linewidth=2)
        plt.plot(all_data[label_name][min_index][0], all_data[label_name][min_index][3], marker='o', markerfacecolor='red', markersize=1,
                 color='red', linewidth=2)
        plt.plot(all_data[label_name][min_index][0], all_data[label_name][0][2][:min_len], marker='o', markerfacecolor='black', markersize=1,
                 color='black', linewidth=2, label = 'baseline')
        plt.plot(all_data[label_name][min_index][0], all_data[label_name][1][2][:min_len], marker='o', markerfacecolor='blue', markersize=1,
                 color='blue', linewidth=2, label ='model selection')
        plt.plot(all_data[label_name][min_index][0], all_data[label_name][2][2][:min_len], marker='o', markerfacecolor='blue', markersize=1,
                 color='purple', linewidth=2, label ='best feature & model')
        plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100/ all_data[label_name][0][6]) for x in plt.gca().get_yticks()])
        plt.gca().set_xticklabels(['{:.0f}%'.format((x+start_data+1) * 100/all_data[label_name][0][5]) for x in plt.gca().get_xticks()])
        ax.set_title(label_name_dict[label_name]  + " #P =" + str(all_data[label_name][0][5]) + " AUC = " + str(round(all_data[label_name][1][4], 2)) + "vs" + str(round(all_data[label_name][0][4], 2)))
        plt.legend()





get_all_data()
# plot_all()