
from simulate import *
# from simulate_game_all import *
import matplotlib.colors as mcolors
label_name_dict =  {'keymove': "Keyboard-Triggered Move", 'jump': "Jump", 'costopall': "Collision-Triggered-Stop-All",
                    'wrap': "Wrap On Screen", 'cochangescore': "Collision-Triggered Change Score",
                    'movetomouse': "Move To or With Mouse",'moveanimate': "Move and Animate"}

import numpy as np
import matplotlib.pyplot as plt

total = 415
label_name_s = [ 'cochangescore']

type = ['Baseline','KNN','LR','C-Support SVC','Nu-Support SVC','Linear-Support SVC','DT','AdaBoost',
        'BaggingClassifier','RandomForest','Gaussian NB','Bernoulli NB','Multinomial NB','Complement NB', 'MLP']

grid_list  = ['']


def plot_all(total, thres,training_method = "", specified_info = ""):
    print("total = "+  str(total)  +  ", thres = " + str(thres) + "  " + training_method + "  " + specified_info)
    all_repetitions = 5
    fig = plt.figure(figsize=(24, 24))
    gs = fig.add_gridspec(2, 2)

    for label_index, label_name in enumerate(label_name_s):
        all_simulation = load_obj( '/all_simulation_' + label_name,
                    "/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets/data/game_labels_" + str(415),
                    'simulation_' + str(thres) + "_" + "all/no_pole")
        game = all_simulation[0]
        total_pos = game.est_num
        def get_x_y_for_plot(game_instance):
            order = np.argsort(np.array(game_instance.body['time'])[game_instance.labeled])
            seq = np.array(game_instance.body['code'])[np.array(game_instance.labeled)[order]]
            counter = 0
            rec = [0]
            for s in seq:
                if s == 'yes':
                    counter += 1
                rec.append(counter)
            return range(len(rec)), rec
        x_axis = get_x_y_for_plot(all_simulation[0])[0]
        baseline_y = []
        average_y = []
        best_y = []

        for number in x_axis:
            baseline_y.append(number * total_pos/total)
            this_sum = 0
            for iteration_item in range(all_repetitions):
                try:
                    this_sum += get_x_y_for_plot(all_simulation[iteration_item])[1][number]
                except:
                    print("error:  this_sum += get_x_y_for_plot(all_simulation[iteration_item])[1][number]")
                    this_sum+= 1
            average_y.append(this_sum/all_repetitions)
            if number <= total_pos:
                best_y.append(number)
            else:
                best_y.append(total_pos)

        color_s = [i for i in mcolors.CSS4_COLORS.keys()]
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 14}

        plt.rc('font', **font)
        paras = {'lines.linewidth': 5, 'legend.fontsize': 20, 'axes.labelsize': 30, 'legend.frameon': False,
                 'figure.autolayout': True, 'figure.figsize': (16, 8)}

        plt.rcParams.update(paras)
        ax = fig.add_subplot(gs[label_index//3, label_index%3])
        for i in (all_types):
            plt.plot(x_axis, get_x_y_for_plot(all_simulation[i])[1], marker='o', markerfacecolor='blue', markersize=1,
                     color=color_s[i], linewidth=1)
        plt.plot(x_axis, baseline_y, marker='o', markerfacecolor='red', markersize=1,
                 color='red', linewidth=2)
        plt.plot(x_axis, best_y, marker='o', markerfacecolor='red', markersize=1,
                 color='red', linewidth=2)
        plt.plot(x_axis, average_y, marker='o', markerfacecolor='black', markersize=1,
                 color='black', linewidth=2)
        plt.gca().set_yticklabels(['{:.0f}%'.format(x * 100/total_pos) for x in plt.gca().get_yticks()])
        plt.gca().set_xticklabels(['{:.0f}%'.format(x * 100/total) for x in plt.gca().get_xticks()])
        ax.set_title(label_name_dict[label_name] + " #Positive=" + str(total_pos))
        plt.axis('tight')
        plt.savefig("/Users/wwang33/Desktop/" + 'fig.png')
    plt.savefig("/Users/wwang33/Desktop/" + 'figAll.png')

plot_all(46, 0, "repetition_10_times_no_pole/")
print("Kernel for this is : RBF")



