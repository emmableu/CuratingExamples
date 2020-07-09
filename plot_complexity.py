import sys
sys.path.append('../Datasets')
from Dataset import *
import copy
import matplotlib.pyplot as plt
plt.rcParams["font.family"] ="Times New Roman"
plt.rcParams["font.size"] = 22
plt.figure(figsize=(13,10))

# behavior_labels = [ "costopall"]
# behavior_labels = ["cochangescore", "keymove", "jump",  "movetomouse", "costopall"]
# behavior_labels_to_show = ["keymove", "cochangescore", "jump",  "movetomouse", "costopall"]
behavior_labels_to_show = ["jump"]
behavior_labels_to_show = list(reversed(behavior_labels_to_show))
methods_to_show = ['OneHot2']
for p in range(1, 4):
    for q in range(1, 5):
        methods_to_show.append("pqGram")
methods_to_show = list(reversed(methods_to_show))

label_dict = {

}

color_dict = { 'AllOneHot': "#D6F2C2",
            'Neighbor': '#8BD9C3',
            'All': '#45B3BF',
            'nGram': '#45B3BF',
            'pqGram': '#45B3BF',
               'OneHotSupport>0':"#D6F2C2",
               'NeighborSupport>0': '#8BD9C3',
               'AllAllFinalSupportOver0': '#45B3BF',
               'OneHot2': '#D6F2C2',
               'DPM': '#8BD9C3',
              'AndAllFull': '#45B3BF',
              'AndAll': '#1FA2BF',
               'AndAllComplete-3-4Gram': '#F2B6BC'}

behavior_dict = {
    "keymove": "KeyboardMove (#n = 197/413)",
    "cochangescore": "CollisionChangeVar (#n = 146/413)",
    "jump":"PlatformerJump (#n = 81/413)",
    "movetomouse":"MoveWithMouse (#n = 49/413)",
    "costopall":"CollisionStopGame (#n = 25/413)",
}


def grouped_bar_chart():
    # set width of bar
    # behavior_results = load_obj("svm_behaviors9", root_dir, "SnapHintsOutputAnalysis")
    behavior_results = load_obj("svm_behaviors15_conjunction_DPM1", root_dir, "SnapHintsOutputAnalysis")
    barWidth = 0.13
    all_methods = ["pqgram_[0.1, 0.2, 0.3, 0.4, 0.5]_conjunction_diff[0.2, 0.3, 0.4, 0.5, 0.6]", "pqgram_only_[0.1, 0.2, 0.3, 0.4, 0.5]",
                   "neighbor_[0.1, 0.2, 0.3, 0.4, 0.5]", "onehot_[0.1, 0.2, 0.3, 0.4, 0.5]"]
    all_methods = ["pqgram_[0.1, 0.2, 0.3, 0.4, 0.5]_conjunction_diff[0.2, 0.3, 0.4, 0.5, 0.6]", "pqgram_all",
                   "neighbor_all", "onehot_all"]
    all_methods = ["training_onehot_[0.1, 0.2, 0.3, 0.4, 0.5]",
                   "training_neighbor_[0.1, 0.2, 0.3, 0.4, 0.5]", "training_pqgram_only_[0.1, 0.2, 0.3, 0.4, 0.5]"]

    all_methods = ["pqRules_[0.1, 0.2, 0.3, 0.4, 0.5]",
                   "NeighborRules_[0.1, 0.2, 0.3, 0.4, 0.5]", "OneHotRules_[0.1, 0.2, 0.3, 0.4, 0.5]"]


    all_methods = ["OneHotRules_[0.1, 0.2, 0.3, 0.4, 0.5]"]

    methods_seed = "pqRules"
    for p in range(1, 4):
        for q in range(1, 5):
            method = methods_seed + "_" + str(p) + "_" + str(q)
            all_methods.append(method)
    bars = []
    recalls = []
    precisions  = []
    for method in all_methods:
        data = load_obj("final_score_dict", "score_df_c_tuned", method)
        bar = []
        recall = []
        precision = []
        # for label in ["costopall", "movetomouse", "jump", "cochangescore", "keymove"]:
        for label in [ "jump"]:
            d = round(data[label]["f1"], 2)
            r = round(data[label]["recall"], 2)
            p = round(data[label]["precision"], 2)
            bar.append(d)
            recall.append(r)
            precision.append(p)

        bars.append(bar)
        recalls.append(recall)
        precisions.append(precision)

    # bars = [[0.23, 0.43, 0.67, 0.54, 0.81], [0.39, 0.42, 0.63, 0.57, 0.83], [0.35, 0.42, 0.60, 0.62, 0.78], [0.32, 0.56, 0.66, 0.53, 0.83]]
    print("bars: ", bars)
    # Set position of bar on X axis
    r = [np.arange(len(bars[0]))]

    for i in range(1, len(methods_to_show)):
        r_prev = r[i - 1]
        r.append([x + barWidth + 0.03 for x in r_prev])

    # Make the plot
    # fig, ax = plt.subplot()
    ax = plt.axes()
    bar_plots = []
    for i in range(len(methods_to_show)):
        rects = ax.barh(r[i], bars[i], color=color_dict[methods_to_show[i]], height=barWidth, edgecolor=color_dict[methods_to_show[i]], label=label_dict[methods_to_show[i]], zorder = 3)
        bar_plots.append(rects)

    # csfont = {"font}
    def autolabel(rects, r, p):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for j in range(len(rects)):
            rect = rects[j]
            height = rect.get_width()
            # print("height: ", height)
            ax.annotate( "F1: " + '{}'.format(height) + " (P: " + str(p[j]) + "; R: " + str(r[j]) + ")",
                        xy=(height, rect.get_y() + rect.get_height() / 2),
                        xytext=(90, -9),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=15)

    for i in range(len(methods_to_show)):
        r = recalls[i]
        p = precisions[i]
        autolabel(bar_plots[i], r, p)
    plt.ylabel('Game Behaviors')
    plt.yticks([r + barWidth for r in range(len(bars[0]))],  ([behavior_dict[i] for i in behavior_labels_to_show]))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 0.1))
    ax.xaxis.grid(zorder=0, color="#F2F2F2", linestyle='dashed', linewidth=1)
    current_handles, current_labels = plt.gca().get_legend_handles_labels()

    # sort or reorder the labels and handles
    reversed_handles = list(reversed(current_handles))
    reversed_labels = list(reversed(current_labels))

    # for i, v in enumerate(x):

    ax.legend(reversed_handles, reversed_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol = 2)

    # Create legend & Show graphic
    plt.title("F1 Scores")
    plt.tight_layout()
    plt.savefig("f1_c_tuned")
    plt.show()


# snaphints_crossvalidation()
grouped_bar_chart()
