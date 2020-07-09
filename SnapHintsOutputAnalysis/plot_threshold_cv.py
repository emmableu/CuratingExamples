import sys

sys.path.append('../Datasets')
from Dataset import *
import copy
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
# plt.rcParams["font.style"] = "italic"
plt.rcParams["font.size"] = 20
plt.figure(figsize=(13, 10))
import matplotlib.colors as mc
import colorsys
from datetime import datetime

# behavior_labels_to_show = ["keymove", "cochangescore", "jump",  "movetomouse", "costopall"]
behavior_labels_to_show = ["jump"]
behavior_labels_to_show = list(reversed(behavior_labels_to_show))
methods_to_show = ['OneHotRules']
# if behavior_labels_to_show[0] == "movetomouse":


for n in [2, 3, 5, 10]:
    methods_to_show.append("nGramRules_" + str(n))

pq_pairs = [[1, 1], [1, 2], [2, 2],  [2, 3], [3, 3], [3, 4]]
for pq_pair in pq_pairs:
    p, q = pq_pair[0], pq_pair[1]
    methods_to_show.append("pqRules_" + str(p) + "_" + str(q))

methods_to_show = list(reversed(methods_to_show))

label_dict = {}
color_dict = {}




def darken_color(color, amount):
    c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])


for method in methods_to_show:
    if method == "OneHotRules":
        label_dict[method] = "Bag-of-Words"
        color_dict[method] = "#D9CCC5"
    else:
        method_list = method.split("_")
        if method_list[0] == "pqRules":
            p, q = method_list[1], method_list[2]
            label_dict[method] = "p(" + str(p) + ")" + "q(" + str(q) + ")-Gram"
            color_dict[method] = darken_color("#868C81", 0.13 * (-int(p) - int(q) + 11.5))
            # color_dict[method] = "#A69586"
        elif method_list[0] == "nGramRules":
            n = method_list[1]
            label_dict[method] = "n(" + str(n) + ")-Gram"
            if n == '10':
                darken_rate = 0.13*3.5
                color_dict[method] = darken_color("#A69586", darken_rate)
            else:
                color_dict[method] = darken_color("#A69586", 0.13 * (-int(n) + 11.5))
            # color_dict[method] = "#868C81"

#
behavior_dict = {
    "keymove": "KeyboardMove (#n = 197/413)",
    "cochangescore": "CollisionChangeVar (#n = 146/413)",
    "jump": "PlatformerJump (#n = 81/413)",
    "movetomouse": "MoveWithMouse (#n = 49/413)",
    "costopall": "CollisionStopGame (#n = 25/413)",
}


def get_bar_data(method, label):
    method_list = method.split("_")
    real_method = method_list[0]
    threshold_cv_grid_score = load_obj("final_score_dict", "threshold_cv", real_method)
    if real_method == "pqRules":
        p, q = method_list[1], method_list[2]
        data =  threshold_cv_grid_score[(label, int(p), int(q), 100)]
    elif real_method == "nGramRules":
        n = method_list[1]
        data = threshold_cv_grid_score[(label, 100, 100, int(n))]
    else:
        data = threshold_cv_grid_score[(label, 100, 100, 100)]
    print(data)
    return data


def grouped_bar_chart():
    barWidth = 0.5
    bars = []
    avg_feature = []
    for method in methods_to_show:
        for label in behavior_labels_to_show:
            data = get_bar_data(method, label)
            d = round(data["f1"], 2)
            bars.append(d)
            avg_feature.append(int(data["avg_feature_count"]))

    print("bars: ", bars)
    r = np.arange(len(bars))
    print("r: ", r)
    ax = plt.axes()
    bar_plots = []
    for i in range(len(methods_to_show)):
        rects = ax.barh(r[i], bars[i], color=color_dict[methods_to_show[i]], height=barWidth,
                        edgecolor=color_dict[methods_to_show[i]], label=label_dict[methods_to_show[i]],  zorder=3)
        bar_plots.append(rects)

    # csfont = {"font}
    def autolabel(rects, r, p):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for j in range(len(rects)):
            rect = rects[j]
            height = rect.get_width()
            # print("height: ", height)
            ax.annotate("F1: " + '{}'.format(height) + " (Avg. # features: " + str(p) + ")",
                        xy=(height, rect.get_y() + rect.get_height() / 2),
                        xytext=(140, -9),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=20, style = "italic")

    for i in range(len(methods_to_show)):
        p = avg_feature[i]
        autolabel(bar_plots[i], r, p)
    # plt.ylabel('Game Behaviors')
    plt.yticks(list(range(len(bars))), label_dict.values(), style = "italic")
    # plt.yticks([r + barWidth for r in range(len(bars[0]))],  ([behavior_dict[i] for i in behavior_labels_to_show]))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    start, end = ax.get_xlim()
    ax.xaxis.set_ticks(np.arange(start, end, 0.1))
    ax.xaxis.grid(zorder=0, color="#F2F2F2", linestyle='dashed', linewidth=1)
    current_handles, current_labels = plt.gca().get_legend_handles_labels()


    plt.title("PlatformerJump - Testing F1 Scores", fontsize = 30)
    plt.tight_layout()
    plt.savefig("plots/threshold_cv" + datetime.now().strftime("%H-%M-%S"))
    plt.show()


# snaphints_crossvalidation()
grouped_bar_chart()
