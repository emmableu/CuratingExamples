
import sys
sys.path.append('../Datasets')
from Dataset import *
import copy
import matplotlib.pyplot as plt
plt.rcParams["font.family"] ="Times New Roman"
plt.rcParams["font.size"] = 22
plt.figure(figsize=(13,10))
from datetime import datetime
# behavior_labels = [ "costopall"]
behavior_labels = ["cochangescore", "keymove", "jump",  "movetomouse", "costopall"]
# behavior_labels = ["jump"]
# behavior_labels = [ "movetomouse"]
# behavior_labels = ["cochangescore"]
# behavior_labels = ["costopall"]
# behavior_labels = ["jump"]
# behavior_labels = ["keymove", "movetomouse", "moveanimate", "costopall", "jump"]
# behavior_labels = ["costopall", "movetomouse",  "jump", "cochangescore","keymove"]

def get_yes_no(data, yes_no_group):
    yes_x = data[yes_no_group].tolist()
    yes_x = [list(eval(item)) for item in yes_x]
    yes_x = np.array(yes_x).transpose()
    return yes_x

def get_x_y_train_snaphints(snaphints_dir, support_based_only = False):
    data = pd.read_csv(snaphints_dir + "train.csv", index_col = 0)
    yes_x = get_yes_no(data, "yes")
    no_x = get_yes_no(data, "no")
    x = np.vstack((yes_x, no_x))
    y = np.hstack((np.array([1]*yes_x.shape[0]), np.array([0]*no_x.shape[0])))
    return x, y
    # if not support_based_only:
    #     diff_params = [0.2, 0.3, 0.4]
    #     jd_yes_params = [0]
    #     max_f1 = 0
    #     x_orig = copy.deepcopy(x)
    #     count = 0
    #     for diff_param in diff_params:
    #         count += 1
    #         print(count)
    #         print("diff_param: ", diff_param)
    #         selected_features, new_feature_output = get_selected_feature_index(snaphints_dir, diff_param)
    #         # save_obj(new_feature_output, "new_features", snaphints_dir)
    #         if len(selected_features) == 0:
    #             continue
    #         x = x_orig[:,selected_features]
    #         f1 = svm_linear.naive_cross_val_predict(x, y)['f1']
    #         if f1 >= max_f1:
    #             best_selected_features = selected_features
    #             selected_feature_grid = [diff_param]
    #             max_f1 = f1
    #         if f1 > 0.9:
    #             print("f1 > 0.9")
    #             break
    #     return x_orig[:, best_selected_features], y, best_selected_features, selected_feature_grid, max_f1
    #
    # else:
    #     support_diff = 0.4
    #     selected_features = get_support_based_selected_feature_index(snaphints_dir, support_diff)
    #     return x[:, selected_features], y, selected_features



def get_x_y_test_snaphints(snaphints_dir, selected_features):
    data = pd.read_csv(snaphints_dir + "test.csv", index_col = 0)
    yes_x = get_yes_no(data, "yes")
    no_x = get_yes_no(data, "no")
    x = np.vstack((yes_x, no_x))
    y = np.hstack((np.array([1]*yes_x.shape[0]), np.array([0]*no_x.shape[0])))
    x = x[:,selected_features]
    return x, y



def get_x_y_split_snaphints(snaphints_dir, selected_features, train_data = False):
    data = pd.read_csv(snaphints_dir + "train.csv", index_col = 0)
    yes_x = get_yes_no(data, "yes")
    no_x = get_yes_no(data, "no")
    yes_x = yes_x[:,selected_features]
    no_x = no_x[:, selected_features]
    return yes_x, no_x



def get_x_y_snaphints(snaphints_dir, partition):
    data = pd.read_csv(snaphints_dir + partition + ".csv", index_col = 0)
    yes_x = get_yes_no(data, "yes")
    no_x = get_yes_no(data, "no")
    x = np.vstack((yes_x, no_x))
    y = np.hstack((np.array([1]*yes_x.shape[0]), np.array([0]*no_x.shape[0])))
    return x, y



def get_selected_feature_index(snaphints_dir, jd_diff):
    features = pd.read_csv(snaphints_dir + "/features.csv")
    selected_features = []
    new_feature_output = pd.DataFrame(columns = features.columns)
    for fid in tqdm(features.index):
        name = features.at[fid, 'name']
        jd_yes = features.at[fid, 'jdYes']
        # print(jd_yes)
        jd_no = features.at[fid, 'jdNo']
        confidence = features.at[fid, 'confidenceYes']
        support_yes = features.at[fid, 'supportA']
        support_no = features.at[fid, 'supportB']
        if "AND" not in name or jd_yes == -1.0:
            selected_features.append(fid)
            new_feature_output.loc[features.index[fid]] = features.iloc[fid]

        elif jd_yes - jd_no >= jd_diff:
            new_feature_output.loc[features.index[fid]] = features.iloc[fid]
            selected_features.append(fid)
    print(selected_features)
    return np.array(selected_features), new_feature_output


def get_support_based_selected_feature_index(snaphints_dir, support_diff):
    features = pd.read_csv(snaphints_dir + "/features.csv")
    selected_features = []
    for fid in tqdm(features.index):
        support_yes = features.at[fid, 'supportA']
        support_no = features.at[fid, 'supportB']
        if support_yes - support_no >= support_diff:
            selected_features.append(fid)
    return np.array(selected_features)


# methods = ['Neighbor', 'AndAllFull', 'AndAll','DPM', 'All', 'And']
# methods = ['AllOneHot', "OneHot2", 'Neighbor', 'All', 'DPM', "AndAllFull", "AndAll", "All-6-3"]
# methods = ['AllOneHot', 'Neighbor']
# methods = ["AndAllComplete-3-4Gram"]
methods = ["All-6-3"]
# methods = ['OneHotSupport>0', 'NeighborSupport>0', 'AllAllFinalSupportOver0', "AndAllComplete-3-4Gram"]
# methods = [ "AndAllAndAllComplete-3-4Gram-Final"]
# methods = ['AndAllComplete-3-4Gram']
# methods = ["All", "AndAll"]


def snaphints_crossvalidation():
    behavior_results = pd.DataFrame(index=pd.MultiIndex.from_product([behavior_labels, methods]),
                                    columns = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "accuracy": 0, "precision": 0, "recall": 0,
                            "f1": 0, "auc": 0}.keys())
    for behavior in behavior_labels:
        for method in methods:
            y_test_total = []
            y_pred_total = []
            for fold in range(10):
                snaphints_dir = "/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/submitted/" \
                                + behavior + "/cv/fold" + str(fold) + "/SnapHints" + method+ "/"

                feature_select = False
                # feature_select = False
                if feature_select:
                    X_train, y_train, best_features, best_grid, best_f1 = get_x_y_train_snaphints(snaphints_dir, support_based_only = False)
                    # X_train, y_train, best_features = get_x_y_train_snaphints(snaphints_dir, support_based=True)
                    X_test, y_test = get_x_y_test_snaphints(snaphints_dir, best_features)
                    # print("bestfeature: ", best_features)
                    # print("x_train shape: ", X_train.shape)
                    # print("bestf1: ", best_f1)
                else:
                    X_train, y_train = get_x_y_snaphints(snaphints_dir, "train")
                    X_test, y_test = get_x_y_snaphints(snaphints_dir, "test")

                    print("x_train shape: ", X_train.shape)
                    print("x_test shape: ", X_test.shape)
                    # X_train = np.vstack((X_train, X_test))
                    # y_train = np.concatenate([y_train, y_test])
                    np.random.seed(42)
                    np.random.shuffle(X_train)
                    np.random.seed(42)
                    np.random.shuffle(y_train)

                    np.random.seed(43)
                    np.random.shuffle(X_test)
                    np.random.seed(43)
                    np.random.shuffle(y_test)
                y_test_total.extend(y_test)
                # print(len(y_test_total))
                # y_test_total.extend(y_train)
                # y_pred  = svm_linear.get_y_pred(X_train,  X_train, y_train)
                y_pred  = svm_linear.get_y_pred(X_train,  X_test, y_train)
                print("y_pred, y_test: ", y_pred, y_test)
                y_pred_total.extend(y_pred)
                performance_temp = svm_linear.get_matrix(y_test_total, y_pred_total)

            y_pred_total = np.array(y_pred_total)
            y_test_total = np.array(y_test_total)
            print(y_pred_total.shape)
            # print(y_pred_total, y_test_total)
            performance = svm_linear.get_matrix(y_test_total, y_pred_total)
            performance = round_return(performance)
            # performance['name']
            behavior_results.loc[(behavior, method)] = performance
            print(behavior_results)


    # save_obj(behavior_results, "svm_behaviors13_34gram_moveanimate", root_dir, "SnapHintsOutputAnalysis")
    save_obj(behavior_results, "svm_behaviors22", root_dir, "SnapHintsOutputAnalysis")
    return behavior_results



behavior_labels_to_show = ["keymove", "cochangescore", "jump",  "movetomouse"]
# behavior_labels_to_show = ["jump"]
# behavior_labels_to_show = ["keymove", "jump", "cochangescore", "movetomouse", "moveanimate"]
behavior_labels_to_show = list(reversed(behavior_labels_to_show))

methods_to_show = ['OneHotRules', 'nGramRules', 'pqRules']
# methods_to_show = ['pqRules']

methods_to_show = list(reversed(methods_to_show))

label_dict =  { 'OneHotRules': "Bag-of-Words",
            'nGramRules': 'n-Gram',
            'pqRules': 'pq-Gram'}

color_dict = { 'OneHotRules': "#D9CCC5",
            'nGramRules': '#A69586',
            'pqRules': '#868C81'}

behavior_dict = {
    "keymove": "KeyboardMove (#n = 197/413)",
    "cochangescore": "CollisionChangeVar (#n = 146/413)",
    "jump":"PlatformerJump (#n = 81/413)",
    "movetomouse":"MoveWithMouse (#n = 49/413)",
    "costopall":"CollisionStopGame (#n = 25/413)",
}


def grouped_bar_chart():
    barWidth = 0.13

    bars = []
    recalls = []
    precisions  = []
    for method in methods_to_show:
        data = load_obj("final_score_dict", "all_tuning", method)
        # data = load_obj("final_score_dict", "all_tuning", method)
        bar = []
        recall = []
        precision = []
        # for label in ["costopall", "movetomouse", "jump", "cochangescore", "keymove"]:
        for label in behavior_labels_to_show:
            d = round(data[label]["f1"], 2)
            r = round(data[label]["recall"], 2)
            p = round(data[label]["precision"], 2)
            bar.append(d)
            recall.append(r)
            precision.append(p)

        bars.append(bar)
        recalls.append(recall)
        precisions.append(precision)

    print("bars: ", bars)
    r = [np.arange(len(bars[0]))]

    for i in range(1, len(methods_to_show)):
        r_prev = r[i - 1]
        r.append([x + barWidth + 0.08 for x in r_prev])

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
    plt.title("Testing F1 Scores")
    plt.tight_layout()
    plt.savefig("f1_c_tuned"+ datetime.now().strftime("%H-%M-%S"))
    plt.show()


# snaphints_crossvalidation()
grouped_bar_chart()