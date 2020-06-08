import sys
sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets')
from Dataset import *
import copy
import matplotlib.pyplot as plt
plt.rcParams["font.family"] ="Times New Roman"
plt.rcParams["font.size"] = 22
plt.figure(figsize=(10,10))

# behavior_labels = ["keymove", "jump", "cochangescore", "movetomouse", "moveanimate", "costopall"]
# behavior_labels = ["jump"]
behavior_labels = ["cochangescore"]
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
    if not support_based_only:
        # diff_params = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5]
        support_diffs = [0.3]
        diff_params = [0.4]
        # yes_params = [0.3, 0.4, 0.5]
        yes_params = [0.3]
        # confidence_params = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        confidence_params = [0.8]
        # feature_grids = [(x, y, z) for x in diff_params for y in yes_params for z in confidence_params]
        max_f1 = 0
        x_orig = copy.deepcopy(x)
        for support_diff in support_diffs:
            for diff_param in diff_params:
                for yes_param in yes_params:
                    for confidence_param in confidence_params:
                        feature_grid = [support_diff, diff_param, yes_param, confidence_param]
                        print(feature_grid)
                        selected_features = get_selected_feature_index(snaphints_dir, feature_grid[0], feature_grid[1], feature_grid[2], feature_grids[3])
                        x = x_orig[:,selected_features]
                        f1 = svm_linear.model_cross_val_predict(x, y)['f1']
                        if f1 >= max_f1:
                            best_selected_features = selected_features
                            selected_feature_grid = feature_grid
                            max_f1 = f1
                            if f1 > 0.9:
                                break
        return x_orig[:, best_selected_features], y, best_selected_features, selected_feature_grid, max_f1

    else:
        support_diff = 0.4
        selected_features = get_support_based_selected_feature_index(snaphints_dir, support_diff)
        return x[:, selected_features], y, selected_features



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



def get_selected_feature_index(snaphints_dir, support_diff, jd_diff, jd_yes_bar, confidence_bar):
    features = pd.read_csv(snaphints_dir + "/features.csv")
    selected_features = []
    for fid in tqdm(features.index):
        jd_yes = features.at[fid, 'jdYes']
        jd_no = features.at[fid, 'jdNo']
        confidence = features.at[fid, 'confidenceYes']
        support_yes = features.at[fid, 'supportA']
        support_no = features.at[fid, 'supportB']
        # if jd_yes == -1:
        #     selected_features.append(fid)
        if support_yes - support_no >= support_diff:
            # print("support_yes - support_no > 0.3")
            # print(features.at[fid, "name"])
            selected_features.append(fid)
        elif jd_yes - jd_no >= jd_diff:
            if jd_yes >= jd_yes_bar:
                # print("jd_yes - jd_no >= jd_diff and jd_yes >= jd_yes_bar")
                # print("jd_yes: ", jd_yes, "jd_no", jd_no, features.at[fid, "name"])
                selected_features.append(fid)
            elif confidence >= confidence_bar:
                selected_features.append(fid)

    return np.array(selected_features)






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
methods = ['AndAllComplete2']


# def visualize_conjunction():
#     snaphints_dir = "/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/submitted/cochangescore/cv/fold1/SnapHintsAndAllComplete2/"
#     yes_x, no_x = get_x_y_split_snaphints(snaphints_dir, [21450, 48436, 48394])
#     # print(x.shape)
#     # print(y.shape)
#     save_obj(yes_x, "yes_x", root_dir)
#     save_obj(no_x, "no_x", root_dir)
#
#
# visualize_conjunction()




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
                                + behavior + "/cv/fold" + str(fold) + "/SnapHints" + method + "/"


                feature_select = True
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
                y_test_total.extend(y_test)
                print(len(y_test_total))
                # y_test_total.extend(y_train)
                # y_pred  = svm_linear.get_y_pred(X_train,  X_train, y_train)
                y_pred  = svm_linear.get_y_pred(X_train,  X_test, y_train)
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


    save_obj(behavior_results, "svm_behaviors11_cochangescore", root_dir, "SnapHintsOutputAnalysis")
    return behavior_results







# snaphints_crossvalidation()

behavior_labels_to_show = ["keymove", "jump", "cochangescore", "movetomouse", "moveanimate", "costopall"]
# behavior_labels_to_show = ["keymove", "jump", "cochangescore", "movetomouse", "moveanimate"]
behavior_labels_to_show = list(reversed(behavior_labels_to_show))
methods_to_show = ['All', 'DPM', "AndAllFull", "AndAll"]
# methods_to_show = ['PQGra', 'DPM', "AndAllFull", "AndAll"]
methods_to_show = ['OneHot2', 'Neighbor', 'All']
methods_to_show = list(reversed(methods_to_show))
label_dict = {'AllOneHot': "One-hot encoding",
              'OneHot2': "One-hot encoding",
            'Neighbor': 'Neighborhood',
                'All': 'All PQGrams',
                'All-6-3': 'PQGrams 6-3',
              'DPM': 'PQGram DPM',
              'AndAllFull': 'All-based Conjunctions',
              'AndAll': 'DPM-based Conjunctions'}

if len(methods_to_show) == 3:
    label_dict = {'OneHot2': "One-hot encoding",
                  'Neighbor': 'Neighborhood',
                  'All': 'PQGrams',
                  'All-6-3': 'PQGrams 6-3',
                  'DPM': 'PQGram DPM',
                  'AndAllFull': 'All-based Conjunctions',
                  'AndAll': 'DPM-based Conjunctions'}

color_dict = { 'AllOneHot': "#D9AE80",
            'Neighbor': '#F2B6BC',
            'All': '#D6F2C2',
               'OneHot2': '#D9AE80',
               'DPM': '#8BD9C3',
              'AndAllFull': '#45B3BF',
              'AndAll': '#1FA2BF'}


def grouped_bar_chart():
    # set width of bar
    # behavior_results = load_obj("svm_behaviors9", root_dir, "SnapHintsOutputAnalysis")
    behavior_results = load_obj("svm_behaviors8", root_dir, "SnapHintsOutputAnalysis")
    barWidth = 0.13
    # print(behavior_results)
    def get_bar(index):
        bars = []
        for behavior in behavior_labels_to_show:
            if methods_to_show[index] == "OneHot2" or methods_to_show[index]== "Neighbor":
                behavior_results2 = load_obj("svm_behaviors10", root_dir, "SnapHintsOutputAnalysis")
                bars.append(behavior_results2.at[(behavior, methods_to_show[index]), "f1"])
            else:
                bars.append(behavior_results.at[(behavior, methods_to_show[index]), "f1"])
        return bars

    bars = []
    for i in range(len(methods_to_show)):
        bars.append(get_bar(i))

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
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_width()
            # print("height: ", height)
            ax.annotate('{}'.format(height),
                        xy=(height, rect.get_y() + rect.get_height() / 2),
                        xytext=(20, -9),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=15)

    for i in range(len(methods_to_show)):
        autolabel(bar_plots[i])
    plt.ylabel('programming behaviors')
    plt.yticks([r + barWidth for r in range(len(bars[0]))],  (behavior_labels_to_show))
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
    # fig.tight_layout()
    plt.savefig("behaviors_4")
    plt.show()


grouped_bar_chart()