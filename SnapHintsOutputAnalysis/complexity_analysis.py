import sys

sys.path.append('.')
import operator
from evaluate_snaphints import *

from classifiers.svm_classifiers.SVM_Linear import SVMLinearModel, SVMLinearP01, SVMLinearP1, SVMLinear10, SVMLinear100

svm_linear_p01 = SVMLinearP01()
svm_linear_p1 = SVMLinearP1()
svm_linear_10 = SVMLinear10()
svm_linear_100 = SVMLinear100()

np.set_printoptions(threshold=sys.maxsize)

game_label_data = pd.read_csv(
    "/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets/data/PaperSubmission/game_label_413.csv")
fold_seed = []
this_list = []
for x in range(413):
    if x % 82 == 0 and len(this_list) > 0:
        if x == 410:
            this_list.extend([410, 411, 412])
        fold_seed.append(this_list)
        this_list = []
    this_list.append(x)


def rule_tuning(pq_rules, pq_tuning=False, ngram_tuning = False, training_f1=False):
    grid_score_dict = {}
    best_score_dict = {}
    final_score_dict = {}
    behavior_labels = ["jump"]
    for label in behavior_labels:
        print(label)
        y_data = game_label_data[label].to_numpy()

        full_y_pred = []
        full_y_test = []

        for fold in range(5):
            sub_dict = {}
            test = fold_seed[fold]
            val = fold_seed[(fold + 1) % 5]
            train = []
            for j in range(2, 5):
                train += fold_seed[(fold + j) % 5]
            col_id = label + str(fold) + "Train"

            # support_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
            c_grids = [svm_linear_100, svm_linear_10, svm_linear, svm_linear_p1, svm_linear_p01]

            p_thres_grid = [1, 2, 3]
            q_thres_grid = [1, 2, 3, 4]
            n_thres_grid = [2, 3, 5, 10]

            for c_grid in c_grids:
                if pq_tuning:
                    for p_thres in p_thres_grid:
                        for q_thres in q_thres_grid:
                            f1 = get_f1(pq_rules, -1, y_data, train, val, col_id, c_grid, p_thres, q_thres)
                            sub_dict[(-1, c_grid, p_thres, q_thres)] = f1
                            grid_score_dict[label + str(fold)] = sub_dict
                if ngram_tuning:
                    for n_thres in n_thres_grid:
                        f1 = get_f1(pq_rules, -1, y_data, train, val, col_id, c_grid, p_thres = 100, q_thres=100, n_thres = n_thres)
                        sub_dict[(-1, c_grid, n_thres)] = f1
                        grid_score_dict[label + str(fold)] = sub_dict
                else:
                    f1 = get_f1(pq_rules, -1, y_data, train, val, col_id, c_grid, p_thres=100, q_thres=100,
                                n_thres=100)
                    sub_dict[(-1, c_grid, 100)] = f1
                    grid_score_dict[label + str(fold)] = sub_dict

            best_key = get_best_key(sub_dict)
            best_score = {"best_grid_set": best_key, "val_f1": sub_dict[best_key]}
            best_score_dict[label + str(fold)] = best_score

            all_train = train + val
            all_col_id = label + str(fold) + "TrainVal"
            if training_f1:
                real_y_pred, y_test = get_y_pred(pq_rules, best_key[0], y_data, all_train, all_train, all_col_id,
                                                 best_key[1], best_key[2], best_key[3])
            if pq_tuning:
                    real_y_pred, y_test = get_y_pred(pq_rules, best_key[0], y_data, all_train, test, all_col_id,
                                                 best_key[1], best_key[2], best_key[3])
            elif ngram_tuning:
                real_y_pred, y_test = get_y_pred(pq_rules, best_key[0], y_data, all_train, test, all_col_id,
                                                 best_key[1], 100, 100, best_key[2])
            else:
                real_y_pred, y_test = get_y_pred(pq_rules, best_key[0], y_data, all_train, test, all_col_id,
                                                 best_key[1], 100, 100, 100)

            full_y_pred.extend(real_y_pred)
            full_y_test.extend(y_test)
        full_y_pred = np.asarray(full_y_pred)
        full_y_test = np.asarray(full_y_test)
        performance = svm_linear.get_matrix(full_y_test, full_y_pred)
        final_score_dict[label] = {"f1": performance['f1'], "recall": performance['recall'],
                                   "precision": performance["precision"]}

    return grid_score_dict, best_score_dict, final_score_dict


def get_f1(pq_rules, support, y_data, train, val, col_id, c_grid, p_thres, q_thres, n_thres):
    y_pred, y_val = get_y_pred(pq_rules, support, y_data, train, val, col_id, c_grid, p_thres, q_thres, n_thres)
    performance_temp = c_grid.get_matrix(y_val, y_pred)
    f1 = performance_temp['f1']
    return f1


def get_y_pred(pq_rules, support, y_data, train, val, col_id, c_grid, p_thres=100, q_thres=100, n_thres = 100):
    pq_rule_data = []
    x_data = []
    pq_rule_removal_list = []
    pq_rule_retain_list = []
    if p_thres<100:
        for i in pq_rules.index:
            if pq_rules.at[i, col_id] < support or pq_rules.at[i, "p"] > p_thres or pq_rules.at[i, "q"] > q_thres:
                pq_rule_removal_list.append(pq_rules.at[i, 'ruleID'])
            else:
                pq_rule_retain_list.append(pq_rules.at[i, 'ruleID'])
    if n_thres<100:
        for i in pq_rules.index:
            if pq_rules.at[i, col_id] < support or pq_rules.at[i, "n"] > n_thres:
                pq_rule_removal_list.append(pq_rules.at[i, 'ruleID'])
            else:
                pq_rule_retain_list.append(pq_rules.at[i, 'ruleID'])
    else:
        for i in pq_rules.index:
            pq_rule_retain_list.append(pq_rules.at[i, 'ruleID'])



    pq_rule_data = pq_rules["snapshotVector"].tolist()
    pq_rule_data = [list(eval(item)) for item in pq_rule_data]
    pq_rule_data = np.asarray(pq_rule_data)
    x_data = np.asarray(x_data)
    pq_rule_data = pq_rule_data[pq_rule_retain_list]
    if x_data.shape[0] != 0:
        x_data = np.vstack((pq_rule_data, x_data))
    else:
        x_data = pq_rule_data
    x_data = x_data.transpose()
    x_train = x_data[train]

    print("x_train shape", x_train.shape)

    y_train = y_data[train]
    x_val = x_data[val]
    y_val = y_data[val]
    y_pred = c_grid.get_y_pred(x_train, x_val, y_train)
    return y_pred, y_val


def get_best_key(sub_dict):
    return max(sub_dict.items(), key=operator.itemgetter(1))[0]

# methods = ["pqRules", "OneHotRules", "nGramRules"]
methods_seed = "nGramRules"
methods_seed = "pqRules"
methods = []
# for i in range(2, 11):
#     method = methods_seed + "_" + str(i)
#     methods.append(method)

# for p in range(1, 4):
for p in range(3, 4):
    for q in range(1, 5):
        method = methods_seed + "_" + str(p) + "_" + str(q)
        methods.append(method)

for method in methods:
    rule_data = pd.read_csv(
        "/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/csedm20/CRV_new_413/" + method + ".csv")
    training_f1 = False

    grid_score_dict, best_score_dict, final_score_dict = rule_tuning(rule_data, pq_tuning=False, ngram_tuning=False)
    grid_score_df = pd.DataFrame(grid_score_dict)
    best_score_dict = pd.DataFrame(best_score_dict)
    if training_f1:
        method = "training_" + method
    save_obj(grid_score_df, "grid_score_dict", "score_df_c_tuned", method)
    save_obj(best_score_dict, "best_score_dict", "score_df_c_tuned", method)
    try:
        final_score_dict = pd.Series(final_score_dict)
    except:
        pass
    save_obj(final_score_dict, "final_score_dict", "score_df_c_tuned", method)

# count_dict = {}
# for i in game_label_data.index:
#     for label in behavior_labels:
#         data = game_label_data.at[i, label]
#         if data == 1:
#             if label in count_dict:
#                 count_dict[label] += 1
#             else:
#                 count_dict[label] = 1
#
# print(count_dict)


#