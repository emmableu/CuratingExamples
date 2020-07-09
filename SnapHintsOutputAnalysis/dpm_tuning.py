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


def dpm_tuning_with_grid_from_disk(tuning_methd):
    grid_score_dict = {}
    best_score_dict = {}
    final_score_dict = {}
    for label in behavior_labels:
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
            all_train = train + val
            sup_pos = label + str(fold) + "Train"
            sup_neg = label + str(fold) + "TrainNegSupport"
            best_key = get_best_key_from_disk(method, label, fold)

            for diff_thres in [0, 0.1, 0.2, 0.3, 0.4, 0.5]:
                f1 = get_f1(y_data, train, val, best_key[0], best_key[1], best_key[2], best_key[3], diff_thres, sup_pos, sup_neg)
                sub_dict[(best_key[0], best_key[1], best_key[2],  best_key[3], diff_thres)] = f1
                grid_score_dict[label + str(fold)] = sub_dict

            best_key = get_best_key(sub_dict)
            best_score = {"best_grid_set": best_key, "val_f1": sub_dict[best_key]}
            best_score_dict[label + str(fold)] = best_score

            all_sup_pos = label + str(fold) + "TrainVal"
            all_sup_neg = label + str(fold) + "TrainValNegSupport"

            real_y_pred, y_test = get_y_pred(y_data, all_train, test, best_key[0], best_key[1], best_key[2],
                                                 best_key[3], best_key[4], all_sup_pos, all_sup_neg)

            full_y_pred.extend(real_y_pred)
            full_y_test.extend(y_test)

        full_y_pred = np.asarray(full_y_pred)
        full_y_test = np.asarray(full_y_test)
        print(full_y_pred, full_y_test)
        performance = svm_linear.get_matrix(full_y_test, full_y_pred)
        final_score_dict[label] = {"f1": performance['f1'], "recall": performance['recall'],
                                   "precision": performance["precision"]}

    return grid_score_dict, best_score_dict, final_score_dict


def get_f1(y_data, train, val, c_grid, p_thres, q_thres, n_thres, diff_thres, sup_pos, sup_neg):
    y_pred, y_val = get_y_pred(y_data, train, val, c_grid, p_thres, q_thres, n_thres, diff_thres,sup_pos, sup_neg)
    if len(y_pred)==0:
        return -1
    performance_temp = c_grid.get_matrix(y_val, y_pred)
    f1 = performance_temp['f1']

    return f1


def get_y_pred(y_data, train, val, c_grid, p_thres, q_thres, n_thres, diff_thres, sup_pos, sup_neg):
    if n_thres < 100:
        method = "nGramRules_" + str(n_thres)
    elif p_thres < 100:
        method = "pqRules_" + str(p_thres) + "_" + str(q_thres)
    pq_rules = pd.read_csv(
        "/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/csedm20/CRV_new_413/" + method + ".csv")
    pq_rule_data = []
    x_data = []
    pq_rule_removal_list = []
    pq_rule_retain_list = []
    for i in pq_rules.index:
        support_difference = pq_rules.at[i, sup_pos] - pq_rules.at[i, sup_neg]
        if support_difference < diff_thres:
            pq_rule_removal_list.append(pq_rules.at[i, 'ruleID'])
        else:
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

    if x_train.shape[1] == 0:
        return [], []
    y_train = y_data[train]
    x_val = x_data[val]
    y_val = y_data[val]
    y_pred = c_grid.get_y_pred(x_train, x_val, y_train)
    return y_pred, y_val





methods = ["pqRules"]
# methods = ["nGramRules"]
for method in methods:
    grid_score_dict, best_score_dict, final_score_dict = dpm_tuning_with_grid_from_disk(method)

    grid_score_df = pd.DataFrame(grid_score_dict)
    best_score_dict = pd.DataFrame(best_score_dict)

    save_obj(grid_score_df, "grid_score_dict", "dpm_tuning", method)
    save_obj(best_score_dict, "best_score_dict", "dpm_tuning", method)
    try:
        final_score_dict = pd.Series(final_score_dict)
    except:
        pass
    save_obj(final_score_dict, "final_score_dict", "dpm_tuning", method)



