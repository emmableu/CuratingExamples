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


def all_tuning(tuning_methd, validation_f1 = False):
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
            c_grids = [svm_linear_100, svm_linear_10, svm_linear, svm_linear_p1, svm_linear_p01]

            p_thres_grid = [1, 2, 3]
            q_thres_grid = [1, 2, 3, 4]
            n_thres_grid = [2, 3, 4, 5, 6, 7, 8, 9, 10]

            for c_grid in c_grids:
                if tuning_methd == "pqRules":
                    for p_thres in p_thres_grid:
                        for q_thres in q_thres_grid:
                            f1 = get_f1(y_data, train, val, c_grid, p_thres, q_thres, 100)
                            sub_dict[(c_grid, p_thres, q_thres, 100)] = f1
                            grid_score_dict[label + str(fold)] = sub_dict
                elif tuning_methd == "nGramRules":
                    for n_thres in n_thres_grid:
                        f1 = get_f1(y_data, train, val, c_grid, 100, 100, n_thres)
                        sub_dict[(c_grid, 100, 100, n_thres)] = f1
                        grid_score_dict[label + str(fold)] = sub_dict
                elif tuning_methd == "OneHotRules":
                    f1 = get_f1(y_data, train, val, c_grid, 100, 100, 100)
                    sub_dict[(c_grid, 100, 100, 100)] = f1
                    grid_score_dict[label + str(fold)] = sub_dict

            best_key = get_best_key(sub_dict)
            best_score = {"best_grid_set": best_key, "val_f1": sub_dict[best_key]}
            best_score_dict[label + str(fold)] = best_score

            all_train = train + val
            if not validation_f1:
                real_y_pred, y_test = get_y_pred(y_data, all_train, test, best_key[0], best_key[1], best_key[2],
                                             best_key[3])

            elif validation_f1:
                real_y_pred, y_test = get_y_pred(y_data, train, val, best_key[0], best_key[1], best_key[2],
                                             best_key[3])

            full_y_pred.extend(real_y_pred)
            full_y_test.extend(y_test)

        full_y_pred = np.asarray(full_y_pred)
        full_y_test = np.asarray(full_y_test)
        performance = svm_linear.get_matrix(full_y_test, full_y_pred)
        final_score_dict[label] = {"f1": performance['f1'], "recall": performance['recall'],
                                   "precision": performance["precision"]}

    return grid_score_dict, best_score_dict, final_score_dict


def all_tuning_with_grid_from_disk(tuning_methd):
    final_score_dict = {}
    for label in behavior_labels:
        y_data = game_label_data[label].to_numpy()

        full_y_pred = []
        full_y_test = []

        for fold in range(5):
            test = fold_seed[fold]
            val = fold_seed[(fold + 1) % 5]
            train = []
            for j in range(2, 5):
                train += fold_seed[(fold + j) % 5]
            all_train = train + val

            best_key = get_best_key_from_disk(method, label, fold)
            # print(best_key)
            real_y_pred, y_test = get_y_pred(y_data, train, val, best_key[0], best_key[1], best_key[2],
                                             best_key[3])
            # real_y_pred, y_test = get_y_pred(y_data, all_train, test, best_key[0], best_key[1], best_key[2],
            #                                  best_key[3])

            full_y_pred.extend(real_y_pred)
            full_y_test.extend(y_test)

        full_y_pred = np.asarray(full_y_pred)
        full_y_test = np.asarray(full_y_test)
        performance = svm_linear.get_matrix(full_y_test, full_y_pred)
        final_score_dict[label] = {"f1": performance['f1'], "recall": performance['recall'],
                                   "precision": performance["precision"]}

    return final_score_dict


def get_f1(y_data, train, val, c_grid, p_thres, q_thres, n_thres):
    y_pred, y_val = get_y_pred(y_data, train, val, c_grid, p_thres, q_thres, n_thres)
    performance_temp = c_grid.get_matrix(y_val, y_pred)
    f1 = performance_temp['f1']
    return f1


def get_y_pred(y_data, train, val, c_grid, p_thres, q_thres, n_thres):
    if n_thres < 100:
        method = "nGramRules_" + str(n_thres)
    elif p_thres < 100:
        method = "pqRules_" + str(p_thres) + "_" + str(q_thres)
    else:
        method = "OneHotRules"
    pq_rules = pd.read_csv(
        "/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/csedm20/CRV_new_413/" + method + ".csv")
    pq_rule_data = []
    x_data = []
    pq_rule_removal_list = []
    pq_rule_retain_list = []
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




# methods = ["pqRules", "nGramRules"]
methods = ["nGramRules"]
for method in methods:
    rule_data = pd.read_csv(
        "/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/csedm20/CRV_new_413/" + method + ".csv")

    for validation_f1 in [True, False]:
        grid_score_dict, best_score_dict, final_score_dict = all_tuning(method)

        grid_score_df = pd.DataFrame(grid_score_dict)
        best_score_dict = pd.DataFrame(best_score_dict)

        save_obj(grid_score_df, "grid_score_dict", "all" + "_validation" * validation_f1 + "_tuning", method)
        save_obj(best_score_dict, "best_score_dict",  "all" + "_validation" * validation_f1 + "_tuning", method)
        try:
            final_score_dict = pd.Series(final_score_dict)
        except:
            pass
        save_obj(final_score_dict, "final_score_dict",  "all" + "_validation" * validation_f1 + "_tuning",method)
