import sys

sys.path.append('.')
import operator
from evaluate_snaphints import *
np.set_printoptions(threshold=sys.maxsize)

# pq_rules = pd.read_csv("/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/csedm20/CRV/pqRules.csv")
game_label_data = pd.read_csv("/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/game_label_413.csv")
# print(game_label_data.head())
fold_seed = []
this_list = []
for x in range(413):
    if x % 82 == 0 and len(this_list) > 0:
        if x == 410:
            this_list.extend([410, 411, 412])
        fold_seed.append(this_list)
        this_list = []
    this_list.append(x)
# conjunction_rules = pd.read_csv(
#     "/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/csedm20/CRV/ConjunctionRules.csv")


pq_rules = pd.read_csv(
    "/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/csedm20/CRV/OneHotRules.csv")
pq_rules = pd.read_csv(
    "/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/csedm20/CRV/NeighborRules.csv")

pq_rule_category = ["Train", "TrainVal"]

conjunction_rule_category = ["TrainSupport", "TrainjdPos", "TrainjdNeg", "TrainValSupport", "TrainValjdPos",
                             "TrainValjdNeg"]
# print(conjunction_rules.columns)


def pq_rule_tuning():
    grid_score_dict = {}

    for label in behavior_labels:
        print(label)
        y_data = game_label_data[label].to_numpy()
        for fold in range(5):
            test = fold_seed[fold]
            val = fold_seed[(fold + 1) % 5]
            train = []
            for j in range(2, 5):
                train += fold_seed[(fold + j) % 5]

            col_id = label + str(fold) + "TrainVal"

            grid_list = [0.1, 0.2, 0.3, 0.4, 0.5]
            sub_dict = {}
            for grid in grid_list:
                x_data = []
                for i in pq_rules.index:
                    if pq_rules.at[i, col_id] < grid:
                        continue
                    else:
                        x_data.append(list(eval(pq_rules.at[i, "snapshotVector"])))

                x_data = np.asarray(x_data)
                x_data = x_data.transpose()
                x_train = x_data[train]
                y_train = y_data[train]
                x_val = x_data[val]
                y_val = y_data[val]
                y_pred = svm_linear.get_y_pred(x_train, x_val, y_train)
                performance_temp = svm_linear.get_matrix(y_val, y_pred)
                f1 = performance_temp['f1']
                sub_dict[grid] = f1

            break
        grid_score_dict[label] = sub_dict
        break

    print(grid_score_dict)


def conjunction_rule_tuning():
    grid_score_dict = {}
    best_score_dict = {}
    final_score_dict = {}
    for label in behavior_labels:
        print(label)
        y_data = game_label_data[label].to_numpy()
        # np.random.shuffle(y_data)

        full_y_pred = []
        full_y_test = []

        for fold in range(5):
            sub_dict = {}
            test = fold_seed[fold]
            # print("fold " , fold , " test: ", test)
            val = fold_seed[(fold + 1) % 5]
            train = []
            for j in range(2, 5):
                train += fold_seed[(fold + j) % 5]
            col_id = label + str(fold) + "Train"
            jd_pos = label + str(fold) + "TrainjdPos"
            jd_neg = label + str(fold) + "TrainjdNeg"

            # support_grid = [0.1, 0.3]
            # support_grid = [0.1, 0.2, 0.3]
            support_grid = [0.1, 0.2, 0.3, 0.4, 0.5]
            jd_diff_grid = [0.2, 0.3, 0.4, 0.5, 0.6]
            # jd_diff_grid = [1]
            # jd_diff_grid = [0.2, 0.6]
            for support in support_grid:
                for jd_diff in jd_diff_grid:
                    # for removal in removal_grid:
                    f1 = get_f1(support, jd_diff, y_data, train, val, col_id, jd_pos, jd_neg)
                    sub_dict[(support, jd_diff)] = f1
                    grid_score_dict[label + str(fold)] = sub_dict

            best_key = get_best_key(sub_dict)
            best_score = {"best_grid_set": best_key, "val_f1": sub_dict[best_key]}
            best_score_dict[label + str(fold)] = best_score

            all_train = train + val
            all_col_id = label + str(fold) + "TrainVal"
            all_jd_pos = label + str(fold) + "TrainValjdPos"
            all_jd_neg = label + str(fold) + "TrainValjdNeg"

            real_y_pred, y_test = get_y_pred(best_key[0], best_key[1], y_data, all_train, test, all_col_id, all_jd_pos,
                                             all_jd_neg)
            # print("real_y_pred, y_test: ", real_y_pred, y_test)
            fake_performance = svm_linear.get_matrix(y_test, real_y_pred)
            full_y_pred.extend(real_y_pred)
            full_y_test.extend(y_test)
        full_y_pred = np.asarray(full_y_pred)
        full_y_test = np.asarray(full_y_test)
        performance = svm_linear.get_matrix(full_y_test, full_y_pred)
        final_score_dict[label] = performance['f1']

    return grid_score_dict, best_score_dict, final_score_dict


def get_f1(support, jd_diff, y_data, train, val, col_id, jd_pos, jd_neg):
    y_pred, y_val = get_y_pred(support, jd_diff, y_data, train, val, col_id, jd_pos, jd_neg)
    performance_temp = svm_linear.get_matrix(y_val, y_pred)
    f1 = performance_temp['f1']
    return f1


def get_y_pred(support, jd_diff, y_data, train, val, col_id, jd_pos, jd_neg):
    pq_rule_data = []
    x_data = []
    pq_rule_removal_list = []
    pq_rule_retain_list = []
    for i in pq_rules.index:
        if pq_rules.at[i, col_id] < support or ("snapshot" in pq_rules.at[i, 'grams']):
            pq_rule_removal_list.append(pq_rules.at[i, 'ruleID'])
        else:
            pq_rule_retain_list.append(pq_rules.at[i, 'ruleID'])

    # for i in conjunction_rules.index:
    #     if (conjunction_rules.at[i, 'ruleIDA'] in pq_rule_removal_list) or (
    #             conjunction_rules.at[i, 'ruleIDB'] in pq_rule_removal_list):
    #         continue
    #     if (conjunction_rules.at[i, jd_pos] - conjunction_rules.at[i, jd_neg]) < jd_diff:
    #         continue
    #     x_data.append(list(eval(conjunction_rules.at[i, "snapshotVector"])))

        # if conjunction_rules.at[i, jd_pos] > 0.8:
        #     try:
        #         pq_rule_retain_list.remove(conjunction_rules.at[i, 'ruleIDA'])
        #         pq_rule_retain_list.remove(conjunction_rules.at[i, 'ruleIDB'])
        #     except:
        #         pass

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
    # print("x_val: ", x_val)
    y_pred = svm_linear.get_y_pred(x_train, x_val, y_train)
    return y_pred, y_val


def get_best_key(sub_dict):
    return max(sub_dict.items(), key=operator.itemgetter(1))[0]


def test_simulate():
    grid_score_dict = {}

    for label in behavior_labels:
        print(label)
        y_data = game_label_data[label].to_numpy()
        x_data = []
        for i in pq_rules.index:
            x_data.append(list(eval(pq_rules.at[i, "snapshotVector"])))
        x_data = np.asarray(x_data)
        print(x_data.shape)

        x_data = x_data.transpose()
        performance_temp = svm_linear.naive_cross_val_predict(x_data, y_data, 10)

        grid_score_dict[label] = performance_temp

    return grid_score_dict


grid_score_dict, best_score_dict, final_score_dict = conjunction_rule_tuning()

grid_score_df = pd.DataFrame(grid_score_dict)
best_score_dict = pd.DataFrame(best_score_dict)
save_obj(grid_score_df, "grid_score_dict", "score_df", "neighbor_[0.1, 0.2, 0.3, 0.4, 0.5]")
save_obj(best_score_dict, "best_score_dict", "score_df", "neighbor_[0.1, 0.2, 0.3, 0.4, 0.5]")
try:
    final_score_dict = pd.Series(final_score_dict)
except:
    pass

save_obj(final_score_dict, "final_score_dict", "score_df", "neighbor_[0.1, 0.2, 0.3, 0.4, 0.5]")

# grid_score_dict = test_simulate()
# save_obj(grid_score_dict, "test_simulate", "score_df")
