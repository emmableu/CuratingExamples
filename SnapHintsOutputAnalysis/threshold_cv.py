import sys

sys.path.append('.')
import operator
# from evaluate_snaphints import *
from all_tuning import *

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


def threshold_cv(tuning_methd):
    final_score_dict = {}
    for label in behavior_labels:

        p_thres_grid = [1, 2, 3]
        q_thres_grid = [1, 2, 3, 4]
        n_thres_grid = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        c_grids = [svm_linear_100, svm_linear_10, svm_linear, svm_linear_p1, svm_linear_p01]

        if tuning_methd == "pqRules":
            for p_thres in p_thres_grid:
                for q_thres in q_thres_grid:
                    fixed_params = [p_thres, q_thres, 100]
                    record = get_tuned_prediction(method, label, fixed_params)
                    final_score_dict[(label, fixed_params[0], fixed_params[1], fixed_params[2])] = record
        elif tuning_methd == "nGramRules":
            for n_thres in n_thres_grid:
                fixed_params = [100, 100, n_thres]
                record = get_tuned_prediction(method, label, fixed_params)
                final_score_dict[(label, fixed_params[0], fixed_params[1], fixed_params[2])] = record
        else:
            fixed_params = [100, 100, 100]
            record = get_tuned_prediction(method, label, fixed_params)
            final_score_dict[(label, fixed_params[0], fixed_params[1], fixed_params[2])] = record



    return final_score_dict


def get_tuned_prediction(method, label, fixed_params):
    y_data = game_label_data[label].to_numpy()
    full_y_pred = []
    full_y_test = []
    total_feature_count = 0
    for fold in range(5):
        test = fold_seed[fold]
        val = fold_seed[(fold + 1) % 5]
        train = []
        for j in range(2, 5):
            train += fold_seed[(fold + j) % 5]
        best_c = get_best_c_from_disk(method, label, fold, fixed_params[0], fixed_params[1], fixed_params[2])
        all_train = train + val
        real_y_pred, y_test, feature_count = get_y_pred(y_data, all_train, test, best_c, fixed_params[0],
                                                        fixed_params[1], fixed_params[2])
        full_y_pred.extend(real_y_pred)
        full_y_test.extend(y_test)
        total_feature_count += feature_count

    avg_feature_count = total_feature_count/5.0
    full_y_pred = np.asarray(full_y_pred)
    full_y_test = np.asarray(full_y_test)
    performance = svm_linear.get_matrix(full_y_test, full_y_pred)
    record = {"f1": performance['f1'], "recall": performance['recall'],
                                               "precision": performance["precision"],
                                               "avg_feature_count": avg_feature_count}
    return record



def get_one_hot_from_disk():
    final_score_dict = {}
    for label in behavior_labels:
        fixed_params = [100, 100, 100]
        record = get_tuned_prediction(method, label, fixed_params)
        final_score_dict[label] = record
    return final_score_dict






# methods = ["pqRules", "nGramRules"]
methods = ["OneHotRules"]
# methods = ["OneHotRules"]
for method in methods:
    rule_data = pd.read_csv(
        "/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/csedm20/CRV_new_413/" + method + ".csv")
    # final_score_dict = threshold_cv(method)
    # final_score_dict = pd.DataFrame(final_score_dict)
    # save_obj(final_score_dict, "final_score_dict", "threshold_cv", method)
    final_score_dict = get_one_hot_from_disk()
    final_score_dict = pd.DataFrame(final_score_dict)
    save_obj(final_score_dict, "final_score_dict", "all_tuning", method)