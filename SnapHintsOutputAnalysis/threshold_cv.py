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


def threshold_cv(tuning_methd, validation_f1 = False):
    # grid_score_dict = {}
    best_score_dict = {}
    final_score_dict = {}
    for label in behavior_labels:
        y_data = game_label_data[label].to_numpy()

        full_y_pred = []
        full_y_test = []


        p_thres_grid = [1, 2, 3]
        q_thres_grid = [1, 2, 3, 4]
        n_thres_grid = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        c_grids = [svm_linear_100, svm_linear_10, svm_linear, svm_linear_p1, svm_linear_p01]

        if tuning_methd == "pqRules":
            for p_thres in p_thres_grid:
                for q_thres in q_thres_grid:



                        for c_grid in c_grids:
                            f1 = get_f1_from_disk(method, label, fold, c_grid, p_thres, q_thres, 100)
                            if f1>best_f1:
                                best_f1 = f1
                                best_c  = c_grid

                        real_y_pred, y_test = get_y_pred(y_data, train, val, best_c, p_thres, q_thres, 100)


        elif tuning_methd == "nGramRules":
            for n_thres in n_thres_grid:

                fixed_params = [100, 100, n_thres]

                f1, feature_dim = get_tuned_prediction(method, label, c_grids, fixed_params)



            full_y_pred.extend(real_y_pred)
            full_y_test.extend(y_test)

        full_y_pred = np.asarray(full_y_pred)
        full_y_test = np.asarray(full_y_test)
        performance = svm_linear.get_matrix(full_y_test, full_y_pred)
        final_score_dict[label] = {"f1": performance['f1'], "recall": performance['recall'],
                                   "precision": performance["precision"]}

    return grid_score_dict, best_score_dict, final_score_dict




def get_tuned_prediction(method, label, c_grids, fixed_params):
    y_data = game_label_data[label].to_numpy()
    full_y_pred = []
    full_y_test = []
    for fold in range(5):
        test = fold_seed[fold]
        val = fold_seed[(fold + 1) % 5]
        train = []
        for j in range(2, 5):
            train += fold_seed[(fold + j) % 5]
        best_c, best_f1 = -1, -1
        for c_grid in c_grids:
            f1 = get_f1_from_disk(method, label, fold, c_grid, fixed_params[0], fixed_params[1], fixed_params[2])
            if f1 > best_f1:
                best_f1 = f1
                best_c = c_grid
        all_train = train + val
        real_y_pred, y_test = get_y_pred(y_data, all_train, test, best_c, fixed_params[0], fixed_params[1], fixed_params[2])
        full_y_pred.extend(real_y_pred)
        full_y_test.extend(y_test)

    full_y_pred = np.asarray(full_y_pred)
    full_y_test = np.asarray(full_y_test)
    performance = svm_linear.get_matrix(full_y_test, full_y_pred)
    final_score_dict[label] = {"f1": performance['f1'], "recall": performance['recall'],
                               "precision": performance["precision"]}


# methods = ["pqRules", "nGramRules"]
methods = ["nGramRules", "pqRules", "OneHotRules"]
# methods = ["OneHotRules"]
for method in methods:
    rule_data = pd.read_csv(
        "/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/csedm20/CRV_new_413/" + method + ".csv")

    # for validation_f1 in [True, False]:
    #     grid_score_dict, best_score_dict, final_score_dict = all_tuning(method)
    #
    #     grid_score_df = pd.DataFrame(grid_score_dict)
    #     best_score_dict = pd.DataFrame(best_score_dict)
    #
    #     save_obj(grid_score_df, "grid_score_dict", "all" + "_validation" * validation_f1 + "_tuning", method)
    #     save_obj(best_score_dict, "best_score_dict",  "all" + "_validation" * validation_f1 + "_tuning", method)
    #     try:
    #         final_score_dict = pd.Series(final_score_dict)
    #     except:
    #         pass
    #     save_obj(final_score_dict, "final_score_dict",  "all" + "_validation" * validation_f1 + "_tuning",method)
    final_score_dict = all_tuning_with_grid_from_disk(method)
    save_obj(final_score_dict, "final_score_dict", "all_tuning", method)