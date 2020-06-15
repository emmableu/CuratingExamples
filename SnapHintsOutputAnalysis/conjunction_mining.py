import sys
sys.path.append('.')
from evaluate_snaphints import *
from sklearn.model_selection import StratifiedKFold
methods = ["All", "AndAll"]
# methods = ["OneHot", "Neighbor"]


def main():
    behavior_results = pd.DataFrame(index=pd.MultiIndex.from_product([behavior_labels, methods]),
                                    columns = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "accuracy": 0, "precision": 0, "recall": 0,
                            "f1": 0, "auc": 0}.keys())
    folds = list(range(10))
    threshold_final = pd.DataFrame(index=pd.MultiIndex.from_product([behavior_labels, methods, folds]),
                                    columns = {"support_thred": 0, "jd_diff_thred": 0}.keys())
    for behavior in behavior_labels:
        for method in methods:
            y_test_total = []
            y_pred_total = []

            for fold in range(10):
                # snaphints_dir = '../Datasets/data/SnapHintsData/submitted/'\
                # snaphints_dir = "/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/submitted/" \
                #              + behavior + "/cv/fold" + str(fold) + "/SnapHints" + method + "Support>0/"

                snaphints_dir =  '../Datasets/data/SnapHintsData/submitted/'\
                                + behavior + "/cv/fold" + str(fold) + "/SnapHintsAllAllFinalSupportOver0/"
                X_all, y_all = get_x_y_snaphints(snaphints_dir, "train")
                X_test, y_test = get_x_y_snaphints(snaphints_dir, "test")
                simple_thres_grid = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                max_f1_simple = 0
                best_thres_simple = -1
                for s_thres in simple_thres_grid:
                    f1_simple = get_f1_with_thres(X_all, y_all, s_thres)
                    print("f1, thres: ", f1_simple, s_thres)
                    if f1_simple > max_f1_simple:
                        max_f1_simple = f1_simple
                        best_thres_simple = s_thres
                        print("f1, thres: ", f1_simple, s_thres)
                    elif f1_simple < max_f1_simple:
                        print("no need to do it")
                        break

                if method != "AndAll":
                    threshold_final.at[(behavior, method, fold), "support_thred"] = best_thres_simple
                    y_pred_here = get_simple_pred_with_thres(X_all, y_all, X_test, best_thres_simple)
                    y_test_total.extend(y_test)
                    y_pred_total.extend(y_pred_here)
                    performance_temp = svm_linear.get_matrix(np.array(y_test_total), np.array(y_pred_total))

                elif method == "AndAll":
                    X_all, X_test = get_simple_data_subset(X_all, X_test, best_thres_simple)
                    print("x all shape:", X_all.shape)
                    print("here is for conjunction")
                    thres_grid = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
                    # thres_grid = [0.1,0.9]
                    max_f1 = 0
                    best_thres = -1
                    for thres in thres_grid:
                        f1 = get_f1_with_thres(X_all, y_all, thres, simple = False)
                        print("f1, thres: ", f1, thres)
                        if f1 > max_f1:
                            max_f1 = f1
                            best_thres = thres
                            print("f1, thres: ", f1, thres)
                        elif f1 < max_f1:
                            print("no need to do it")
                            break
                    threshold_final.at[(behavior, method, fold), "jd_diff_thred"] = best_thres
                    save_obj(threshold_final, "NestedCVThresPQGrams", ".")

                    y_pred_here = get_pred_with_thres(X_all, y_all, X_test, best_thres)
                    y_test_total.extend(y_test)
                    y_pred_total.extend(y_pred_here)
                    print("----------------performance_temp-------------------")
                    performance_temp = svm_linear.get_matrix(np.array(y_test_total), np.array(y_pred_total))


            y_pred_total = np.array(y_pred_total)
            y_test_total = np.array(y_test_total)
            print(y_pred_total.shape)
            performance = svm_linear.get_matrix(y_test_total, y_pred_total)
            performance = round_return(performance)
            behavior_results.loc[(behavior, method)] = performance
            print(behavior_results)

            save_obj(behavior_results, "NestedCVResultPQGrams", ".")
    return behavior_results




def get_f1_with_thres(X_all, y_all, thres, simple = True):
    y_val_total = []
    y_pred_total = []
    split_strategy = StratifiedKFold(10)

    for train_index, val_index in split_strategy.split(X_all, y_all):
        X_train, X_val = X_all[train_index], X_all[val_index]
        y_train, y_val = y_all[train_index], y_all[val_index]
        if not simple:
            y_pred = get_pred_with_thres(X_train, y_train, X_val, thres)
        else:
            y_pred = get_simple_pred_with_thres(X_train, y_train, X_val, thres)

        y_pred_total.extend(y_pred)
        y_val_total.extend(y_val)

    y_pred_total = np.array(y_pred_total)
    y_val_total = np.array(y_val_total)
    print(y_pred_total.shape)
    performance = svm_linear.get_matrix(y_val_total, y_pred_total)
    performance = round_return(performance)

    return performance['f1']




def get_pred_with_thres(X_train, y_train, X_test, thres):
    feature_dim = X_train.shape[1]
    new_conjunction_list = []
    for i in tqdm(range(feature_dim)):
        for j in range(i + 1, feature_dim):
            jd_yes = get_jd(X_train, y_train, i, j, 1)
            jd_no = get_jd(X_train, y_train, i, j, 0)
            if jd_yes - jd_no >= thres:
                new_conjunction_list.append((i, j))
    print("start appending")
    for conjunction_feature_tuple in tqdm(new_conjunction_list):
        X_train = add_conjunction(X_train, conjunction_feature_tuple)
        X_test = add_conjunction(X_test, conjunction_feature_tuple)
    y_pred = svm_linear.get_y_pred(X_train, X_test, y_train)
    return y_pred



def get_simple_pred_with_thres(X_train, y_train, X_test, thres):
    feature_dim = X_train.shape[1]
    kept = []
    for i in tqdm(range(feature_dim)):
        support = np.sum(X_train[:,i])/X_train.shape[0]
        # print(support)
        if support > thres:
            kept.append(i)
    x_orig = copy.deepcopy(X_train)
    X_tr = x_orig[:,kept]
    print("len kept: ", len(kept))

    x_orig_test = copy.deepcopy(X_test)
    X_te = x_orig_test[:,kept]

    # X_test = add_conjunction(X_test, conjunction_feature_tuple)
    y_pred = svm_linear.get_y_pred(X_tr, X_te, y_train)
    return y_pred

def get_simple_data_subset(X_train, X_test, thres):
    feature_dim = X_train.shape[1]
    kept = []
    for i in tqdm(range(feature_dim)):
        support = np.sum(X_train[:,i])/X_train.shape[0]
        if support > thres:
            kept.append(i)

    print("kept length", len(kept))
    x_orig = copy.deepcopy(X_train)
    X_tr = x_orig[:,kept]

    x_orig_test = copy.deepcopy(X_test)
    X_te = x_orig_test[:,kept]

    return X_tr, X_te


def get_jd(x, y_data, i, j, yes_or_no):
    subset_index = np.where(y_data == yes_or_no)
    x_subset = x[subset_index]
    union = 0
    intersection = 0
    for d in range(len(x_subset)):
        if x_subset[d, i] == 1 or x_subset[d, j] == 1:
            union += 1
            if x_subset[d, i] == 1 and x_subset[d, j] == 1:
                intersection += 1
    if union == 0:
        return 0
    else:
        return intersection/union


def add_conjunction(X_train, conjunction_feature_tuple):
    new_col = [0]*len(X_train)
    i, j = conjunction_feature_tuple[0], conjunction_feature_tuple[1]
    for d in range(len(X_train)):
        if X_train[d, i] == 1 and X_train[d, j] == 1:
            new_col[d] = 1
    new_col = np.asarray(new_col).reshape(len(X_train),1)

    return np.hstack((X_train, new_col))

main()






