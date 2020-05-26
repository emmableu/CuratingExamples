import sys
sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets')
from Dataset import *



# behavior_labels = ["keymove", "jump", "cochangescore", "movetomouse", "moveanimate", "costopall"]
behavior_labels = ["cochangescore"]
# behavior_labels = ["costopall", "movetomouse",  "jump", "cochangescore","keymove"]

def get_yes_no(data, yes_no_group):
    yes_x = data[yes_no_group].tolist()
    yes_x = [list(eval(item)) for item in yes_x]
    yes_x = np.array(yes_x).transpose()
    return yes_x

def get_x_y_train_snaphints(snaphints_dir, select = True):
    data = pd.read_csv(snaphints_dir + "train.csv", index_col = 0)
    yes_x = get_yes_no(data, "yes")
    no_x = get_yes_no(data, "no")
    x = np.vstack((yes_x, no_x))
    y = np.hstack((np.array([1]*yes_x.shape[0]), np.array([0]*no_x.shape[0])))
    if select:
        selected_features = get_selected_feature_index(snaphints_dir, 0.3, 0.3, 0.8)
        # selected_features = get_selected_feature_index(snaphints_dir, jd_diff, jd_yes, confidence)
        x = x[:,selected_features]
    return x, y


def get_x_y_test_snaphints(snaphints_dir, selected_features):
    data = pd.read_csv(snaphints_dir + "test.csv", index_col = 0)
    yes_x = get_yes_no(data, "yes")
    no_x = get_yes_no(data, "no")
    x = np.vstack((yes_x, no_x))
    y = np.hstack((np.array([1]*yes_x.shape[0]), np.array([0]*no_x.shape[0])))
    x = x[:,selected_features]
    return x, y



def get_selected_feature_index(snaphints_dir, jd_diff, jd_yes_bar, confidence_bar):
    features = pd.read_csv(snaphints_dir + "/features.csv")
    selected_features = []
    for fid in features.index:
        jd_yes = features.at[fid, 'jdYes']
        jd_no = features.at[fid, 'jdNo']
        confidence = features.at[fid, 'confidenceYes']
        if jd_yes - jd_no >= jd_diff:
            if jd_yes >= jd_yes_bar:
                selected_features.append(fid)
            elif confidence >= confidence_bar:
                selected_features.append(fid)
        elif jd_yes == -1:
            selected_features.append(fid)
    return np.array(selected_features)













# methods = ['Neighbor', 'AndAllFull', 'AndAll','DPM', 'All', 'And']
methods = ['AndAllComplete']



def snaphints_crossvalidation():
    behavior_results = pd.DataFrame(index=pd.MultiIndex.from_product([behavior_labels, methods]),
                                    columns = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "accuracy": 0, "precision": 0, "recall": 0,
                            "f1": 0, "auc": 0}.keys())
    for behavior in behavior_labels:
        for method in methods:
            y_test_total = []
            y_pred_total = []
            # try:
            for fold in range(10):
                snaphints_dir = "/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/submitted/" \
                                + behavior + "/cv/fold" + str(fold) + "/SnapHints" + method + "/"
                X_train, y_train = get_x_y_snaphints(snaphints_dir, "train")
                X_test, y_test = get_x_y_snaphints(snaphints_dir, "test")
                y_test_total.extend(y_test)
                y_pred  = lr.get_y_pred(X_train,  X_test, y_train)
                print(y_pred)
                for i, y in enumerate(y_pred):
                    if y_pred[i] != y_test[i]:
                        print(i - Counter(y_test)[1]+1)
                y_pred_total.extend(y_pred)
                performance_temp = svm_linear.get_matrix(np.array(y_test_total), np.array(y_pred_total))
            # except:
            #     continue
                # print(performance_temp)
            y_pred_total = np.array(y_pred_total)
            y_test_total = np.array(y_test_total)
            # print(y_pred_total, y_test_total)
            performance = svm_linear.get_matrix(y_test_total, y_pred_total)
            performance = round_return(performance)
            # performance['name']
            behavior_results.loc[(behavior, method)] = performance
            print(behavior_results)


    save_obj(behavior_results, "svm_behaviors4", root_dir, "SnapHintsOutputAnalysis")



snaphints_crossvalidation()


# sklearn.feature_selection.f_regression(X, y, center=True)

#
#
# snaphints_dir = "/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/submitted/" \
#                 + behavior + "/cv/fold" + str(fold) + "/SnapHintsArchive/"
#
# X_train, y_train = get_x_y_snaphints(snaphints_dir, "train")
# X_test, y_test = get_x_y_snaphints(snaphints_dir, "test")
# y_pred = svm_linear.get_y_pred(X_train, X_test, y_train)
# performance_temp = svm_linear.get_matrix(np.array(y_test), np.array(y_pred
#                                                                     ))


