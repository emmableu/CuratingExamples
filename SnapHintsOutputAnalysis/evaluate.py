import sys
sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets')
from Dataset import *



behavior_labels = ["keymove", "jump", "cochangescore", "movetomouse", "moveanimate"]

def get_yes_no(data, yes_no_group):
    yes_x = data[yes_no_group].tolist()
    yes_x = [list(eval(item)) for item in yes_x]
    yes_x = np.array(yes_x).transpose()
    return yes_x

def get_x_y_snaphints(snaphints_dir, partition):
    data = pd.read_csv(snaphints_dir + partition + ".csv", index_col = 0)
    yes_x = get_yes_no(data, "yes")
    no_x = get_yes_no(data, "no")
    # print(yes_x, no_x)
    x = np.vstack((yes_x, no_x))
    y = np.hstack((np.array([1]*yes_x.shape[0]), np.array([0]*no_x.shape[0])))
    # print(x, y)
    return x, y






def snaphints_crossvalidation():
    behavior_results = pd.DataFrame(index = behavior_labels, columns = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "accuracy": 0, "precision": 0, "recall": 0,
                            "f1": 0, "auc": 0}.keys())
    for behavior in behavior_labels:
        performance_temp = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "accuracy": 0, "precision": 0, "recall": 0,
                            "f1": 0, "auc": 0}
        for fold in range(10):
            snaphints_dir = "/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/submitted/" \
                            + behavior + "/cv/fold" + str(fold) + "/SnapHintsTrainTest/"
            X_train, y_train = get_x_y_snaphints(snaphints_dir, "train")
            X_test, y_test = get_x_y_snaphints(snaphints_dir, "test")
            add_performance = svm_linear.get_performance(X_train, X_test, y_train, y_test)
            performance_temp = add_by_ele(performance_temp, add_performance)

        performance = get_dict_average(performance_temp, cv_total=10)
        performance = round_return(performance)
        behavior_results.loc[behavior] = performance
        print(behavior_results)


    save_obj(behavior_results, "svm_behaviors", root_dir, "SnapHintsOutputAnalysis")



snaphints_crossvalidation()
