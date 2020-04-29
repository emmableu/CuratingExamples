from preprocessors.PreProcessor import PreProcessor
from alms_helper import *

def select_feature(x, y, jaccard):
    x = np.digitize(x, bins=[1])

    y = y.astype(int)
    y_yes_index = np.where(y == 1)[0]
    yes_x = x[y_yes_index]
    # print('y_yes_index:', y_yes_index)
    y_no_index = np.where(y == 0)[0]
    no_x = x[y_no_index]
    feature_freq1 =  np.mean(yes_x, axis = 0)
    feature_freq2 =  np.mean(no_x, axis = 0)
    feature_sd1 = np.std(yes_x, axis=0)
    feature_sd2 = np.std(no_x, axis=0)
    # print(feature_freq1)
    n1 = len(yes_x)
    n2 = len(no_x)
    print("n1, n2", n1, n2)
    selected_patterns = []

    for i in tqdm(range(len(x[0]))):
        z,p = twoSampZ(feature_freq1[i], feature_freq2[i], feature_sd1[i], feature_sd2[i], n1, n2)
        # print(p)
        if p<0.05 and z>0:
            selected_patterns.append(i)
    print("pattern selected with length:" ,len(selected_patterns))


    if not jaccard:
        # return np.array([92, 93])
        return selected_patterns
    else:
        keep = jaccard_select(selected_patterns, x)
        print("jaccard similarity returns feature with length: ", len(keep))
        return keep





class DPMPreProcessor(PreProcessor):
    def __init__(self, one_hot = True):
        super().__init__()
        self.one_hot = one_hot

    def preprocess(self, X, y, X_test):
        selected_features = select_feature(X, y, jaccard=False)
        self.print_selected_features(selected_features)
        return X[:, selected_features], X_test[:, selected_features]

    def print_selected_features(self, selected_features):
        if not self.one_hot:
            return None
        patterns = load_obj("full_patterns", base_dir + "/xy_0heldout/code_state[[1, 0]]")
        patterns = np.array(patterns)
        selected_features_print  = patterns[selected_features]
        print(selected_features_print)





