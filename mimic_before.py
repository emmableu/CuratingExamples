import sys

sys.path.append('.')
import operator
from evaluate_snaphints import *
np.set_printoptions(threshold=sys.maxsize)

pq_rules = pd.read_csv("/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/csedm20/CRV/pqRules.csv")
old_features = pd.read_csv("/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/submitted/costopall/cv/fold0/SnapHintsAll/features.csv")
pid_output = pd.read_csv("/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/submitted/costopall/cv/fold0/SnapHintsAll/pid_output.csv", index_col = 0)
# train_pos_list = pid_output.at[]
train_pos_list = list(eval(pid_output.at["train", "yes"]))
train_neg_list = list(eval(pid_output.at["train", "no"]))
game_label_data = pd.read_csv("/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/game_label_413.csv")
pid_list = game_label_data['pid'].to_list()
features = old_features['name'].to_list()

features = [feature.split(": ", 1)[1] for feature in features]
print(features)

train_pos_index = [pid_list.index(d) for d in train_pos_list]
train_neg_index = [pid_list.index(d) for d in train_neg_list]
print(train_pos_index)

pq_rule_features = [feature.split(": ", 1)[1] for feature in pq_rules["grams"].to_list()]
# for i in pq_rules.index:
#     f = pq_rules.at[i, "grams"]
#     f = f.split(": ", 1)[1]
#     if f in features:
#         print(f)
#         feature_x = list(eval(pq_rules.at[i, "snapshotVector"]))
#         feature_x = np.asarray(feature_x)
#         print(feature_x[train_pos_index])

for f in features:
    if f in pq_rule_features:
        print(f)
        i = pq_rule_features.index(f)
        feature_x = list(eval(pq_rules.at[i, "snapshotVector"]))
        feature_x = np.asarray(feature_x)
        print(feature_x[train_neg_index])


        # break

# print(game_label_data.loc[281])

