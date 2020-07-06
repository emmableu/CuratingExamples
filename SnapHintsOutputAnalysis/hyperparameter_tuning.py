import sys
sys.path.append('.')
from evaluate_snaphints import *
pq_rules = pd.read_csv("/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/csedm20/CRV/pqRules.csv")
game_label_data = pd.read_csv("/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/game_label_413.csv")

# conjunction_rules = pd.read_csv("/Users/wwang33/Documents/SnapHints/data/csc110/fall2019project1/csedm20/CRV/ConjunctionRules.csv")

print(pq_rules.head())

pq_rule_category = ["Train", "TrainVal"]

conjunction_rule_category = ["TrainSupport", "TrainjdPos", "TrainjdNeg", "TrainValSupport", "TrainValjdPos", "TrainValjdNeg"]

for label in behavior_labels:
    for fold in range(5):
        for category in pq_rule_category:
            col_id = label + str(fold) + category
            print(col_id)


col_id = "keymove0TrainVal"
fold_seed = []
this_list = []
for x in range(413):
    if x%82 == 0 and len(this_list)>0:
        if x//82 ==5:
            this_list.extend([410, 411, 412])
        fold_seed.append(this_list)
        this_list = []
    this_list.append(x)


for i in range(5):
    test = fold_seed[i]
    val = fold_seed[(i+1)%5]
    train = fold_seed[(i+2)%5] + fold_seed[(i+3)%5] + fold_seed[(i+4)%5]
    print(len(test), len(val), len(train))


keymove = game_label_data["keymove"].to_numpy()


for i in pq_rules.index:
    if pq_rules.at[i, col_id] < 0.3:
        continue
    else:
        x.append(pq_rules.at[i, "snapshotVector"])


