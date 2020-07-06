import sys
sys.path.append('.')
from evaluate_snaphints import *
game_label_csv = pd.read_csv(submission_dir + "/game_label_413.csv")
game_label_csv = game_label_csv.sort_values(by = ['pid'])
print(game_label_csv)

fold_count = 5
start_index = 0

for i in range(fold_count):
    end_index = start_index + 413//fold_count+1
    test_index = list(range(start_index, end_index))
    start_index = end_index

