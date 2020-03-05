import sys
sys.path.append("/Users/wwang33/Documents/IJAIED20/CuratingExamples/my_module")
from Test import *
import pandas as pd

class ActionData:
    def __init__(self, code_state, game_label , action_name):
        self.code_state = code_state
        self.game_label = game_label
        self.action_name = action_name

    def __get_code_shape_from_pid(self, pid):
        for i in self.code_state.index:
            if self.code_state.at[i, 'pid'] == pid:
                return self.code_state.at[i, 'codeshape_count_dict']

    def get_yes_patterns(self):
        self.pattern_set = set()
        for i in self.game_label.index:
            if self.game_label.at[i, 'good'] and self.game_label.at[i, 'good'] == True:
                if self.game_label.at[i, self.action_name] == True:
                    pid = self.game_label.at[i, 'pid']
                    code_shape = self.__get_code_shape_from_pid(pid)
                    new_pattern_s = code_shape.keys()
                    self.__atomic_add(new_pattern_s)

    def __atomic_add(self, new_pattern_s):
        for pattern in new_pattern_s:
            self.pattern_set.add(pattern)


    def __get_pattern_df(self, pattern):
        pattern_df = pd.DataFrame(columns = ['pid', 'occurance', 'label'])
        for i in self.game_label.index:
            pid = self.game_label.at[i, 'pid']
            code_shape = self.__get_code_shape_from_pid(pid)
            try:
                occurance = code_shape[pattern]
            except KeyError:
                occurance = 0
            # print(occurance)
            if self.game_label.at[i, self.action_name]:
                label = 'yes'
            else:
                label = 'no'
            new_row = {'pid': pid, 'occurance': occurance, 'label': label}
            # print("new_row" , new_row)
            # print("pattern_df",pattern_df)
            pattern_df.loc[len(pattern_df)] = new_row
        return pattern_df

    def get_pattern_statistics(self):
        significant_patterns = []
        for pattern in self.pattern_set:
            pattern_df = self.__get_pattern_df(pattern)
            test = Test(pattern_df)

            if test.freq_compare_test() == "discard":
                continue
            elif test.freq_compare_test() or test.chi_square_test() or test.kruskal_wallis_test():
                significant_patterns.append(pattern)
        print(len(significant_patterns))
        print(significant_patterns)
        return significant_patterns


    def get_xy(self):
        total = len(self.game_label.index)
        significant_patterns = self.get_pattern_statistics()
        num_patterns = len(significant_patterns)
        x = np.zeros((total, num_patterns))
        y = np.zeros(total)
        for game_index, i in enumerate(self.game_label.index):
            pid = self.game_label.at[i, 'pid']
            code_shape = self.__get_code_shape_from_pid(pid)
            for pattern_index, p in enumerate(significant_patterns):
                try:
                    occurance = code_shape[p]
                except KeyError:
                    occurance = 0
                x[game_index][pattern_index] = occurance
            if self.game_label.at[i, self.action_name]:
                y[game_index] = 1
            else:
                y[game_index] = 0



        return x,y











