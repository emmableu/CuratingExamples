import sys
sys.path.append("/home/wwang33/IJAIED20/CuratingExamples/my_module")
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

    def get_yes_patterns(self, train_pid):
        pool = self.game_label[self.game_label.pid.isin(train_pid)].reset_index(drop = True)
        self.pattern_set = set()
        for i in pool.index:
            if pool.at[i, 'good'] and pool.at[i, 'good'] == True:
                if pool.at[i, self.action_name] == True:
                    pid = pool.at[i, 'pid']
                    code_shape = self.__get_code_shape_from_pid(pid)
                    new_pattern_s = code_shape.keys()
                    self.__atomic_add(new_pattern_s)

    def __atomic_add(self, new_pattern_s):
        for pattern in new_pattern_s:
            self.pattern_set.add(pattern)


    def __get_pattern_df(self, pattern, train_pid):
        pool = self.game_label[self.game_label.pid.isin(train_pid)].reset_index(drop = True)

        pattern_df = pd.DataFrame(columns = ['pid', 'occurance', 'label'])
        for i in pool.index:
            pid = pool.at[i, 'pid']
            code_shape = self.__get_code_shape_from_pid(pid)
            try:
                occurance = code_shape[pattern]
            except KeyError:
                occurance = 0
            # print(occurance)
            if pool.at[i, self.action_name]:
                label = 'yes'
            else:
                label = 'no'
            new_row = {'pid': pid, 'occurance': occurance, 'label': label}
            # print("new_row" , new_row)
            # print("pattern_df",pattern_df)
            pattern_df.loc[len(pattern_df)] = new_row
        return pattern_df

    def get_pattern_statistics(self, train_pid):
        self.get_yes_patterns(train_pid)
        significant_patterns = []
        for pattern in self.pattern_set:
            pattern_df = self.__get_pattern_df(pattern, train_pid)
            test = Test(pattern_df)

            if test.freq_compare_test() == "discard":
                continue
            elif test.freq_compare_test() or test.chi_square_test() or test.kruskal_wallis_test():
                significant_patterns.append(pattern)
        print(len(significant_patterns))
        print(significant_patterns)
        return significant_patterns


    def get_xy(self,train_pid, test_pid):
        significant_patterns = self.get_pattern_statistics(train_pid)
        num_patterns = len(significant_patterns)

        train_df = self.game_label[self.game_label.pid.isin(train_pid)].reset_index(drop = True)
        test_df = self.game_label[self.game_label.pid.isin(test_pid)].reset_index(drop = True)

        def get_xy(df):
            x = np.zeros((len(df.index), num_patterns))
            y = np.zeros(len(df.index))
            for game_index, i in enumerate(df.index):
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
            return x, y

        X_train, y_train = get_xy(train_df)
        X_test, y_test = get_xy(test_df)

        return X_train, X_test, y_train, y_test



        return x,y











