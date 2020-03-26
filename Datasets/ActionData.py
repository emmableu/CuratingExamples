import sys
sys.path.append("/Users/wwang33/Documents/IJAIED20/CuratingExamples/my_module")
from Test import *
import pandas as pd
from save_load_pickle import *
root_dir = "/Users/wwang33/Documents/IJAIED20/CuratingExamples/"
from tqdm import tqdm
from alms_helper import *
class ActionData:
    def __init__(self, code_state, game_label , action_name, selected_p_q_list):
        self.code_state = code_state
        self.game_label = game_label
        self.action_name = action_name
        self.selected_p_q_list = selected_p_q_list
        self.pid_list = load_obj('pid', base_dir, "")

    def memory_get_code_shape_from_pid(self,pid):
        code_shape_dict = {}
        if base_dir.split("/")[-2] == 'ScratchASTData':
            code_shape_dict.update(self.code_state.at[(pid)])
        else:
            # print(self.code_state.loc['329266361'])
            for p_q in self.selected_p_q_list:
                try:
                    code_shape_dict.update(self.code_state.at[str(pid), 'code_state'+ str(p_q)])
                except KeyError:
                    print(pid, "keyerror!")
        # start = time.time()
        # code_shape_dict.update(self.code_state.at[str(pid), 'code_state[1, 3]'])
        # end = time.time()
        # print("Time elapsed for: " + inspect.stack()[0][3] + " is: ", end - start, " seconds")


        return code_shape_dict



    def memory_get_code_shape_from_pid_temp(self,pid):
        code_shape_dict = {}
        # print(self.code_state.loc['329266361'])
        x = np.random.randint(3, size=100*10000).reshape(100, 10000)
        y = np.random.randint(2, size = 100).reshape(100, 1)

        for p_q in self.selected_p_q_list:
            try:
                code_shape_dict.update(self.code_state.at[str(pid), 'code_state'+ str(p_q)])
            except KeyError:
                print(pid, "keyerror!")
        # start = time.time()
        # code_shape_dict.update(self.code_state.at[str(pid), 'code_state[1, 3]'])
        # end = time.time()
        # print("Time elapsed for: " + inspect.stack()[0][3] + " is: ", end - start, " seconds")


        return code_shape_dict



    def __get_pattern_df(self, pattern, train_pid):

        pool = self.game_label[self.game_label.pid.isin(train_pid)].reset_index(drop = True)
        pattern_df = pd.DataFrame(columns = ['pid', 'occurance', 'label'])
        for i in pool.index:
            pid = str(pool.at[i, 'pid'])
            code_shape = self.memory_get_code_shape_from_pid(pid)
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
            pattern_df.loc[len(pattern_df)] = new_row
        return pattern_df


    def get_pattern_key_from_pid(self, train_pid):
        # code_state = load_obj( "code_state|0|414", self.root_dir+"Datasets/data/SnapASTData", "game_labels_" + str(415) + "/code_state" + str(self.code_shape_p_q_list) )
        # pool = self.data
        pattern_set = set()
        for i in (train_pid):
            code_shape = {}
            for pqlist in self.selected_p_q_list:
                code_shape.update(self.code_state.at[i, "code_state" + str(pqlist)])
            print("len(code_shape.keys()): ", len(code_shape.keys()))
            new_pattern_s = code_shape
            pattern_set = atomic_add(pattern_set, new_pattern_s)
        return pattern_set



    def get_pattern_statistics(self, train_pid, baseline = True):

        if baseline:
            pattern_set = load_obj("pattern_set" + str(self.selected_p_q_list), base_dir,
                                   "CodeState")
            # print("pattern_set", pattern_set.keys())
            full_pattern_set = set()
            print(base_dir.split("/"))
            if base_dir.split("/")[-2] == 'ScratchASTData':
                atomic_add(full_pattern_set, pattern_set)
            else:
                for p in (self.selected_p_q_list):
                    atomic_add(full_pattern_set, pattern_set['code_state' + str(p)])

            full_pattern_set = sorted(full_pattern_set)
            print(full_pattern_set)
            save_obj(full_pattern_set, "full_patterns", base_dir, "xy_0heldout/code_state" + str(self.selected_p_q_list))
            return full_pattern_set

        else:
            pattern_set = self.get_pattern_key_from_pid(train_pid)
        significant_patterns = []
        for pattern in tqdm(pattern_set):
            # print(pattern)
            pattern_df = self.__get_pattern_df(pattern, train_pid)
            test = Test(pattern_df)

            if test.freq_compare_test() == "discard":
                continue
            elif test.freq_compare_test() or test.chi_square_test() or test.kruskal_wallis_test():
                significant_patterns.append(pattern)
        print(len(significant_patterns))
        print(significant_patterns)
        return significant_patterns


    def submission_get_pattern_statistics(self, train_pid, baseline):
        self.get_pattern_statistics(train_pid, baseline)


    def save_x_y_train_test(self,train_pid, test_pid, x_save_dir, save_dir,reduce_size = True, baseline = True):
        # significant_patterns = self.get_pattern_statistics(train_pid, baseline)
        significant_patterns = load_obj( "full_patterns", base_dir, "xy_0heldout/code_state" + str(self.selected_p_q_list))
        num_patterns = len(significant_patterns)
        train_df = self.game_label[self.game_label.pid.isin(train_pid)].reset_index(drop = True)
        if not test_pid:
            pass
        else:
            test_df = self.game_label[self.game_label.pid.isin(test_pid)].reset_index(drop = True)
        # print("train_df: ", train_df)
        def get_x(df):
            x = np.empty((len(df.index), num_patterns))
            for game_index, i in enumerate(df.index):
                pid = df.at[i, 'pid']
                code_shape = self.memory_get_code_shape_from_pid(pid)
                for pattern_index, p in enumerate(significant_patterns):
                    if p in code_shape.keys():
                        occurance = code_shape[p]
                        # print(occurance)
                    else:
                        occurance = 0
                    x[game_index, pattern_index] = occurance
            if reduce_size == True:
                keep = jaccard_select(list(range(len(significant_patterns))), x)
                save_obj(keep, 'jaccard_reduced_patterns', x_save_dir)
                x = x[:, keep]
            return x

        def get_y(df):
            y = np.zeros(len(df.index))
            for game_index, i in enumerate(df.index):
                if df.at[i, self.action_name]:
                    y[game_index] = 1
                else:
                    y[game_index] = 0
            return y

        if is_obj( 'jaccard_reduced_patterns', x_save_dir):
            pass
        else:
            X_train = get_x(train_df)
            # X_test = get_x(train_df)
            save_obj(X_train, "X_train", x_save_dir, "")
            # save_obj(X_test, "X_test_reduced", x_save_dir, "")

        y_train = get_y(train_df)
        # y_test = get_y(test_df)
        save_obj(y_train, "y_train", save_dir, "")
        # save_obj(y_test, "y_test", x_save_dir, "")










    def save_x_y_train_test_temp(self,train_pid, save_dir,baseline = True):
        significant_patterns = self.get_pattern_statistics_temp(train_pid, baseline)
        save_obj(significant_patterns, "significant_patterns", save_dir, "")





    def get_pattern_statistics_temp(self, train_pid, baseline):

        if baseline:
            pattern_set = load_obj("pattern_set", base_dir,
                                    "code_state" + str([[1, 0], [1, 1], [1, 2], [1, 3]]))
            print("pattern_set", pattern_set.keys())

            full_pattern_set = set()
            for p in (self.selected_p_q_list):
                atomic_add(full_pattern_set, pattern_set['code_state' + str(p)])
            return full_pattern_set

        else:
            pattern_set = self.get_pattern_key_from_pid(train_pid)
        significant_patterns = []
        start_start = time.time()
        count = 0
        for pattern in tqdm(pattern_set):
            count += 1
            if count == 30:
                break
            # print(pattern)
            start = time.time()
            pattern_df = self.__get_pattern_df_temp(pattern, train_pid)
            end = time.time()
            print("time it takes for pattern_df = self.__get_pattern_df(pattern, train_pid):  ", end-start)
        #     test = Test(pattern_df)
        #
        #     if test.freq_compare_test() == "discard":
        #         continue
        #     elif test.freq_compare_test() or test.chi_square_test() or test.kruskal_wallis_test():
        #         significant_patterns.append(pattern)
        # end_end = time.time()
        # print("for pattern in tqdm(pattern_set[:30]): loop takes time: ", end_end-start_start)
        # print(len(significant_patterns))
        # print(significant_patterns)
        return 0


    def __get_pattern_df_temp(self, pattern, train_pid):

        pool = self.game_label[self.game_label.pid.isin(train_pid)].reset_index(drop = True)

        # pattern_df = pd.DataFrame(columns = ['pid', 'occurance', 'label'])

        x = np.random.randint(3, size=100 * 10000).reshape(100, 10000)
        # y = np.random.randint(2, size=100).reshape(100, 1)


        return x[:4]
