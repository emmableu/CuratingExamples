import numpy as np
import pandas as pd
import pickle
from scipy import stats

from importlib.machinery import SourceFileLoader
GetSequence = SourceFileLoader("get-sequence.py", "model/get-sequence.py").load_module()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
import re
import time

pd.options.display.max_columns = 30

innerdict_colnames = ['n', 'cur_count', 's_support', 'sum', 'sum_squared', 'mean', 'variance']

get_all_patterns(pattern_type, accumulation_type)



def combine_codemap(n, codemap, accu_code_map):
    # print('in combine, accu_code_map:', accu_code_map)
    for pattern, value in codemap.items():
        if pattern in accu_code_map:
            cur_count, s_support, sum, sum_squared, mean, variance = accu_code_map[pattern][1:]
            cur_count = codemap[pattern]
            s_support += 1
            sum += cur_count
            sum_squared += cur_count**2
            mean = sum/n
            variance = get_variance(n, sum_squared, mean)
        else:
            cur_count = codemap[pattern]
            s_support = 1
            sum = cur_count
            sum_squared = cur_count**2
            mean = sum/n
            variance = get_variance(n, sum_squared, mean)
        innerdict_values = [n, cur_count, s_support, sum, sum_squared, mean, variance]
        accu_code_map[pattern] = innerdict_values
    for pattern, value in accu_code_map.items():
        if pattern not in codemap:
            cur_count, s_support, sum, sum_squared, mean, variance = accu_code_map[pattern][1:]
            cur_count = 0
            mean = sum/n
            variance = get_variance(n, sum_squared, mean)
            innerdict_values = [n, cur_count, s_support, sum, sum_squared, mean, variance]
            accu_code_map[pattern] = innerdict_values
    return accu_code_map


def get_all_patterns(all_data, pattern_type):
    action_name_s = ['keymove', 'jump', 'costopall', 'wrap', 'cochangescore', 'movetomouse', 'moveanimate']
    for action_name in action_name_s:
        start = time.time()
        data = all_data[action_name]['df']
        all_patterns = {}
        accu_code_map = {}
        for i in range(0, len(data.index)):
            pid = data.at[i, 'pid']
            abstract = data.at[i, 'abstract']
            success = data.at[i, 'label']
            if success:
                key = str(pid) + "|" + action_name + '|pass'
            else:
                key = str(pid) + action_name + '|fail'
            codemap = abstract
            all_patterns[key] = accu_code_map
            n = n+1
            accu_code_map = combine_codemap(n, codemap, accu_code_map)
        save_pickle(all_patterns, "AllPatternsStudents", "Datasets/data", "")
        end = time.time()
        print('time for get_all_patterns(): ', (end - start) / 60)



def get_pattern_stats(train_keys):
    filename = 'data/AllPatternsProblem.pkl'
    with open(filename, 'rb') as handle:
        all_patterns_problem = pickle.load(handle)
    success = {}
    failure = {}
    for train_key in train_keys:
        s = train_key.split('|')
        student, problem, yes = s[0], s[1], s[2] == 'pass'
        if yes:
            success[train_key] = all_patterns_problem[train_key]
        else:
            failure[train_key] = all_patterns_problem[train_key]

    pattern_success, pattern_both_success, pattern_failure, pattern_both_failure = pd.DataFrame(columns=colnames), pd.DataFrame(columns=colnames), pd.DataFrame(columns=colnames), pd.DataFrame(columns=colnames)
    for pattern in success.index:
        if pattern not in failure.index:
            failure_s_support, failure_total_student = 0, failure['TotalStudent'][0]
            failure_i_support, failure_i_support_variance = 0, 0
        else:
            failure_s_support = failure.at[pattern, 'SSupport']
            failure_total_student = failure.at[pattern, 'TotalStudent']
            failure_i_support = failure.at[pattern, 'ISupport']
            failure_i_support_variance = failure.at[pattern, 'ISupportVariance']
        success_s_support = success.at[pattern, 'SSupport']
        success_total_student = success.at[pattern, 'TotalStudent']

        if success_s_support < threshold * success_total_student and failure_s_support < threshold* failure_total_student:
            continue
        success_i_support = success.at[pattern, 'ISupport']
        success_i_support_variance = success.at[pattern, 'ISupportVariance']

        pvalue = t_test(success_i_support, failure_i_support,success_i_support_variance, failure_i_support_variance,
                  success_total_student, failure_total_student)[2]
        if pvalue > 0.05:
            continue


        newrow = {"Pattern": pattern, 'SuccessSSupport': success_s_support,
                  'SuccessISupport': success_i_support,
                  'FailureSSupport': failure_s_support, 'FailureISupport': failure_i_support,
                  'SuccessTotalStudent': success_total_student, 'FailureTotalStudent': failure_total_student}
        if success_s_support >=  threshold* success_total_student and failure_s_support >=threshold* failure_total_student:
            if success_i_support > failure_i_support:
                pattern_both_success = pattern_both_success.append(newrow, ignore_index=True)
            else:
                pattern_both_failure = pattern_both_failure.append(newrow, ignore_index=True)
        elif success_s_support >=threshold* success_total_student:
            pattern_success = pattern_success.append(newrow, ignore_index=True)
        else:
            pattern_failure = pattern_failure.append(newrow, ignore_index=True)

    for pattern in failure.index:
        if pattern in success.index:
            continue
        success_s_support, success_total_student = 0, success['TotalStudent'][0]
        success_i_support, success_i_support_variance = 0, 0

        failure_s_support = failure.at[pattern, 'SSupport']
        failure_total_student = failure.at[pattern, 'TotalStudent']

        if success_s_support < threshold * success_total_student and failure_s_support <threshold* failure_total_student:
            continue
        pvalue = t_test(success_i_support, failure_i_support, success_i_support_variance, failure_i_support_variance,
                        success_total_student, failure_total_student)[2]
        if pvalue > 0.05:
            continue

        newrow = {"Pattern": pattern, 'SuccessSSupport': success_s_support,
                  'SuccessISupport': success_i_support,
                  'FailureSSupport': failure_s_support, 'FailureISupport': failure_i_support,
                  'SuccessTotalStudent': success_total_student, 'FailureTotalStudent': failure_total_student}

        pattern_failure = pattern_failure.append(newrow, ignore_index=True)

    pattern_success.to_csv("result/featured-patterns-" + pattern_type + "/problem" + str(cur_problem)+ "/pattern-success.csv")
    pattern_both_success.to_csv("result/featured-patterns-" + pattern_type + "/problem" + str(cur_problem)+"/pattern-both-success.csv")
    pattern_failure.to_csv("result/featured-patterns-" + pattern_type + "/problem"+ str(cur_problem)+"/pattern-failure.csv")
    pattern_both_failure.to_csv("result/featured-patterns-" + pattern_type + "/problem"+ str(cur_problem)+"/pattern-both-failure.csv")

    # i = i + 1
    end = time.time()
    print((end-start)/60)
    return pattern_success, pattern_both_success, pattern_failure, pattern_both_failure




def runme():
    data = pd.read_csv("data/Predict.csv", error_bad_lines=False)
    data = data.sort_values(['SubjectID', 'ProblemID'], ascending=[1, 1]).reset_index(drop=True)
    data_group = data
    cur_problem = 4
    threshold = 0.3
    groups = ['success', 'failure']
    pattern_type = '3grams'
    for groupname in groups:
        get_student_pattern_per_problem(data_group, cur_problem, groupname, pattern_type)
        sequence_mining(cur_problem, groupname)
    get_featured_patterns(cur_problem, threshold, pattern_type)

def get_student_pattern_per_problem(action_name):
    student_pattern_per_problem = self.action[action_name]["df"]
    return student_pattern_per_problem

def sequence_mining(list_of_code_shape, all_patterns):
    start = time.time()
    student_pattern_per_problem = get_student_pattern_per_problem()
    colnames = ['Pattern','SSupport', 'ISupport', 'ISupportVariance',  'TotalStudent', 'ISupportVarianceTotal', 'ISupportTotal']
    pattern_statistics = pd.DataFrame(columns= colnames).set_index('Pattern')
    for pattern in list_of_code_shape.keys():
        s_support, i_support, i_support_variance, total_student, i_support_variance_total, i_support_total = 0, 0, 0, 0, 0, 0
        for i in range(len(student_pattern_per_problem.index)):
            count = list_of_code_shape[pattern]
            s_support += np.sign(count)
            i_support_total += count
            i_support_variance_total += count ** 2
            i_support = i_support_total/len(student_pattern_per_problem.index)
            i_support_variance = i_support_variance_total/len(student_pattern_per_problem.index)
        pattern_row = {'Pattern': pattern, 'SSupport': s_support, 'ISupport': i_support, 'ISupportVariance': i_support_variance,
                       'TotalStudent': len(student_pattern_per_problem.index), 'ISupportVarianceTotal': i_support_variance_total, 'ISupportTotal': i_support_total}
        pattern_statistics.at[pattern] = (pattern_row)
    end = time.time()
    print((end - start) / 60)
    return pattern_statistics


def t_test(x1, x2, s1, s2, n1, n2):
    import math
    sdelta = math.sqrt((s1**2)/n1 + (s2**2)/n2)
    t = (x1-x2)/sdelta
    df_up = ((s1**2)/n1 + (s2**2)/n2)**2
    df_down = ((s1**2)/n1)**2/(n1-1) +  ((s2**2)/n2)**2/(n2-1)
    df = df_up/df_down
    pvalue = 1 - stats.t.cdf(t, df = df)
    return t, df, pvalue

def get_featured_patterns(cur_problem, threshold, pattern_type):
    start = time.time()
    success_statistics = pd.read_hdf("generated-data/pattern_statistics-per-problem" + str(cur_problem) + "-" + 'success' +".pkl")
    failure_statistics =  pd.read_hdf("generated-data/pattern_statistics-per-problem" + str(cur_problem) + "-" + 'failure' +".pkl")
    colnames = ['Pattern', 'SuccessSSupport', 'SuccessISupport', 'FailureSSupport', 'FailureISupport', 'SuccessTotalStudent', 'FailureTotalStudent']
    pattern_success, pattern_both_success, pattern_failure, pattern_both_failure = pd.DataFrame(columns=colnames), pd.DataFrame(columns=colnames), pd.DataFrame(columns=colnames), pd.DataFrame(columns=colnames)
    for pattern in success_statistics.index:
        if pattern not in failure_statistics.index:
            failure_s_support, failure_total_student = 0, failure_statistics['TotalStudent'][0]
            failure_i_support, failure_i_support_variance = 0, 0
        else:
            failure_s_support = failure_statistics.at[pattern, 'SSupport']
            failure_total_student = failure_statistics.at[pattern, 'TotalStudent']
            failure_i_support = failure_statistics.at[pattern, 'ISupport']
            failure_i_support_variance = failure_statistics.at[pattern, 'ISupportVariance']
        success_s_support = success_statistics.at[pattern, 'SSupport']
        success_total_student = success_statistics.at[pattern, 'TotalStudent']

        if success_s_support < threshold* success_total_student and failure_s_support < threshold* failure_total_student:
            continue
        success_i_support = success_statistics.at[pattern, 'ISupport']
        success_i_support_variance = success_statistics.at[pattern, 'ISupportVariance']

        pvalue = t_test(success_i_support, failure_i_support,success_i_support_variance, failure_i_support_variance,
                  success_total_student, failure_total_student)[2]
        if pvalue > 0.05:
            continue


        newrow = {"Pattern": pattern, 'SuccessSSupport': success_s_support,
                  'SuccessISupport': success_i_support,
                  'FailureSSupport': failure_s_support, 'FailureISupport': failure_i_support,
                  'SuccessTotalStudent': success_total_student, 'FailureTotalStudent': failure_total_student}
        if success_s_support >=  threshold* success_total_student and failure_s_support >=threshold* failure_total_student:
            if success_i_support > failure_i_support:
                pattern_both_success = pattern_both_success.append(newrow, ignore_index=True)
            else:
                pattern_both_failure = pattern_both_failure.append(newrow, ignore_index=True)
        elif success_s_support >=threshold* success_total_student:
            pattern_success = pattern_success.append(newrow, ignore_index=True)
        else:
            pattern_failure = pattern_failure.append(newrow, ignore_index=True)

    for pattern in failure_statistics.index:
        if pattern in success_statistics.index:
            continue
        success_s_support, success_total_student = 0, success_statistics['TotalStudent'][0]
        success_i_support, success_i_support_variance = 0, 0

        failure_s_support = failure_statistics.at[pattern, 'SSupport']
        failure_total_student = failure_statistics.at[pattern, 'TotalStudent']

        if success_s_support < threshold * success_total_student and failure_s_support <threshold* failure_total_student:
            continue
        pvalue = t_test(success_i_support, failure_i_support, success_i_support_variance, failure_i_support_variance,
                        success_total_student, failure_total_student)[2]
        if pvalue > 0.05:
            continue

        newrow = {"Pattern": pattern, 'SuccessSSupport': success_s_support,
                  'SuccessISupport': success_i_support,
                  'FailureSSupport': failure_s_support, 'FailureISupport': failure_i_support,
                  'SuccessTotalStudent': success_total_student, 'FailureTotalStudent': failure_total_student}

        pattern_failure = pattern_failure.append(newrow, ignore_index=True)

    pattern_success.to_csv("result/featured-patterns-" + pattern_type + "/problem" + str(cur_problem)+ "/pattern-success.csv")
    pattern_both_success.to_csv("result/featured-patterns-" + pattern_type + "/problem" + str(cur_problem)+"/pattern-both-success.csv")
    pattern_failure.to_csv("result/featured-patterns-" + pattern_type + "/problem"+ str(cur_problem)+"/pattern-failure.csv")
    pattern_both_failure.to_csv("result/featured-patterns-" + pattern_type + "/problem"+ str(cur_problem)+"/pattern-both-failure.csv")

    # i = i + 1
    end = time.time()
    print((end-start)/60)
    return pattern_success, pattern_both_success, pattern_failure, pattern_both_failure






#
# import sys
#
# sys.path.append("/Users/wwang33/Documents/ProgSnap2DataAnalysis/Datasets")
# from Dataset import *
# from CodeGraph import *
# from datetime import datetime
#
# """
# every assignment has multiple submissions, each from a student
# student-submission can give 1-1 mappings of students to submissions,
# we need to extract the student list from the attempt.csv dataset
# Each submission should include
# """
#
#
# class Assignment:
#     def __init__(self, data, assignment_id, p, q, lower_threshold, upper_threshold):
#         self.lower_threshold = lower_threshold
#         self.upper_threshold = upper_threshold
#         self.threshold_dir = self.pqgram_dir + "/threshold_" + str(self.lower_threshold) + "_" + str(
#             self.upper_threshold)
#         self.data = data
#         self.subject_id_list = data["pid"]
#         self.correct_subject_id_list = self.__get_correct_subject_id_list()
#
#     def __get_correct_subject_id_list(self):
#         df = self.data['df']
#         yes_pid = df[df.label == 'yes']['pid'].tolist()
#         return yes_pid
#
#     def __get_correct_subject_submission_pqgram_dict(self):
#         df = self.data['df']
#         subject_submission_pqgram_dict = df['abstract'].tolist()
#         return subject_submission_pqgram_dict
#
#     def get_correct_submission_pqgram_string_support_dict(self):
#         subject_submission_dict = self.__get_correct_subject_submission_pqgram_dict()
#         all_submission_pqgram_string_support_dict = {}
#         total_submission_count = len(self.correct_subject_id_list)
#         for code_graph in (subject_submission_dict).values():
#             try:
#                 pqgram_string_set = code_graph.pqgram_set.pqgram_string_set
#                 for pqgram_string in pqgram_string_set:
#                     if pqgram_string not in all_submission_pqgram_string_support_dict.keys():
#                         all_submission_pqgram_string_support_dict[pqgram_string] = 1 / total_submission_count
#                     else:
#                         all_submission_pqgram_string_support_dict[pqgram_string] += 1 / total_submission_count
#             except:
#                 print("code_graph is null: ", code_graph)
#         all_submission_pqgram_string_support_df = pd.DataFrame.from_dict(all_submission_pqgram_string_support_dict,
#                                                                          orient='index', columns=["support"])
#         save_obj(all_submission_pqgram_string_support_df, "correct_submission_pqgram_string_support_df",
#                  self.pqgram_dir, "")
#
#
#     def __get_patterns(self):
#         all_submission_pqgram_string_support_df = load_obj("correct_submission_pqgram_string_support_df",
#                                                            self.pqgram_dir, "")
#         patterns = []
#         for index in all_submission_pqgram_string_support_df.index:
#             if (all_submission_pqgram_string_support_df.ix[index, "support"] < self.upper_threshold and
#                     all_submission_pqgram_string_support_df.ix[index, "support"] > self.lower_threshold):
#                 patterns.append(index)
#         return patterns
#
#     def generate_x(self):
#         patterns = self.__get_patterns()
#         x = pd.DataFrame(columns=patterns, index=self.subject_id_list)
#         for subject_id in self.subject_id_list:
#             for pattern in patterns:
#                 time = self.__search_pattern_appear_time(pattern, subject_id)
#                 x.ix[subject_id, pattern] = time
#         save_obj(x, "x", self.threshold_dir, "")
#
# assignment = Assignment("CampMS2019", "daisy", 2, 3, 0.4, 0.6)
# assignment.generate_x()
# # assignment.generate_y()
#
#

