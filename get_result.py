import sys
sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets')
from Dataset import *
sys.setrecursionlimit(10**8)


if __name__=='__main__':
    # code_shape_p_q_list = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0]]
    # code_shape_p_q_list = [[1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [2, 0], [2, 1], [2, 2], [2, 3], [2, 4]]
    code_shape_p_q_list2 = [[1, 0], [1, 1], [1, 2], [1, 3], [2, 3]]
    code_shape_p_q_list1 = [[1, 0]]
    for code_shape_p_q_list in [code_shape_p_q_list1]:
        dataset = Dataset(code_shape_p_q_list = code_shape_p_q_list)
        # dataset.create_code_state()
        # dataset.get_all_pattern_keys()
        dataset.submission_save_x_y_to_hard_drive(selected_p_q_list= code_shape_p_q_list)
    # dataset.get_result(baseline = True)
    # print(dataset.action['keymove'])