import sys
sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets')
from Dataset import *
sys.setrecursionlimit(10**8)

code_shape_p_q_list = [[1, 0], [1, 1], [1, 2], [1, 3]]
dataset = Dataset(code_shape_p_q_list = code_shape_p_q_list)
# dataset.create_code_state()
# dataset.get_all_pattern_keys()
dataset.save_x_y_to_hard_drive_temp(selected_p_q_list=[[1, 0], [1, 1], [1, 2], [1, 3]], baseline = False)
# dataset.get_result(baseline = False)
