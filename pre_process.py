import sys
sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets')
from Dataset import *

if __name__=='__main__':
    code_shape_p_q_list = [[1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0]]
    allow_gap = True
    dataset = Dataset(total = 186, code_shape_p_q_list = code_shape_p_q_list, allow_gap = allow_gap)
    dataset.create_input_data()
    print(dataset.action['keymove'])