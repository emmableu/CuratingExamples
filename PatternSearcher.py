import sys
sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets')
from Dataset import *

pqgram_folder = root_dir + "/Datasets/data/SnapPQGram_413/"
pid_s = []
file_list = os.listdir(pqgram_folder)

def check_pid_are_same(file_list):
    for file_name in file_list:
        if file_name.endswith(".csv"):
            pid = file_name.split("-")[0]
            pid_s.append(pid)
    # print(len(pid_s))
    assert len(pid_s) == 413, "pid length is not 413"
    pid_s.sort()
    original_pid = load_obj("pid", base_dir, "")
    assert pid_s == original_pid, "new pid should be the same as old pid"

