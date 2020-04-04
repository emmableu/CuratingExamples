from save_load_pickle import *
save_dir = base_dir + "xy_0heldout/code_state[[1, 0]]/credesclones"

y = [0]*34
y[23] = 1
y[26] = 1
y[29] = 1
y[31] = 1
y[32] = 1
y[33] = 1


y = np.array(y)
save_obj(y, 'y_train', save_dir)