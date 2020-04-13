from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from trainers.CrossValidationTrainer import *
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
#
# y = [0]*34
# y[23] = 1
# y[26] = 1
# y[29] = 1
# y[31] = 1
# y[32] = 1
# y[33] = 1
#
#
# y = np.array(y)
# save_obj(y, 'y_train', save_dir)



from save_load_pickle import *
save_dir_y = base_dir + "xy_0heldout/code_state[[1, 0]]/credesclones"
save_dir_x = base_dir + "xy_0heldout/code_state[[1, 0]]/credesclones"

x = np.zeros([5, 6])
for i in range(5):
    x[i, 0] = 1
# x[1, 3] = 1
for i in range(2, 5):
    x[i, 1] = 1

x[3, 1] = 1
print(x)
x = x[:,[1,2]]

# x = np.array(x)

y = np.zeros(5)
y[2] = 1
y[3] = 1
y[4] = 1
y = np.array(y).transpose()
save_obj(x, 'X_train_manual', save_dir_x)
save_obj(x, 'y_train_manual', save_dir_y)

# trainer = DPMCrossValidationTrainer()
# trainer = CrossValidationTrainer()
# trainer.populate(x, y)
# trainer.cross_val_get_score()



pid = load_obj()




