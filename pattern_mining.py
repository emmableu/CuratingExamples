import sys
sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets')
from Dataset import *
sys.setrecursionlimit(10**8)
np.set_printoptions(threshold=sys.maxsize)
from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer(smooth_idf=False)



code_shape_p_q_list = [[1, 0], [1, 1], [1, 2], [1, 3]]
dataset = Dataset(code_shape_p_q_list = code_shape_p_q_list)
# dataset.create_code_state()
# dataset.get_all_pattern_keys()

label_name = 'moveanimate'


orig_dir = base_dir + "/cv/test_size0/fold0/code_state[[1, 0], [1, 1], [1, 2], [1, 3]]baseline/" + label_name

x_orig = load_obj('X_train', orig_dir, "")
y_orig = load_obj('y_train', orig_dir, "")

patterns = load_obj("significant_patterns", orig_dir, "")
pattern_list = [pattern for pattern in patterns]

x = np.digitize(x_orig,bins=[1])




y_yes_index = np.where(y_orig == 1)[0]
print(len(y_yes_index))
yes_x = x[y_yes_index]


tfidf_all = transformer.fit_transform(x_orig).toarray()

print("print(transformer.idf_)")
all_weights = transformer.idf_



tfidf_yes = transformer.fit_transform(yes_x).toarray()

print("print(transformer.idf_)")
yes_weights = transformer.idf_
print(yes_weights)



y_no_index = np.where(y_orig == 0)[0]
# print(y_yes_index)

no_x = x[y_no_index]

tfidf_no = transformer.fit_transform(no_x).toarray()
no_weights = transformer.idf_

count = 0

feature_left = []
for i in range(len(yes_weights)):
    if yes_weights[i]  > 0 and  yes_weights[i]  < 100 and abs(yes_weights[i]- no_weights[i]) > 1:
        #this is good for costopall


    # if yes_weights[i]  > 3 and  yes_weights[i]  < 100 and abs(yes_weights[i]- no_weights[i]) > 3:
        #this is good for jump and keymove


    # if abs(yes_weights[i]- no_weights[i]) > 1 and all_weights[i] > 6.5:
    # this is good for moveanimate


    # if abs(yes_weights[i]- no_weights[i]) > 1 and all_weights[i] > 4:
    # this is good for cochangescore


    # if abs(yes_weights[i]- no_weights[i]) > 1 and all_weights[i] > 4:
    # this is good for movetomouse


    # should use voi for these grid points
        count += 1
        print(i)
        print(pattern_list[i])
        print(yes_weights[i])
        print(no_weights[i])
        print(all_weights[i])
        feature_left.append(i)
print(count)









#
#
#
#
# print(yes_x.shape)
#
# x_yes_index = np.where(yes_x == 1)
# probs = []
# for i in (np.unique(x_yes_index)):
#     prob = len(np.where(x[:, i])[0])
#     probs.append(prob)
#
# order = np.argsort(probs)[::-1]
#
# patterns = load_obj("significant_patterns", orig_dir, "")
# pattern_list = [pattern for pattern in patterns]
# print(pattern_list)
#
# data = np.array(pattern_list)[order]
#
# probs.sort()
# # save_obj(data, "patterns_moveanimate", base_dir, 'temp')
# print(probs[::-1])
#
#
# # Removed baseline from models in case baseline result changes

x = x_orig[:, feature_left]
y = y_orig


save_obj(x, 'x', base_dir+"/best_train/tfidf_transformed", label_name)
save_obj(y, 'y', base_dir+"/best_train/tfidf_transformed", label_name)




model_list = [ adaboost, gaussian_nb,
                    bernoulli_nb, multi_nb, complement_nb, mlp]

for model in model_list:
    print(model.name)
    model.naive_predict(x, y)
#
#
#
#
#
#
#
