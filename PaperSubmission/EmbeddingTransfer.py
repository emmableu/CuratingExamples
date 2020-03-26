import sys
sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets')
from Dataset import *
sys.setrecursionlimit(10**8)

embedding_dir = base_dir + "/Embeddings/"
word2idx, idx2word = tuple(np.load(embedding_dir + "dict.npy", allow_pickle=True))
embeddings = np.load(embedding_dir + "embeddings.npy")
# one_hot_embeddings = np.diag(np.ones((embeddings.shape[0])))

data_dir = base_dir + "/xy_0heldout/code_state[[1, 0]]"
x_train = load_obj('X_train', data_dir)
pattern_data = load_obj('full_patterns', data_dir)
w2i = np.array([data for data in word2idx])


# for i in pattern_data:
#     if i in w2i:
#         print(i)

# word2idx.keys()
#
w2i = np.array([data for data in word2idx])
w2i = sorted(w2i)
print(w2i)
print(pattern_data)