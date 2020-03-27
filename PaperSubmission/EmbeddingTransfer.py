import sys
sys.path.append('/Users/wwang33/Documents/IJAIED20/CuratingExamples/Datasets')
from Dataset import *
sys.setrecursionlimit(10**8)


embedding_dir = base_dir + "/Embeddings/"
word2idx, idx2word = tuple(np.load(embedding_dir + "dict.npy", allow_pickle=True))
embeddings = np.load(embedding_dir + "embeddings.npy")

data_dir = base_dir + "/xy_0heldout/scratch_code_state[[1, 0]]"
pattern_data = load_obj('full_patterns', data_dir)

idx2sb3 = {}
kept_index = []
for ind in idx2word:
    try:
        new_sb3_word = SB2_TO_SB3_OP_MAP[idx2word[ind]]
        idx2sb3[ind] = new_sb3_word
        kept_index.append(ind)
    except:
        pass

sb32idx = {value: key for key, value in idx2sb3.items()}

data_dir = base_dir + "/xy_0heldout/scratch_code_state[[1, 0]]"
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

# print(pattern_data)
translated_snap = []
for i in w2i:
    try:
        translated_snap.append(SB2_TO_SB3_OP_MAP[i])
        print(i)
    except:
        print("no")


start_x = [0] * 100


