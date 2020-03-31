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
pattern_data = np.array(load_obj('full_patterns', data_dir))

embeddings = embeddings[kept_index]
print(embeddings.shape)
print(len(pattern_data))

sb3_pattern_list = []
for ind in (idx2sb3):
    sb3_pattern_list.append(idx2sb3[ind])
sb3_pattern_list = np.array(sb3_pattern_list)

xsorted = np.argsort(pattern_data)

ypos = np.searchsorted(pattern_data[xsorted], sb3_pattern_list)
indices2 = xsorted[ypos]

Mat=np.random.randint(1,10,3*4).reshape((3,4))  # some random data vector


def index_verification():
    # below three should be the same!
    print(pattern_data[indices2])
    print(sb3_pattern_list)
    # just the idx number be different
    print(idx2sb3)
#    these two numbers also should be same:
    assert len(indices2)==len(embeddings)==135, "there should be 135 words"

def f(x_row):
    transfer_matrix = np.zeros([100, len(indices2)])
    x_row = np.digitize(x_row, bins=[1])
    x_row = x_row[indices2]
    for x_ind, x_data in enumerate(x_row):
        if x_data > 0:
            transfer_matrix[:, x_ind] = embeddings[x_ind]
            # assert(idx2sb3[x_ind] == sb3_pattern_list[x_ind]), "index incoherent!"
    x_new = np.sum(transfer_matrix, axis = 1)
    return x_new

def f_135(x_row):
    transfer_matrix = np.zeros([100, len(indices2)])
    x_row = np.digitize(x_row, bins=[1])
    x_row = x_row[indices2]
    return x_row


# index_verification()
data_dir = base_dir + "/xy_0heldout/scratch_code_state[[1, 0]]"
x_train = load_obj('X_train', data_dir)
print(x_train.shape)
m = np.apply_along_axis(lambda x: f_135(x), 1, x_train)
print(m.shape)

assert x_train.shape[0] == m.shape[0], 'convertion incorrect'
save_obj(m, 'x_train_135', data_dir)