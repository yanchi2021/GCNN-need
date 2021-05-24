# get tfidf

from clean_data import load_data
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import pandas as pd
train_X,test_X,train_Y,test_Y=load_data('book')
X_all=train_X+test_X
Y_all=train_Y+test_Y
# np_tfidf=np.load('np_tfidf.npy')
# print(np_tfidf.shape)
# df_tfidf=pd.DataFrame(np_tfidf)
vectorizer = TfidfVectorizer(
    strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
    use_idf=1, smooth_idf=1, sublinear_tf=1
)
vectorizer.fit(X_all)
df_tfidf = vectorizer.transform(X_all)
print('df_tfidf done')
# np_tfidf = df_tfidf.toarray()
# print(np_tfidf.shape)
# vocab = vectorizer.get_feature_names()
# vocab = np.array(vocab)
# df_tfidf=pd.DataFrame(df_tfidf,columns=vocab)
vocab=np.load('vocab.npy')
word_map = {}
num = 0;
for word in vocab:
    word_map[word] = num
    num = num + 1

windows_size = 10
dum = []
window_num = 0
word_count = {}
# bi_gram_count = np.zeros((vocab.shape[0], vocab.shape[0]))
word_pairs_ = {}

# In[19]:
# caculate PMI

# for doc in X_all:
#     doc = doc.split(' ')
#     if len(doc) > windows_size:
#         for i in range(len(doc) - windows_size + 1):  # i is the index of the whole doc , the index of the window
#             # 每个文档滑动
#             window_num = window_num + 1  # window num count
#             for j in range(0, windows_size):  # j is the index of the window, from 0 to window_size
#                 the_word = doc[i + j]
#                 if the_word in word_count:
#                     word_count[the_word] = word_count[the_word] + 1
#                 else:
#                     word_count[the_word] = 1
#                 # get single-word count
#                 for k in range(0, j):  # k is the head-index of j ,from 0 to j
#                     if j == 0:
#                         continue
#                     else:
#                         word_pair = doc[i + k] + ',' + doc[i + j]
#                         if word_pair in word_pairs_:
#                             word_pairs_[word_pair] += 1
#                         else:
#                             word_pairs_[word_pair] = 1
#
#                         word_pair = doc[i + j] + ',' + doc[i + k]
#                         if word_pair in word_pairs_:
#                             word_pairs_[word_pair] += 1
#                         else:
#                             word_pairs_[word_pair] = 1
#     #                         head_index=word_map[doc[i+k]]
#     #                         back_index=word_map[doc[i+j]]
#
#     #                         bi_gram_count[head_index][back_index]+=1
#     #                         bi_gram_count[back_index][head_index]+=1
#     else:
#         window_num = window_num + 1
#         for j in range(0, len(doc)):  # j is the index of the window, from 0 to window_size
#             the_word = doc[j]
#             if the_word in word_count:
#                 word_count[the_word] = word_count[the_word] + 1
#             else:
#                 word_count[the_word] = 1
#             # get single-word count
#             for k in range(0, j):  # k is the head-index of j ,from 0 to j
#                 if j == 0:
#                     continue
#                 else:
#                     word_pair = doc[k] + ',' + doc[j]
#                     if word_pair in word_pairs_:
#                         word_pairs_[word_pair] += 1
#                     else:
#                         word_pairs_[word_pair] = 1
#                     word_pair = doc[j] + ',' + doc[k]
#                     if word_pair in word_pairs_:
#                         word_pairs_[word_pair] += 1
#                     else:
#                         word_pairs_[word_pair] = 1
# #                     head_index=word_map[doc[k]]
# #                     back_index=word_map[doc[j]]
#
# #                     bi_gram_count[head_index][back_index]+=1
# #                     bi_gram_count[back_index][head_index]+=1
#
#
# # In[20]:
# row = []
# col = []
# weight = []
#
# for word_pair in word_pairs_:
#     word_a, word_b = word_pair.split(',')
#     word_a_fre = word_count[word_a]
#     word_b_fre = word_count[word_b]
#     word_ab_fre = word_pairs_[word_pair]
#
#     pmi = np.log((1.0 * word_ab_fre / window_num) /
#
#               (1.0 * word_a_fre * word_b_fre / (window_num * window_num)))
#     if pmi <= 0:
#         continue
#     row.append(word_map[word_a])
#     col.append(word_map[word_b])
#     weight.append(pmi)
#
# # In[24]:
#
# idf_coo = df_tfidf.tocoo()
# vocab_size = vocab.shape[0]
#
#
# doc_row = idf_coo.row
# doc_col = idf_coo.col
# doc_data = idf_coo.data
#
#
#
# row = row + list(doc_row + vocab_size)
# col = col + list(doc_col)
# weight = weight + list(doc_data)
# # 左下角
# print(max(row), max(col))
# # word-word word-doc  word 18007 doc 10662
# # doc-word doc-doc
# row = row + list(doc_col)
# col = col + list(doc_row + vocab_size)
# weight = weight + list(doc_data)
# print(max(row), max(col))
# # 右上角
vocab_size = vocab.shape[0]
node_size = vocab_size + len(X_all)
print('vocab size:',vocab_size,'nodes:',node_size)
import scipy.sparse as sp


# adj = sp.csr_matrix(
#     (weight, (row, col)), shape=(node_size, node_size))
# sp.save_npz('adj.npz',adj, compressed=True)
adj=sp.load_npz('adj.npz')
print('load adj done')
def preprocess_adj(adj, symmetric=True):
    adj = adj + sp.eye(adj.shape[0])    # 对加入自连接的邻接矩阵进行对称归一化处理    adj = normalize_adj(adj, symmetric)
    return adj

def normalize_adj(adj, symmetric=True):
    # 如果邻接矩阵为对称矩阵，得到对称归一化邻接矩阵    # D^(-1/2) * A * D^(-1/2)
    if symmetric:        # A.sum(axis=1)：计算矩阵的每一行元素之和，得到节点的度矩阵D
        # np.power(x, n)：数组元素求n次方，得到D^(-1/2)
        # sp.diags()函数根据给定的对象创建对角矩阵，对角线上的元素为给定对象中的元素
        d = sp.diags(np.power(np.array((adj!=0).sum(1)), -0.5).flatten(), 0)
        # tocsr()函数将矩阵转化为压缩稀疏行矩阵
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    # 如果邻接矩阵不是对称矩阵，得到随机游走正则化拉普拉斯算子    # D^(-1) * A
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten(), 0)
        a_norm = d.dot(adj).tocsr()
    return a_norm
