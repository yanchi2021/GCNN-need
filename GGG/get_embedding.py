from bert_serving.client import BertClient

import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# # train_X,test_X,train_Y,test_Y=load_data(dataset)
# print('clean data done')
# # X_all=train_X+test_X
# # Y_all=train_Y+test_Y
# f = open('texts_cleaned.pckl', 'rb')
# data_list = pickle.load(f)
# f.close()
# vectorizer = TfidfVectorizer(
#     strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
#     use_idf=1, smooth_idf=1, sublinear_tf=1
# )
# vectorizer.fit(data_list)
#
# vocab = vectorizer.get_feature_names()
# vocab = np.array(vocab)
# # df_tfidf=pd.DataFrame(df_tfidf,columns=vocab)
# np.save('vocab.npy',vocab)
vocab=np.load('vocab.npy')
vocab=vocab.tolist()


bc = BertClient()
vec = bc.encode(vocab)
word=vec[:,1,:]
np.save('embedding.npy',word)
print(word.shape)  # [2, 25, 768]
# vec[0]  # [1, 25, 768], sentence embeddings for `hey you`
# vec[0][0]  # [1, 1, 768], word embedding for `[CLS]`
# vec[0][1]  # [1, 1, 768], word embedding for `hey`
# vec[0][2]  # [1, 1, 768], word embedding for `you`
# vec[0][3]  # [1, 1, 768], word embedding for `[SEP]`
# vec[0][4]  # [1, 1, 768], word embedding for padding symbol
# vec[0][25]  # error, out of index!
