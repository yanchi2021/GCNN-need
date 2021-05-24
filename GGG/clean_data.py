
import numpy as np
import pandas as pd
import chardet
import sklearn.preprocessing as pre_processing
dataset='book'

def get_encoding(file):
    # äºŒè¿›åˆ¶æ–¹å¼è¯»å–ï¼ŒèŽ·å–å­—èŠ‚æ•°æ®ï¼Œæ£€æµ‹ç±»åž?
    with open(file, 'rb') as f:
        return chardet.detect(f.read())['encoding']

from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords
st_words=stopwords.words('english')
for w in ['!',',','.','?','-s','-ly','</s>','s']:
    st_words.append(w)

def clean_(text):
    review_text = BeautifulSoup(text, "html.parser").get_text()
    # ç”¨æ­£åˆ™è¡¨è¾¾å¼å–å‡ºç¬¦åˆè§„èŒƒçš„éƒ¨åˆ?
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    # å°å†™åŒ–æ‰€æœ‰çš„è¯ï¼Œå¹¶è½¬æˆè¯list
    words = review_text.lower().split()
    # è¿”å›žwords list
    new_words=[]
    for word in words:
        if word not in st_words and word !='\n':
            new_words.append(word)
    return new_words

def load_data(dataset):
  if dataset=='mr':
    file_path='text_gcn-master/data/mr/text_train.txt'
    # encoding_ = get_encoding(file_path)
    train_data_mr_x=pd.read_table(file_path,encoding='utf-8',names=['text'])
    train_data_mr_y=pd.read_table('text_gcn-master/data/mr/label_train.txt',encoding='utf-8',names=['label'])
    test_data_mr_x=pd.read_table('text_gcn-master/data/mr/text_test.txt',encoding='utf-8',names=['text'])
    test_data_mr_y=pd.read_table('text_gcn-master/data/mr/label_test.txt',encoding='utf-8',names=['label'])

    train_list=[]
    for i in range(len(train_data_mr_x)):
        text_list=clean_(train_data_mr_x['text'][i])
        train_list.append(' '.join(text_list))
    test_list=[]
    for i in range(len(test_data_mr_x)):
        text_list=clean_(test_data_mr_x['text'][i])
        test_list.append(' '.join(text_list))

    train_data_mr_all=train_list+test_list
    label_data_mr_all=list(train_data_mr_y['label'])+list(test_data_mr_y['label'])

  elif dataset=='R52':
    train_data_=pd.read_table('text_gcn-master/data/R52/train.txt',encoding='utf-8',sep='\t',names=['label','text'])
    test_data_=pd.read_table('text_gcn-master/data/R52/test.txt',encoding='utf-8',sep='\t',names=['label','text'])
    train_list = []
    for i in range(len(train_data_)):
        text_list = clean_(train_data_['text'][i])
        train_list.append(' '.join(text_list))
    test_list = []
    for i in range(len(test_data_)):
        text_list = clean_(test_data_['text'][i])
        test_list.append(' '.join(text_list))
    train_data_mr_all=train_list+test_list
    label_data_mr_all=list(train_data_['label'])+list(test_data_['label'])
    R_label=pre_processing.LabelEncoder()
    label_data_mr_all=R_label.fit_transform(label_data_mr_all)
    label_data_mr_all=list(label_data_mr_all)
  elif dataset=='R8':
    train_data_=pd.read_table('text_gcn-master/data/R8/train.txt',encoding='utf-8',sep='\t',names=['label','text'])
    test_data_=pd.read_table('text_gcn-master/data/R8/test.txt',encoding='utf-8',sep='\t',names=['label','text'])
    train_list = []
    for i in range(len(train_data_)):
        text_list = clean_(train_data_['text'][i])
        train_list.append(' '.join(text_list))
    test_list = []
    for i in range(len(test_data_)):
        text_list = clean_(test_data_['text'][i])
        test_list.append(' '.join(text_list))
    train_data_mr_all=train_list+test_list
    label_data_mr_all=list(train_data_['label'])+list(test_data_['label'])
    R_label=pre_processing.LabelEncoder()
    label_data_mr_all=R_label.fit_transform(label_data_mr_all)
    label_data_mr_all=list(label_data_mr_all)
  elif dataset=='book':
      import pickle

      f = open('texts_cleaned.pckl', 'rb')
      data_list = pickle.load(f)
      f.close()
      f=open('labels.pckl','rb')
      labels=pickle.load(f)
      f.close()
      # data_list=[]
      # count=0
      # for line in texts:
      #     print('clean text'+str(count))
      #     count+=1
      #     line_cleaned=clean_(line)
      #     data_list.append(' '.join(line_cleaned))
      R_label = pre_processing.LabelEncoder()
      train_data_mr_all=data_list
      label_data_mr_all = R_label.fit_transform(labels)
      label_data_mr_all = list(label_data_mr_all)
  # f = open('texts_cleaned.pckl', 'wb')
  # pickle.dump(train_data_mr_all, f)
  # f.close()
  from sklearn.model_selection import train_test_split
  train_X,test_X,train_Y,test_Y = train_test_split(train_data_mr_all,label_data_mr_all,test_size=0.2)
  return train_X,test_X,train_Y,test_Y

from sklearn.feature_extraction.text import TfidfVectorizer
#
# train_X,test_X,train_Y,test_Y=load_data(dataset)
# print('clean data done')
# X_all=train_X+test_X
# Y_all=train_Y+test_Y
#
# # vectorizer = TfidfVectorizer(
# #     strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
# #     use_idf=1, smooth_idf=1, sublinear_tf=1
# # )
# # vectorizer.fit(X_all)
# np_tfidf=np.load('np_tfidf.npy')
# df_tfidf=pd.DataFrame(np_tfidf)
# # df_tfidf = vectorizer.transform(X_all)
# # np_tfidf = df_tfidf.toarray()
# # np.save('np_tfidf.npy',np_tfidf)
# vocab=np.load('vocab.npy')
# # vocab = vectorizer.get_feature_names()
# # vocab = np.array(vocab)
# # df_tfidf=pd.DataFrame(df_tfidf,columns=vocab)
# # np.save('vocab.npy',vocab)
# print(vocab.shape)
# word_map = {}
# num = 0
# for word in vocab:
#     word_map[word] = num
#     num = num + 1
# print('voca cal done')
