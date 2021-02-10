#!/usr/bin/env python
# coding: utf-8
import argparse
import time
start_time = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("-vs", "--voc_size", help="set the dimensiion of the one_hot vector, default=1000", default=1000, type=int)
parser.add_argument("-sl", "--sent_length", help="set the length for sentences entering the model, default=100", default=100, type=int)
parser.add_argument("-es", "--embedvec_size", help="set the length for sentences entering the model, default=100", default=100, type=int)
parser.add_argument("-bd", "--bi_dir", help="whether the LSTM is bi-directional or not, default=False", default=False, type=bool)
parser.add_argument("-dr", "--dropout_rate", help="set the dropout_rate for LSTM, default=0.1", default=0.1, type=float)
parser.add_argument("-ll", "--lstm_layers", help="the number of LSTM layers, default=0", default=0, type=int)
parser.add_argument("-l1s", "--lstm1_size", help="the number of neurons for the first LSTM layer, default=100", default=100, type=int)
parser.add_argument("-l2s", "--lstm2_size", help="the number of neurons for the second LSTM layer, default=64", default=64, type=int)
parser.add_argument("-l3s", "--lstm3_size", help="the number of neurons for the third LSTM layer, default=32", default=32, type=int)
args = parser.parse_args()
voc_size=args.voc_size
sent_length=args.sent_length
embedvec_size = args.embedvec_size  
bi_dir=args.bi_dir
dropout_rate = args.dropout_rate
lstm_layers = args.lstm_layers    
lstm1_size=args.lstm1_size
lstm2_size=args.lstm2_size
lstm3_size=args.lstm3_size

# output the parameters
print(f'voc_size: {voc_size}')
print(f'sent_length: {sent_length}')
print(f'embedvec_size : {embedvec_size }')
print(f'bi_dir: {bi_dir}')
print(f'dropout_rate : {dropout_rate }')
print(f'lstm_layers : {lstm_layers }')
_sizes=[lstm1_size, lstm2_size, lstm3_size]
for i in range(lstm_layers):
    print(f'the size for {i+1} layer: {_sizes[i]}')

# #### [中文停用字詞(含標點符號)](https://raw.githubusercontent.com/wfhsiao/opendata/master/data/stopwords.txt)處理範例

# In[1]:


import requests 
url = 'https://raw.githubusercontent.com/wfhsiao/opendata/master/data/stopwords.txt'
r=requests.get(url, stream=True)
stopwds=[]
for line in r.iter_lines():    
    line = line.decode('utf-8')
    line = line.strip()
    stopwds.append(line)
print(len(stopwds),stopwds[:10])
last_time = time.time()
print("--- %s seconds elasped for getting stopwords---" % (last_time - start_time))

# ##### 使用 [jieba](https://github.com/fxsjy/jieba)進行中文斷詞範例
# - 建議使用 [dict.txt.big](https://github.com/fxsjy/jieba/blob/master/extra_dict/dict.txt.big) 來獲得比較好的斷詞

# In[2]:


#!rm -f dict.txt.big
#
#get_ipython().system('wget https://github.com/fxsjy/jieba/raw/master/extra_dict/dict.txt.big')


# In[3]:


import jieba
import string
jieba.set_dictionary('dict.txt.big')  # 使用此行來獲得更好的斷詞
#origin=list(jieba.cut(df.loc[1, 'text']))
#withoutstopwds=[e for e in origin if e not in stopwds and e not in string.punctuation]


# In[4]:


import pandas as pd
import jieba

df=pd.read_pickle('https://raw.githubusercontent.com/wfhsiao/opendata/master/data/cofacts_2_categories.pkl')


# In[5]:


#df.tail(2)


# In[6]:


#groupby().size() is faster than values_count() 
#df.groupby('replyType').size()


# In[7]:


mapping={'RUMOR':1, 'NOT_RUMOR':0}
df.replace({'replyType':mapping}, inplace=True)
X=df.drop('replyType',axis=1)
y=df['replyType']
print(y.value_counts())

print("--- %s seconds elasped for getting cofacts_2---" % (time.time() - last_time))
last_time = time.time()

# In[8]:


#X.shape


# In[9]:


#y.shape


# In[10]:


import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.layers import Dropout, Flatten, SpatialDropout1D
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support as score


# In[11]:


print(tf.__version__)
'''import logging

logger = tf.get_logger()
logger.setLevel(logging.ERROR)
'''

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# # Onehot Representation

# In[12]:


messages=X.copy()


# In[13]:


import re, string


# In[14]:

messages['seg_text']=messages['text'].apply(lambda x: ' '.join([e for e in jieba.cut(x.lower()) if e not in stopwds and e not in string.punctuation]))
corpus=messages['seg_text'].tolist()
print("--- %s seconds elasped for jieba segmentation ---" % (time.time() - last_time))
last_time = time.time()

# In[15]:


#corpus[:5]


# In[16]:

onehot_repr=[one_hot(words,voc_size) for words in corpus] 
#onehot_repr[15]


# # Embedding Representation

# In[17]:

embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
#print(embedded_docs)

print("--- %s seconds elasped for one_hot encoding and padding ---" % (time.time() - last_time))
last_time = time.time()

# In[18]:


#embedded_docs[0]


# In[20]:


# In[22]:
#from keras.backend import clear_session

def create_model():
    tf.compat.v1.reset_default_graph()
    model=Sequential()
    model.add(Embedding(voc_size,embedvec_size,input_length=sent_length))
    model.add(SpatialDropout1D(0.1))
    lstm_sizes=[lstm1_size, lstm2_size, lstm3_size]
    for i in range(lstm_layers):
        size = lstm_sizes[i]
        if args.bi_dir:
            model.add(Bidirectional(LSTM(size, dropout=dropout_rate, 
                      recurrent_dropout=0.2, return_sequences=True)))
        else:
            model.add(LSTM(size, dropout=dropout_rate, recurrent_dropout=0.2,
                           return_sequences=True))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1,activation='sigmoid'))
    return model


# In[23]:


'''model = create_model()
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
'''

# In[24]:


print(len(embedded_docs),y.shape)


# In[25]:


import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)


# In[26]:


#y_final


# In[27]:


print(X_final.shape,y_final.shape)


# In[29]:


#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42))


# # Model Training
# - [accuracy and f1-score for keras model](https://datascience.stackexchange.com/questions/45165/how-to-get-accuracy-f1-precision-and-recall-for-a-keras-model)
# - [evaluate the performance of deep learning models with k-fold startified cross validation](https://machinelearningmastery.com/evaluate-performance-deep-learning-models-keras/)
# - [k-fold stratified cross validation model metrics for f1](https://stackoverflow.com/questions/52892099/how-can-i-use-k-fold-cross-validation-in-scikit-learn-to-get-precision-recall-pe)
# - [check the tensorflow add-ons](https://stackoverflow.com/questions/64474463/custom-f1-score-metric-in-tensorflow)

# In[33]:


# 
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

# In[34]:


from sklearn.model_selection import cross_validate
import tensorflow as tf

#k-fold cross validation
import numpy

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
print("--- %s seconds elasped for others ---" % (time.time() - last_time))
last_time = time.time()

with tf.device('/device:GPU:2'):
    naccuracy=[]
    nf1=[]
    for j in range(10):
          print(f'seed:{j}')
          cvscores_accuracy = []
          cvscores_f1 = []

          kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=j)
          print('fold: ', end='')
          for i,(train, test) in enumerate(kfold.split(X_final, y_final)):
                print(f'{i}', end=' ')
                ## Creating model
                model = create_model()    
                # Compile model
                model.compile(loss='binary_crossentropy', optimizer='adam')
                # Fit the model
                history = model.fit(X_final[train],y_final[train],
                          epochs=10,batch_size=64, verbose=0)
                # evaluate the model
                '''loss, accuracy, f1, precision, recall = \
                      model.evaluate(X_final[test], 
                                     y_final[test], verbose=0)'''
                y_pred = model.predict(X_final[test]).flatten()
                
                y_pred[y_pred>=.5]=1
                y_pred[y_pred< .5]=0
                f1=f1_score(y_final[test], y_pred, average="binary")
                accuracy=accuracy_score(y_final[test], y_pred)
                #print("%s: %.2f%%" % (model.metrics_names[1], accuracy*100))
                cvscores_accuracy.append(accuracy * 100)  
                #f1 = 2*((precision*recall)/(precision+recall+K.epsilon()))
                cvscores_f1.append(f1*100)
                #print("--- %s seconds elasped for one fold ---" % (time.time() - last_time))
                #last_time = time.time()

          naccuracy.extend(cvscores_accuracy)
          nf1.extend(cvscores_f1)
          print()
          print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores_accuracy), numpy.std(cvscores_accuracy)))
          print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores_f1), numpy.std(cvscores_f1)))


# In[35]:


print(np.mean(naccuracy), np.std(naccuracy))


# In[36]:


print(np.mean(nf1), np.std(nf1))


# # Adding Dropout

# In[ ]:

print("--- %s seconds elasped for the whole program ---" % \
      (time.time() - start_time))


