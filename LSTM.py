import numpy as np 
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import numpy as np
from nltk.corpus import stopwords 
from time import time
import scipy.stats as stats
from keras.models import Model
from keras.metrics import Recall
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
import XXE

data= pd.read_csv("./dataset/kaggleDataset/sqliv2/sqliv2.csv",encoding='utf-16')

vectorizer = CountVectorizer(min_df=2, max_df=0.7, stop_words=stopwords.words('english')) 
vectorizer.fit(data['Sentence'].values.astype('U')) # .astype('U') -> konverto ne Unicode   
print(vectorizer.vocabulary_) 
vector= vectorizer.transform(data['Sentence'].values.astype('U'))
posts= vector.toarray()

transformed_posts=pd.DataFrame(posts)
df=pd.concat([data,transformed_posts],axis=1)
df.dropna(inplace=True)

X=df['Sentence']
y=df['Label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_words = 1000
max_len = 150
tok = Tokenizer(num_words=max_words)
tok.fit_on_texts(X_train)
sequences = tok.texts_to_sequences(X_train)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)

from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding

def RNN():
    inputs = Input(name='inputs',shape=[150])
    layer = Embedding(1000,50,input_length=150)(inputs)
    layer = LSTM(64)(layer)
    layer = Dense(256,name='FC1')(layer)
    layer = Activation('relu')(layer)
    layer = Dropout(0.5)(layer)
    layer = Dense(1,name='out_layer')(layer)
    layer = Activation('sigmoid')(layer)
    model = Model(inputs=inputs,outputs=layer)
    return model

model = RNN()
model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=[Recall()])
history = model.fit(sequences_matrix, y_train,batch_size=128,epochs=10,
          validation_split=0.2,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])

X_test_sequences = tok.texts_to_sequences(X_test)
X_test_sequences_matrix = sequence.pad_sequences(X_test_sequences,maxlen=max_len)
recall = model.evaluate(X_test_sequences_matrix,y_test)
y_pred=model.predict(X_test_sequences_matrix)

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

for i in range(len(y_pred)):
    if y_pred[i]>0.5:
        y_pred[i]=1
    elif y_pred[i]<=0.5:
        y_pred[i]=0

accuracy=accuracy_score(y_test, y_pred)
precision=precision_score(y_test, y_pred, zero_division=1)
recall=recall_score(y_test, y_pred, zero_division=1)
print(" Accuracy : {0} \n Precision : {1} \n Recall : {2}".format(accuracy, precision, recall))