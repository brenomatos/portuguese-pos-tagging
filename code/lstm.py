import numpy as np
from keras import initializers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

with open("classes.txt","r") as classes:
	text = classes.read().split(' ')

n = 10
data_train = []
classes_train = []
vectorizer = CountVectorizer(lowercase=False, token_pattern='[A-Z;+;-]+')
corpus = vectorizer.fit_transform(text)
corpus = corpus.toarray()

count1=0
count2=n-1
while(count2<len(corpus)-1):
	count2 += 1
	data_train.append(corpus[count1:count2])
	classes_train.append(corpus[count2])
	count1 += 1

data_train = np.array(data_train)
classes_train = np.array(classes_train)
print(data_train.shape)
print(classes_train.shape)
print(len(data_train))


np.random.seed(7)
model = Sequential()
model.add(LSTM(50,input_shape=(n,26)))
model.add(Dense(25,activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(26, activation=lambda x: x))
model.compile(loss='binary_crossentropy',optimizer ='adam',metrics=['accuracy'])
model.fit(data_train,classes_train,epochs=2, batch_size=20,validation_split=0.2,verbose=1);
# callbacks=[EarlyStopping(min_delta=0.00025, patience=2)] entra no model.fit
