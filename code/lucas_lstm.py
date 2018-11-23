import numpy as np
from keras import initializers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(lowercase=False, token_pattern='[A-Z;+;-]+')
n = 10

#Treino
with open("classes.txt","r") as classes:
	text = classes.read().split(' ')
data_train = []
classes_train = []
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

print(vectorizer.get_feature_names())
print(data_train.shape)
print(classes_train.shape)
print(len(data_train))

#Teste

# with open("classes_teste.txt","r") as classes:
# 	text = classes.read().split(' ')
# data_test = []
# classes_test = []
# corpus = vectorizer.fit_transform(text)
# corpus = corpus.toarray()

# count1=0 
# count2=n-1
# while(count2<len(corpus)-1):
# 	count2 += 1
# 	data_test.append(corpus[count1:count2])
# 	classes_test.append(corpus[count2])
# 	count1 += 1

# data_test = np.array(data_train)
# classes_test = np.array(classes_train)

# print(vectorizer.get_feature_names())
# print(data_test.shape)
# print(classes_test.shape)
# print(len(data_test))


np.random.seed(7)
model = Sequential()
model.add(LSTM(50,input_shape=(n,26)))
model.add(Dense(25,activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(26, activation=lambda x: x))
model.compile(loss='binary_crossentropy',optimizer ='adam',metrics=['accuracy'])
model.fit(data_train,classes_train,epochs=2, batch_size=20,validation_split=0.2,verbose=1);
#predict = model.predict(data_test, batch_size=20, verbose=0) #alterar para data_test, usar mesma janela
# predict eh vetor de proabilidade
# comparar com o classes_test, posicao por posicao
# salvar true positive, true negative, false positive, false negative para cada classe

# callbacks=[EarlyStopping(min_delta=0.00025, patience=2)] entra no model.fit
