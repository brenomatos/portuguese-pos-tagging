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
# 
n = 10 # window size
data_train = []
classes_train = []
vectorizer = CountVectorizer(lowercase=False, token_pattern='[A-Z;+;-]+')
corpus = vectorizer.fit_transform(text)
corpus = corpus.toarray()
# 
count1=0 #sliding window
count2=n-1 #sliding window
while(count2<len(corpus)-1):
	count2 += 1
	data_train.append(corpus[count1:count2])
	classes_train.append(corpus[count2])
	count1 += 1
# 
data_train = np.array(data_train)
classes_train = np.array(classes_train)
print(data_train.shape)
print(classes_train.shape)
print(len(data_train))
# 
# 
np.random.seed(7)
model = Sequential()
model.add(LSTM(50,input_shape=(n,26)))
model.add(Dense(25,activation='relu'))
model.add(Dense(2, activation='sigmoid'))
model.add(Dense(26, activation=lambda x: x))
model.compile(loss='binary_crossentropy',optimizer ='adam',metrics=['accuracy'])
model.fit(data_train,classes_train,epochs=1, batch_size=20,validation_split=0.2,verbose=1);
# callbacks=[EarlyStopping(min_delta=0.00025, patience=2)] entra no model.fit


def getOneHot(test_corpus):
	for i in range(0,len(test_corpus)):
		if(test_corpus[i]==1):
			return i

#generating test samples
data_test = []
classes_test = []

classes = open("test_words.txt","w")
words = open("test_classes.txt", "w")
with open("../data/macmorpho-test.txt","r") as f:
    text = f.read()
    words_list = text.split( )
    for word in words_list: # get all different tokens in text
        class_aux = word.split("_")
        words.write(str(class_aux[1])+" ")
        classes.write(str(class_aux[0])+" ")
classes.close()
words.close()


with open("classes.txt","r") as test_classes:
	test_text = test_classes.read().split(' ')


test_corpus = vectorizer.fit_transform(test_text)
test_corpus = test_corpus.toarray()

knownTestByClass = {}
predictedTestByClass = {}

count1=0 #sliding window
count2=n-1 #sliding window

while(count2<len(test_corpus)-1):
	index = getOneHot(test_corpus[(count2)])
	if index not in knownTestByClass:
		knownTestByClass[index] = []

	if index not in predictedTestByClass:
		predictedTestByClass[index] = []

	count2 += 1
	knownTestByClass[index].append(test_corpus[count1:count2])
	predictedTestByClass[index].append(test_corpus[count2])
	count1 += 1

for i in knownTestByClass:
	knownTestByClass[i] = np.array(knownTestByClass[i])
for i in predictedTestByClass:
	predictedTestByClass[i] = np.array(predictedTestByClass[i])


	

score_list = []
for index in knownTestByClass:
	score = model.evaluate(knownTestByClass[index],predictedTestByClass[index],batch_size=20,verbose=2)
	print("index: {} - accuracy: {}".format(index,score[1]))
	score_list.append(score[1])