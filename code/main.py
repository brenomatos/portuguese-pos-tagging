import numpy as np
from keras import initializers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer
from graphs import graph_by_class
import os

def pre_processing(): #separates the words from its classes
	classes_set = set() # this list will keep track of all classes of words
	classes = open("words.txt","w")
	words = open("classes.txt", "w")
	with open("../data/macmorpho-train.txt","r") as f:
	    text = f.read()
	    words_list = text.split( )
	    for word in words_list: # get all different tokens in text
	        class_aux = word.split("_")
	        classes_set.add(class_aux[1])
	        words.write(str(class_aux[1])+" ")
	        classes.write(str(class_aux[0])+" ")
	classes.close()
	words.close()
	print("Pre-processing Done")


def getOneHot(test_corpus): #returns one-hot index
	for i in range(0,len(test_corpus)):
		if(test_corpus[i]==1):
			return i


def return_training_data(text, window_size, epochs):
	data_train = []
	classes_train = []
	vectorizer = CountVectorizer(lowercase=False, token_pattern='[A-Z;+;-]+')
	corpus = vectorizer.fit_transform(text)
	corpus = corpus.toarray()

	window_start=0 #sliding window
	window_end=window_size-1 #sliding window
	while(window_end<len(corpus)-1):
	    window_end += 1
	    data_train.append(corpus[window_start:window_end])
	    classes_train.append(corpus[window_end])
	    window_start += 1

	data_train = np.array(data_train)
	classes_train = np.array(classes_train)

	return data_train,classes_train,vectorizer,corpus


def return_testing_data(vectorizer, window_size, corpus, text):
	data_test = []
	classes_test = []

	test_corpus = vectorizer.fit_transform(text)
	test_corpus = test_corpus.toarray()

	knownTestByClass = {}
	predictedTestByClass = {}

	window_start=0 #sliding window
	window_end=window_size-1 #sliding window

	while(window_end<len(test_corpus)-1):
	    index = getOneHot(test_corpus[(window_end)])
	    if index not in knownTestByClass:
	        knownTestByClass[index] = []

	    if index not in predictedTestByClass:
	        predictedTestByClass[index] = []

	    window_end += 1
	    knownTestByClass[index].append(test_corpus[window_start:window_end])
	    predictedTestByClass[index].append(test_corpus[window_end])

	    data_test.append(corpus[window_start:window_end])
	    classes_test.append(corpus[window_end])

	    window_start += 1

	for i in knownTestByClass:
	    knownTestByClass[i] = np.array(knownTestByClass[i])
	for i in predictedTestByClass:
	    predictedTestByClass[i] = np.array(predictedTestByClass[i])

	data_test = np.array(data_test)
	classes_test = np.array(classes_test)

	return data_test,classes_test,knownTestByClass,predictedTestByClass


def main(window_size,epochs,batch_size):
	with open("classes.txt","r") as classes:
		text = classes.read().split(' ')
	#generating training samples
	data_train = []
	classes_train = []
	data_train,classes_train,vectorizer,corpus = return_training_data(text, window_size, epochs)


	np.random.seed(7)
	model = Sequential()
	model.add(LSTM(50,input_shape=(window_size,26)))
	model.add(Dense(25,activation='relu'))
	model.add(Dense(2, activation='sigmoid'))
	model.add(Dense(26, activation=lambda x: x))
	model.compile(loss='binary_crossentropy',optimizer ='adam',metrics=['accuracy'])
	model.fit(data_train,classes_train,epochs=epochs, batch_size=batch_size,validation_split=0.2,verbose=1);

    #generating test samples
	data_test = []
	classes_test = []
	data_test,classes_test,knownTestByClass,predictedTestByClass = return_testing_data(vectorizer, window_size, corpus, text)

	result_file_name = str(window_size)+'-'+str(epochs) # stores the LSTM's parameters as string to use as file name

	# check if the total_accuracy file exists
	if os.path.exists("../results/total_accuracy.csv"):
		header_exists = True
	else:
		header_exists = False
	# if it does not exist, save the header
	with open("../results/total_accuracy.csv", "a+") as f:
		if not header_exists:
			f.write("window_size,epochs,accuracy\n")
		f.write(str(window_size)+","+str(epochs)+","+str(model.evaluate(data_test,classes_test,batch_size=batch_size,verbose=2)[1])+"\n")

	with open("../results/"+result_file_name+".csv","w") as f:
		f.write("index,accuracy\n")

	classes_list = vectorizer.get_feature_names() # will be used to return each class's accuracy, but without using an index
	for index in knownTestByClass:
		score = model.evaluate(knownTestByClass[index],predictedTestByClass[index],batch_size=batch_size,verbose=2)
		with open("../results/"+result_file_name+".csv","a") as f:
			f.write(str(classes_list[index])+","+str(score[1])+"\n")

	graph_by_class("../results/"+result_file_name+".csv",window_size,epochs) # generating graphics


pre_processing()
for i in range(3,6):
	main(i,1,8192)

#
# main(3,1,8192)
# main(6,1,8192)
