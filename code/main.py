import numpy as np
from keras import initializers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer


def getOneHot(test_corpus):
	for i in range(0,len(test_corpus)):
		if(test_corpus[i]==1):
			return i


n = 10 # window size
epochs = 1
batch_size = 20
def main(n,epochs,batch_size):
    with open("classes.txt","r") as classes:
        text = classes.read().split(' ')
    # 

    data_train = []
    classes_train = []
    vectorizer = CountVectorizer(lowercase=False, token_pattern='[A-Z;+;-]+')
    corpus = vectorizer.fit_transform(text)
    corpus = corpus.toarray()
    with open("features_"+str(n)+'-'+str(epochs)+".txt","w") as f:
        f.write(str(vectorizer.get_feature_names()))
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
    # print(data_train.shape)
    # print(classes_train.shape)
    # print(len(data_train))
    # 
    # 
    np.random.seed(7)
    model = Sequential()
    model.add(LSTM(50,input_shape=(n,26)))
    model.add(Dense(25,activation='relu'))
    model.add(Dense(2, activation='sigmoid'))
    model.add(Dense(26, activation=lambda x: x))
    model.compile(loss='binary_crossentropy',optimizer ='adam',metrics=['accuracy'])
    model.fit(data_train,classes_train,epochs=epochs, batch_size=batch_size,validation_split=0.2,verbose=1);
    # callbacks=[EarlyStopping(min_delta=0.00025, patience=2)] entra no model.fit

    #generating test samples
    data_test = []
    classes_test = []

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

        data_test.append(corpus[count1:count2])
        classes_test.append(corpus[count2])

        count1 += 1

    for i in knownTestByClass:
        knownTestByClass[i] = np.array(knownTestByClass[i])
    for i in predictedTestByClass:
        predictedTestByClass[i] = np.array(predictedTestByClass[i])

    data_test = np.array(data_test)
    classes_test = np.array(classes_test)

    with open("total_acuracy_"+str(n)+'-'+str(epochs)+".txt","w") as f:
        f.write(str(model.evaluate(data_test,classes_test,batch_size=batch_size,verbose=2)))
        
    with open(str(n)+'-'+str(epochs)+".csv","w") as f:
            f.write("index,accuracy\n")

    score_list = []
    for index in knownTestByClass:
        score = model.evaluate(knownTestByClass[index],predictedTestByClass[index],batch_size=batch_size,verbose=2)
        with open(str(n)+'-'+str(epochs)+".csv","a") as f:
            f.write(str(index)+","+str(score[1])+"\n")
        # print("index: {} - accuracy: {}".format(index,score[1]))
        score_list.append(score[1])


#for i in range(5,30,5):
 #   main(i,15,20)

main(3,15,20)
main(4,15,20)
