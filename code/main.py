import numpy as np
from keras import initializers
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM


np.random.seed(7)
model = Sequential()
model.add(LSTM(50,input_shape=(n,num_ativos)))
model.add(Dense(25,activation='relu'))
model.add(Dense(2, activation='sigmoid'))
#model.add(Dense(2, activation=lambda x: x))
model.compile(loss='binary_crossentropy',optimizer ='RMSprop',metrics=['accuracy'])
model.fit(data_train,classes_train,epochs=100, batch_size=len(data_train),validation_split=0.2,verbose=0);
scores = model.evaluate(data_test,classes_test,batch_size=len(data_test))
predict = model.predict(data_test,batch_size=len(data_test),verbose=0)
print('Accurracy: {}'.format(scores[1]))
