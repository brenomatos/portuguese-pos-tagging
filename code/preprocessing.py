import pandas as pd
import numpy as np

# classes_list = []
classes_set = set() # this list will keep track of all classes of words
classes = open("words.txt","w")
words = open("classes.txt", "w")
with open("../data/macmorpho-train.txt","r") as f:
    text = f.read()
    words_list = text.split( )
    for word in words_list: # get all different tokens in text
        class_aux = word.split("_")
        classes_set.add(class_aux[1])
        classes.write(str(class_aux[1])+" ")
        words.write(str(class_aux[0])+" ")


classes.close()
words.close()

# dataframe dimensions
columns = ['classes']
index = range(0,len(classes_set))

# creating a new dataframe then passing it to a new one that will have one_hot_encoding
df = pd.DataFrame(index=index, columns=columns, data=list(classes_set))
df.columns = pd.Index(map(lambda x : str(x)[8:], df.columns))
one_hot_encoding = pd.get_dummies(df)
one_hot_encoding.to_csv("one_hot.csv", index=False)

# return every one_hot_encoding as numpy array
# for column in one_hot_encoding:
#     aux = np.array(one_hot_encoding[column])
#     print(column, aux)
