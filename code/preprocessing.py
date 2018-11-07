import pandas as pd


# classes_list = []
classes_set = set() # this list will keep track of all classes of words
with open("../data/macmorpho-train.txt") as f:
    text = f.read()
    words_list = text.split( )
    for word in words_list: # get all different tokens in text
        class_aux = word.split("_")
        classes_set.add(class_aux[1])


# dataframe dimensions
columns = ['classes']
index = range(0,len(classes_set))

# creating a new dataframe then passing it to a new one that will have one_hot_encoding
df = pd.DataFrame(index=index, columns=columns, data=list(classes_set))
one_hot_encoding = pd.get_dummies(df)
