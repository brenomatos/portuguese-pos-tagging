import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Receives a results file path and returns it as a dictionary
def return_dict(df):
    index = df.columns.values[0] # indexing using the first column of any giver dataframe
    dict = df.set_index(index).T.to_dict('records') # returns a list of dicts
    dict = dict[0] # so we need to get the only dict that matters, the first and only
    return dict

# This method creates a graph comparing every class's accuracy using it's respective results file
def graph_by_class(file_path,window_size,epochs):
    df = pd.read_csv(file_path)
    class_result = return_dict(df)
    plt.rcdefaults()
    fig, ax = plt.subplots()
    keys = [(k) for k,v in sorted(class_result.items())]
    y_pos = np.arange(len(np.array(keys)))
    ax.barh(y_pos, np.array([(v) for k,v in sorted(class_result.items())]), align='center', color='blue', ecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(np.array(keys))
    ax.invert_yaxis()
    ax.set_xlabel('Accuracy')
    ax.set_title('Accuracy by Class')
    ax.grid(True)

    plt.xlim(0.92, 1.00005)
    if not os.path.exists('../results/graphs'):
        os.makedirs('../results/graphs')

    plt.savefig('../results/graphs/accuracy_by_class_'+str(window_size)+"-"+str(epochs)+'.png')
    del keys

# Creates one graph representing data stored in the "total_accuracy.csv" file.
def graph_by_window_size(file_path):
    df = pd.read_csv(file_path)
    df = df[["window_size", "accuracy"]]
    total_accuracy = return_dict(df)
    plt.title("Window Size x Accuracy")
    # plt.grid(True)
    plt.xlabel("Window Size")
    plt.ylabel("Accuracy")
    width = 1/3
    plt.bar(list(total_accuracy.keys()), list(total_accuracy.values()), width)

    if not os.path.exists('../results/graphs'):
        os.makedirs('../results/graphs')
    plt.savefig('../results/graphs/total_accuracy.png')


# graph_by_window_size("../results/total_accuracy.csv")
# graph_by_class("../results/7-1.csv",3,1)
