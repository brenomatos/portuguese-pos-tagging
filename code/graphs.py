import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Receives a results file path and returns it as a dictionary
def return_dict(file_path):
    df = pd.read_csv(file_path)
    dict = df.set_index('index').T.to_dict('records') # returns a list of dicts
    dict = dict[0] # so we need to get the only dict that matters, the first and only
    return dict

# This method creates a graph comparing every class's accuracy using it's respective results file
def graph_by_class(file_path,window_size,epochs):
    class_result = return_dict(file_path)
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

    # plt.show()
    if not os.path.exists('../results/graphs'):
        os.makedirs('../results/graphs')

    plt.savefig('../results/graphs/accuracy_by_class_'+str(window_size)+"-"+str(epochs)+'.png')

    del keys

def graph_by_window_size():
    quit()
