import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
def read_data(file_path):
    df = pd.read_csv(file_path)
    # Make fake dataset
    classes = list(df['index'])
    accuracy = list(df['accuracy'])
    y_pos = np.arange(len(classes))

    # Create horizontal bars
    plt.barh(y_pos, accuracy)

    # Create names on the y-axis
    plt.yticks(y_pos, classes)

    # Show graphic
    plt.show()


read_data('../results/3-1.csv')
