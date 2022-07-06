
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd

#pixels to display 


with open('./test.csv', 'r') as csv_file:#open('./faketest.csv', 'r') as csv_file: #open('./test.csv', 'r') as csv_file: # for the actual data set
    csvreader = csv.reader(csv_file)
    next(csvreader) # ignores first line
    for data in csvreader:
        
        # The first column is the label
        label = data[0]
        data1 = pd.read_csv('./test.csv', nrows=0)#data1 = pd.read_csv('./faketest.csv', nrows=0) #data1 = pd.read_csv('./test.csv', nrows=0) # for the actual data set

        # The rest of columns are pixels
        pixels = data[0:]

        # Make those columns into a array of 8-bits pixels
        # This array will be of 1D with length 784
        # The pixel intensity values are integers from 0 to 255
        pixels = np.array(pixels, dtype = 'int64')
        print(pixels.shape)
        # Reshape the array into 28 x 28 array (2-dimensional array)
        pixels = pixels.reshape((28, 28))
        print(pixels.shape)
        # Plot
        #plt.title('Label is {label}'.format(label=label))
        plt.imshow(pixels, cmap='gray')
        plt.show()


