import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Using this library to understand pixels/images
from PIL import Image as im

# loads the MNIST Dataset
class MNIST():
    def __init__(self, filename):
        self.raw_data = pd.read_csv(filename)
        self.l = raw_data['label']
        self.d = raw_data.drop('label', axis = 1)

class InputLayer():
    def __init__(self, num_inputs, num_neurons):
        self.weights = 0.01 * np.random.randn(num_inputs, num_neurons) # initializes weights as a inputs x neruons size array with numbers close to 0
        self.biases = np.zeros((1, num_neurons)) # iniitalizes biases as 0s

    def feedforward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Softmax():
    def feed(self, inputs):
        pass

class LossFunction():
    pass


if __name__ == '__main__':
    # test = Layer(3,3)
    # print(test.weights)
    # print("\n")
    # print(test.biases)
    # print("\n")


    # Data 

    # data = MNIST('./train.csv')
    raw_data = pd.read_csv('./train.csv')
    l = raw_data['label']
    d = raw_data.drop('label', axis = 1)

    # print(d.head(3).T) # prints the first 5 rows

    # print(l)

    # MNIST is a data set of 28x28 images
    # Creates a layer with 784 inputs 
    layer0 = InputLayer(784, 10)
    layer0.feedforward(d)
    print(layer0.output.shape)

    df = pd.DataFrame(layer0.output)
    df.to_csv('layer0output.csv')
    # print(layer0.output)

