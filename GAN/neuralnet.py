import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Using this library to understand pixels/images
from PIL import Image as im

class Layer():
    def __init__(self, num_inputs, num_neurons):
        self.weights = 0.01 * np.random.randn(num_inputs, num_neurons) # initializes weights as a inputs x neruons size array with numbers close to 0
        self.biases = np.zeros((1, num_neurons)) # iniitalizes biases as 0s

    def feedforward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class ActivationFunction():
    pass

class LossFunction():
    pass


if __name__ == '__main__':
    test = Layer(3,3)
    print(test.weights)
    print("\n")
    print(test.biases)
    print("\n")


    # Data 

    data = pd.read_csv('./train.csv')

