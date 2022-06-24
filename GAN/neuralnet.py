import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss

# loads the MNIST Dataset
# class MNIST():
#     def __init__(self, filename):
#         self.raw_data = pd.read_csv(filename)
#         self.l = raw_data['label']
#         self.d = raw_data.drop('label', axis = 1)

class InputLayer():
    def __init__(self, num_inputs, num_neurons):
        self.weights = 0.01 * np.random.randn(num_inputs, num_neurons) # initializes weights as a inputs x neruons size array with numbers close to 0
        self.biases = np.zeros((1, num_neurons)) # initalizes biases as 0s

    def feedforward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Softmax():
    def feed(self, inputs):
        ex = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = ex / np.sum(ex, axis=1, keepdims=True)

class ReLU:
    def feed(self, inputs):
        self.output = np.maximum(0, inputs)

class LossFunction():
    # Categorical cross entropy
    def CCE(self, predicted_value, actual_value):
        # print(predicted_value.shape)
        # print(actual_value.shape)
        # self.loss = -np.log(predicted_value, actual_value)
        # print(self.loss)

        i = len(predicted_value)
        pvclip = np.clip(predicted_value, 1e-7, 1-1e7)
        confidences = pvclip[range(i), actual_value]
        print(confidences)
        # self.loss = -np.log(confidences)
    
    # def calc(self):
    #     lossval = np.mean(self.loss)
    #     return lossval



