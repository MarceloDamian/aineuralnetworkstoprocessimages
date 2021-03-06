import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.metrics import log_loss

# loads the MNIST Dataset
# class MNIST():
#     def __init__(self, filename):
#         self.raw_data = pd.read_csv(filename)
#         self.l = raw_data['label']
#         self.d = raw_data.drop('label', axis = 1)

class InputLayer():
    def __init__(self, num_inputs, num_neurons):
        self.weights = 0.01 * np. random.randn(num_inputs, num_neurons) # initializes weights as a inputs x neruons size array with numbers close to 0
        self.biases = np.zeros((1, num_neurons)) # initalizes biases as 0s

    def feedforward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def optimize(self):
        self.weights

class Softmax():
    def feed(self, inputs):
        ex = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = ex / np.sum(ex, axis=1, keepdims=True)

class ReLU:
    def feed(self, inputs):
        self.output = np.maximum(0, inputs)

''' Only works for measuring loss for a single image'''
class SingleLossFunction():
    # Categorical cross entropy
    def CCE(self, predicted_value, actual_value):
        self.loss = -np.sum(actual_value * np.log(predicted_value + 10**-100))
        print(self.loss)
        
class MultiLossFunction():
    # Categorical cross entropy
    def CCE(self, predicted_value, actual_value):
        self.loss = -np.sum(actual_value * np.log(predicted_value + 10**-100))
        self.loss = self.loss / len(predicted_value)
        print(self.loss)


# class Optimizer():
#     def __init__(self, inputlayer):
#         self.learning_rate = 0.001
#         self.opweights = inputlayer.weights
#         self.opbiases = inputlayer.biases

#     def gradientdescent(self, loss):
#         self.opweights = -(2/n)*sum()
        

    #     n = len(predicted_value) # the number of inputs
    #     print(f'This is the number of inputs {n}')
    #     # for entry in predicted_value:
    #     # pvclip = np.clip(predicted_value[0], 1e-7, 1-1e7) # try to compensate for infinite loss by clipping values to a low number not 0
    #     print(predicted_value)
    #     # print(pvclip)
    #     # confidences = pvclip[range(n), actual_value]
    #     # print(f"This is the confidences: {confidences}")
    #     # self.loss = -np.log(predicted_value, actual_value)
    #     # print(f'The loss: {self.loss}')

    #     # i = len(predicted_value)
    #     # pvclip = np.clip(predicted_value, 1e-7, 1-1e7)
    #     # confidences = pvclip[range(i), actual_value]
    #     # print(confidences)
    #     # self.loss = -np.log(confidences)
    
    # # def calc(self):
    # #     lossval = np.mean(self.loss)
    # #     return lossval



