import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# loads the MNIST Dataset
class MNIST():
    def __init__(self, filename):
        self.raw_data = pd.read_csv(filename)
        self.l = raw_data['label']
        self.d = raw_data.drop('label', axis = 1)

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
        i = len(predicted_value)
        pvclip = np.clip(predicted_value, 1e-7, 1-1e7)
        confidences = pvclip[range(i), actual_value]
        print(confidences)
        # self.loss = -np.log(confidences)
    
    # def calc(self):
    #     lossval = np.mean(self.loss)
    #     return lossval



if __name__ == '__main__':
    # test = Layer(3,3)
    # print(test.weights)
    # print("\n")
    # print(test.biases)
    # print("\n")

    # Load Data here
    # data = MNIST('./train.csv')
    raw_data = pd.read_csv('./train.csv')
    l = raw_data['label']
    d = raw_data.drop('label', axis = 1)


    '''
    Isolate one image to work with that for now 
    to make sure outputs are correct
    '''

    row = d.iloc[:1, :]
    # print(row)
    dfrow = pd.DataFrame(row)
    dfrow.to_csv('row.csv')

    # convert row dataframe to numpy array
    row = dfrow.to_numpy()

    # print(row.T.shape)
    # print(row.T)

    # plt.imshow(row.reshape(28,28), cmap='Greys', interpolation='None') # print the image after reshaping into numpy array
    # plt.show()
    # print(l[0])
    # print(d.head(3).T) # prints the first 5 rows
    # print(l)

    def single_image(data):
        layer0 = InputLayer(784, 10)
        layer0.feedforward(data)

        layer1 = ReLU()
        layer1.feed(layer0.output)

        df1 = pd.DataFrame(layer1.output)
        df1.to_csv('layer1output.csv')

        layer2 = Softmax()
        layer2.feed(layer1.output)

        df2 = pd.DataFrame(layer2.output)
        df2.to_csv('layer2output.csv')

        lossfunc = LossFunction()
        lossfunc.CCE(layer2.output, l[0])
        # loss = lossfunc.calc()
        # print(loss)

    single_image(dfrow)
    '''
    MNIST is a data set of 28x28 images 
    So we need a layer with 784 inputs 
    '''

    # layer0 = InputLayer(784, 10)
    # layer0.feedforward(d)
    # # print(layer0.output.shape)
    
    # # Create CSV file for layer0 output
    # df0 = pd.DataFrame(layer0.output)
    # df0.to_csv('layer0output.csv')
    
    # # activate layer 0 data
    # layer1 = ReLU()
    # layer1.feed(layer0.output)

    # # Create CSV file for layer1 output
    # df1 = pd.DataFrame(layer1.output)
    # df1.to_csv('layer1output.csv')

    # # activate layer 1 data
    # layer2 = Softmax()
    # layer2.feed(layer1.output)

    # # Create CSV file for layer2 output
    # df2 = pd.DataFrame(layer2.output)
    # df2.to_csv('layer2output.csv')
