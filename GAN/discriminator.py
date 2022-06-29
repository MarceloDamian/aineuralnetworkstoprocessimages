import neuralnet as nn
import numpy as np
import pandas as pd

if __name__ == '__main__':

    layer0 = nn.InputLayer(784, 10)
    layer1 = nn.ReLU()
    layer2 = nn.Softmax()
    lossfunc = nn.MultiLossFunction()

    # ''' raw data '''
    # raw_data = pd.read_csv('./train.csv')
    # l = raw_data['label']
    # d = raw_data.drop('label', axis = 1)

    # dfrow = pd.DataFrame(d)
    # dfrow.to_csv('row.csv')

    # ''' One hot encode label '''
    # dflabel = pd.DataFrame(l)
    # one_hot = pd.get_dummies(dflabel['label'])
    # dflabel.drop('label', axis = 1)
    # dflabel = dflabel.join(one_hot)
    # dflabel = dflabel.iloc[: , 1:] # drops the first column
    # dflabel.to_csv("label.csv")

    # ''' converts to numpy arrays '''
    # label = dflabel.to_numpy()
    # pixel = dfrow.to_numpy() # convert row dataframe to numpy array
    
    # ''' Casts those arrays as int64'''
    # label = label.astype(np.int64)
    # pixel = pixel.astype(np.int64)

    # lowest = 10000000

    # for epoch in range(10):
    #     layer0.weights += 0.05 * np.random.randn(784, 10)
    #     layer0.biases += 0.05 * np.random.randn(1, 10)

    #     layer0.feedforward(pixel)
    #     layer1.feed(layer0.output)
    #     layer2.feed(layer1.output)
    #     lossfunc.CCE(layer2.output, label)

    #     print(f"Loss : {lossfunc.loss}, epoch: {epoch}, weights: {layer0.weights}, biases: {layer0.biases}")

    #     if lossfunc.loss < lowest:
    #         opweights = layer0.weights.copy()
    #         opbiases = layer0.biases.copy()
    #         lowest = lossfunc.loss
    #     else:
    #         layer0.weights = opweights.copy()
    #         layer0.biases = opbiases.copy()

    ''' Testing Data set'''

    ''' Loading test data'''
    test_data = pd.read_csv('./test.csv')

    dftest = pd.DataFrame(test_data)

    ''' converts to numpy arrays '''
    testpixel = dftest.to_numpy() # convert row dataframe to numpy array
    

    ''' Predict the images '''
    layer0.feedforward(testpixel)
    layer1.feed(layer0.output)
    layer2.feed(layer1.output)

    dfresults = pd.DataFrame(layer2.output)
    dfresults.to_csv("pred.csv")
    
    


