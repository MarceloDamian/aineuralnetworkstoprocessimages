

import dneuralnet as nn
import numpy as np
import pandas as pd

''' Discriminator model'''
def multi_image(data):


    ''' Creates the input layer with 784 inputs each for
    a pixel in a 28x28 image. Then 10 nodes to output for the next layer'''
    dlayer0 = nn.InputLayer(784, 10)
    dlayer0.feedforward(data)

    ''' Activation Layer using the ReLU function and the output from layer 0'''
    dlayer1 = nn.ReLU()
    dlayer1.feed(dlayer0.output)

    df1 = pd.DataFrame(dlayer1.output)
    df1.to_csv('dlayer1output.csv')

    ''' Activation Layer using the Softmax function and the output from layer 1'''
    dlayer2 = nn.Softmax()
    dlayer2.feed(dlayer1.output)

    df2 = pd.DataFrame(dlayer2.output)
    df2.to_csv('dlayer2output.csv')

    print(f'The predicted values {dlayer2.output}')
    print(f'The actual values {label}') 

    ''' Calculate Loss using the output from layer 2 (predicted value) 
    and the label data for actual value '''
    lossfunc = nn.SingleLossFunction() # changed from lossfunction to singlelossfunction
    lossfunc.CCE(dlayer2.output, label) 

if __name__ == '__main__':


    ''' raw data '''
    raw_data = pd.read_csv('./trainminst.csv') # changed to trainminst
    l = raw_data['label']
    d = raw_data.drop('label', axis = 1)

    dfrow = pd.DataFrame(d)
    dfrow.to_csv('drow.csv')

    ''' One hot encode label '''
    dflabel = pd.DataFrame(l)
    one_hot = pd.get_dummies(dflabel['label'])
    dflabel.drop('label', axis = 1)
    dflabel = dflabel.join(one_hot)
    dflabel = dflabel.iloc[: , 1:] # drops the first column
    dflabel.to_csv("dlabel.csv")

    ''' converts to numpy arrays '''
    label = dflabel.to_numpy()
    pixel = dfrow.to_numpy() # convert row dataframe to numpy array
    
    ''' Casts those arrays as int64'''
    label = label.astype(np.int64)
    pixel = pixel.astype(np.int64)

    multi_image(pixel)