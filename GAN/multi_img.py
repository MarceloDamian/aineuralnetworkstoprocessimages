import neuralnet as nn
import numpy as np
import pandas as pd

''' Discriminator model'''
def multi_image(data):
    
    ''' Creates the input layer with 784 inputs each for
    a pixel in a 28x28 image. Then 10 nodes to output for the next layer'''
    layer0 = nn.InputLayer(784, 10)
    layer0.feedforward(data)

    ''' Activation Layer using the ReLU function and the output from layer 0'''
    layer1 = nn.ReLU()
    layer1.feed(layer0.output)

    df1 = pd.DataFrame(layer1.output)
    df1.to_csv('layer1output.csv')

    ''' Activation Layer using the Softmax function and the output from layer 1'''
    layer2 = nn.Softmax()
    layer2.feed(layer1.output)

    df2 = pd.DataFrame(layer2.output)
    df2.to_csv('layer2output.csv')

    print(f'The predicted values {layer2.output}')
    print(f'The actual values {label}')

    ''' Calculate Loss using the output from layer 2 (predicted value) 
    and the label data for actual value '''
    lossfunc = nn.LossFunction()
    lossfunc.CCE(layer2.output, label)

if __name__ == '__main__':


    ''' raw data '''
    raw_data = pd.read_csv('./train.csv')
    l = raw_data['label']
    d = raw_data.drop('label', axis = 1)

    dfrow = pd.DataFrame(d)
    dfrow.to_csv('row.csv')

    ''' One hot encode label '''
    dflabel = pd.DataFrame(l)
    one_hot = pd.get_dummies(dflabel['label'])
    dflabel.drop('label', axis = 1)
    dflabel = dflabel.join(one_hot)
    dflabel = dflabel.iloc[: , 1:] # drops the first column
    dflabel.to_csv("label.csv")

    ''' converts to numpy arrays '''
    label = dflabel.to_numpy()
    pixel = dfrow.to_numpy() # convert row dataframe to numpy array
    
    ''' Casts those arrays as int64'''
    label = label.astype(np.int64)
    pixel = pixel.astype(np.int64)

    multi_image(pixel)
