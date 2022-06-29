import neuralnet as nn
import numpy as np
import pandas as pd

''' Discriminator model'''
def single_image(data):
    
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

    print(f'The predicted values {layer2.output[0]}')
    print(f'The actual values {label[0]}')

    ''' Calculate Loss using the output from layer 2 (predicted value) 
    and the label data for actual value '''
    singlelossfunc = nn.SingleLossFunction()
    singlelossfunc.CCE(layer2.output[0], label[0])

    # loss = lossfunc.calc()
    # print(loss)


if __name__ == '__main__':

    ''' raw data '''
    raw_data = pd.read_csv('./train.csv')
    l = raw_data['label']
    d = raw_data.drop('label', axis = 1)
    
    ''' Isolate one image to work with that for now to make sure outputs are correct '''
    row = d.iloc[:1, :]
    dfrow = pd.DataFrame(row)
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

    # print(row.T)

    # plt.imshow(row.reshape(28,28), cmap='Greys', interpolation='None') # print the image after reshaping into numpy array
    # plt.show()
    # print(l[0])
    # print(d.head(3).T) # prints the first 5 rows
    # print(l)

    single_image(pixel)
    '''
    MNIST is a data set of 28x28 images 
    So we need a layer with 784 inputs 
    '''

