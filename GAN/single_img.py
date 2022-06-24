import neuralnet as nn
import numpy as np
import pandas as pd


# 
def single_image(data):
    layer0 = nn.InputLayer(784, 10)
    layer0.feedforward(data)

    layer1 = nn.ReLU()
    layer1.feed(layer0.output)

    df1 = pd.DataFrame(layer1.output)
    df1.to_csv('layer1output.csv')

    layer2 = nn.Softmax()
    layer2.feed(layer1.output)

    df2 = pd.DataFrame(layer2.output)
    df2.to_csv('layer2output.csv')
    print(layer2.output)

    lossfunc = nn.LossFunction()
    lossfunc.CCE(layer2.output, l[0])
    # loss = lossfunc.calc()
    # print(loss)


if __name__ == '__main__':

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

    label = dflabel.to_numpy()
    pixel = dfrow.to_numpy() # convert row dataframe to numpy array

    # print(pixel)
    print(label[0])

    # print(row.T.shape)
    # print(row.T)

    # plt.imshow(row.reshape(28,28), cmap='Greys', interpolation='None') # print the image after reshaping into numpy array
    # plt.show()
    # print(l[0])
    # print(d.head(3).T) # prints the first 5 rows
    # print(l)

    single_image(dfrow)
    '''
    MNIST is a data set of 28x28 images 
    So we need a layer with 784 inputs 
    '''

