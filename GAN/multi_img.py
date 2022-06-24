import neuralnet as nn
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # Load Data here
    # data = MNIST('./train.csv')
    raw_data = pd.read_csv('./train.csv')
    l = raw_data['label']
    d = raw_data.drop('label', axis = 1)
    
    layer0 = nn.InputLayer(784, 10)
    layer0.feedforward(d)
    # print(layer0.output.shape)
    
    # Create CSV file for layer0 output
    df0 = pd.DataFrame(layer0.output)
    df0.to_csv('layer0output.csv')
    
    # activate layer 0 data
    layer1 = nn.ReLU()
    layer1.feed(layer0.output)

    # Create CSV file for layer1 output
    df1 = pd.DataFrame(layer1.output)
    df1.to_csv('layer1output.csv')

    # activate layer 1 data
    layer2 = nn.Softmax()
    layer2.feed(layer1.output)

    # Create CSV file for layer2 output
    df2 = pd.DataFrame(layer2.output)
    df2.to_csv('layer2output.csv')