

import neuralnet as nn
import numpy as np
import pandas as pd
import csv
import random



layer0= nn.firstlayer()  #layer0 = nn.InputLayer(42000, 784) #layer0 = nn.InputLayer(784, 10) # danwrote

arrayzero = layer0.firstnodes(1,784)
#print (arrayzero)

#basically the actual row - 2
#0 - 4131

realarr = layer0.replacewithreal(0)# change later on will be added a for loop to map thru all 42000
print (realarr)

layer0.activaterowrelu(realarr)
layer0.derivativeofrelu(realarr)


#realdotprod = layer0.dotoutput(0,realarr) # bias added and realarray
#print (f"dotproduct of the real first one with added bias {realdotprod}\n")
 


randomarr = layer0.replacerowrand(0)# replacerow later on will be added a for loop to map thru all 42000
print (randomarr)


arraywithwe = layer0.replacewithw(1)# weight input
print (arraywithwe)




#dotprod = layer0.dotoutput(1,arraywithwe) # bias added and randarraywithwe
#print (f"\ndotproduct of the first one with added bias {dotprod}")

avgdot  = layer0.averagedot(9)
#print (f"\nAverage dot {avgdot}")


layer0.activaterowrelu(arraywithwe)
layer0.derivativeofrelu(arraywithwe)




#layer1 = nn.ReLU()












#  ################# to change data with a different bias could also be used to change weight ##############################

#reviseddata =[]
#num= 0

#with open('./alldotprod.csv', 'r') as csv_file:
#    csvreader = csv.reader(csv_file)
#    header =  next(csvreader)

#    for row in csvreader:
#        num = list(map(int, row))

#        totalcolumns = 0
#        while totalcolumns != len(num):
#            reviseddata.append (num[totalcolumns] - 1)
#            totalcolumns+=1
#        reviseddata.append([]) # read csv file and split whenever these are seen , also increment by 1 the label 
#    print (reviseddata)
#    data = [reviseddata]

#with open('finalalldotprod.csv', 'w', encoding='UTF8', newline='') as f:
    
#    writer = csv.writer(f)
    # write the header
#    writer.writerow(header)
    # write multiple rows
#    writer.writerows(data)


#  ################# to change data with a different bias could also be used to change weight ##############################














#nn.weightsandbiases.updatebias() # prints out 1 from n.n.
#nn.weightsandbiases.updateweights() # prints out 0.001 for weight in nn

#nn.feedingforward.feedforward()



#layer1 = nn.ReLU()
#layer2 = nn.Softmax()
#lossfunc = nn.MultiLossFunction()

#    ''' raw data '''
#    raw_data = pd.read_csv('./train.csv')
#    l = raw_data['label']
#    
#    d = raw_data.drop('label', axis = 1) #### to create row.csv not needed to do. as you can just skip it over during your functions.
#    dfrow = pd.DataFrame(d)
#    dfrow.to_csv('row.csv')    # row.csv ist just train.csv without the first column 

#    ''' One hot encode label '''
#    dflabel = pd.DataFrame(l)
#    one_hot = pd.get_dummies(dflabel['label'])
#    dflabel.drop('label', axis = 1)
#    dflabel = dflabel.join(one_hot)
#    dflabel = dflabel.iloc[: , 1:] # drops the first column
#    dflabel.to_csv("label.csv")

#    ''' converts to numpy arrays '''
#    label = dflabel.to_numpy()
#    pixel = dfrow.to_numpy() # convert row dataframe to numpy array
    
#    ''' Casts those arrays as int64'''
#    label = label.astype(np.int64)
#    pixel = pixel.astype(np.int64)

#    lowest = 10000000

#    arrayofnodes = [] # empty array 

#    for epoch in range(1):
#        layer0.weights += 0.05 * np.random.randint(784, 10)
#        layer0.biases += 0.05 * np.random.randint(1, 10)

#        arrayofnodes.append(layer0.weights) # for generatornoise
#        arrayofnodes.append(layer0.biases)  # for generatornoise

#        layer0.feedforward(pixel)
#        layer1.feed(layer0.output)
#        layer2.feed(layer1.output)
#        lossfunc.CCE(layer2.output, label)

#        print(f"Loss : {lossfunc.loss}, epoch: {epoch}, weights: {layer0.weights}, biases: {layer0.biases}")

#        if lossfunc.loss < lowest:
#            opweights = layer0.weights.copy()
#            opbiases = layer0.biases.copy()
#            lowest = lossfunc.loss
#        else:
#            layer0.weights = opweights.copy()
#            layer0.biases = opbiases.copy()

#    ''' Testing Data set'''
#    print (arrayofnodes)

#    ''' Loading test data'''
#    test_data = pd.read_csv('./train.csv')
#    test_data = raw_data.drop('label', axis = 1)

#    dftest = pd.DataFrame(test_data)

#    ''' converts to numpy arrays '''
#    testpixel = dftest.to_numpy() # convert row dataframe to numpy array
    

#    ''' Predict the images '''
#    layer0.feedforward(testpixel)
#    layer1.feed(layer0.output)
#    layer2.feed(layer1.output)

#    dfresults = pd.DataFrame(layer2.output)
#    dfresults.to_csv("pred.csv")
    
#    return arrayofnodes # to return array to generator
