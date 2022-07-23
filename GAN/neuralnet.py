import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv


class weightsandbiases():
    
    def updateweights(self):

        self.weights = 1.10 # placeholder this would be updated with backprop
        #print (self.weights)
        return self.weights
        
    def updatebias(self):

        self.biases = 1 # placeholder this would be updated with backprop
        #print (self.biases)
        return self.biases


class Zerolayer():

    def firstnodes(self,num_neurons,num_inputs):
        self.nodes = np.random.randint(1,size=(num_neurons, num_inputs)) # not right because it should be reading from train.csv # multiplied by a weight 
        # ceiling is at 1 so it could only be 000000.
        #print (self.nodes)
        return self.nodes

    def replacewithrealarray(self, rnumber): # replaces with one picture from real array 
        
        realarray = self.nodes
        
        with open('./newreorg.csv', 'r') as csv_file:
            csvreader = csv.reader(csv_file)
            next(csvreader) # skips pixelnumbers skips 0 
            #label = data[1] # can be used for train.csv to read each set of pixels as a label             
            
            for index, row in enumerate(csvreader):
               if index==rnumber:
                pixels = row[1:]
    
            for i in range(len(realarray)):
                self.realnest=realarray[i]
            
            for i in range(len(self.realnest)):
                self.realnest[i] = pixels[i]

            #print (self.realnest)
            return self.realnest

    def replacerowwithrand(self, rnumber): # replace with row from random test . csv 

        arraybefore = self.nodes
 
        with open('./faketest.csv', 'r') as csv_file:
            csvreader = csv.reader(csv_file)
            next(csvreader) # skips pixelnumbers skips 0 

            for index, row in enumerate(csvreader):
               if index==rnumber:
                pixels = row
                break
                        
            for i in range(len(arraybefore)):
                self.insidenest=arraybefore[i]
            
            for i in range(len(self.insidenest)):
                self.insidenest[i] = pixels[i]

        return self.insidenest
            
    def replacerandwithwandbias(self): #weight, bias):  # replaces rand with rand * weights and + bias 
        
        #Generates new random file everytime you run it
        #############################
        #data1=np.random.uniform(size=(784,10),low = -0.99, high= 0.99) 

        #with open('L1toL2constantweights.csv', 'w', encoding='UTF8', newline='') as func:
        #    writer = csv.writer(func)

        # write multiple rows
        #    writer.writerows(data1)
            ########## this along with sort (reorgconstantweights.csv) to finally enumerate ###
        ###########################

        pslist = self.insidenest
        dot = 0
        weight = []
        data = []
        nesteddata = []

        header = ['wnode0,wnode1,wnode2,wnode3,wnode4,wnode5,wnode6,wnode7,wnode8.wnode9,delimiter']

        with open('./enumeratedconstantweights.csv', 'r') as csv_file:
            csvreader = csv.reader(csv_file)
            next(csvreader) # skips pixelnumbers skips 0 

            for index, row in enumerate(csvreader):
               if index==0:
                weight.append(row[1:])
                #print(weight)
                break
    
            realweight = weight[0]

            for i in range(len(pslist)):
                for k in range (10): # this is for the hidden layer
                    dot=int(pslist[i]) * float(realweight[k]) # the starter weight #+ bias         #times the weight. 
                    data.append(dot)
                data.append("\n")

            nesteddata = [data]
               
        with open('layer1nodedweights.csv', 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            
            # write multiple rows
            writer.writerow(header)

            # write multiple rows
            writer.writerows(nesteddata)
            
        return nesteddata

    def summedoutputofanyarray(self, bias, array): # adds bias 
  
        storelist = array
        listwithweights = 0

        for i in range(len(storelist)):
            listwithweights += storelist[i] 
        
        self.accum = listwithweights + bias # 1 is the bias that # straight line to linear line.

        return self.accum

        # have it read in nested list and insert the input from there  and find dot product and then add biases
    
    def averageofsupersumMSE (self, neuron):

        with open('./sumdp.csv', 'r') as csv_file:
            csvreader = csv.reader(csv_file)

            for data in csvreader:
                #print (f'data [0]  {data[0]}')
                #print (f'neuron [0]  {neuron}')
                if data[0]==str(neuron) and str(neuron)=='0':
                    avgdot = int(data[1]) / 4132
                    print (avgdot)
                    return avgdot
                elif data[0]==str(neuron) and str(neuron)=='1':                        
                    avgdot = int(data[1]) / 4684
                    print (avgdot)
                    return avgdot
                elif data[0]==str(neuron) and str(neuron)=='2':                        
                    avgdot = int(data[1]) / 4177
                    print (avgdot)
                    return avgdot
                elif data[0]==str(neuron) and str(neuron)=='3':                        
                    avgdot = int(data[1]) / 4351
                    print (avgdot)    
                    return avgdot
                elif data[0]==str(neuron) and str(neuron)=='4':                        
                    avgdot = int(data[1]) / 4072
                    print (avgdot)
                    return avgdot
                elif data[0]==str(neuron) and str(neuron)=='5':   
                    avgdot = int(data[1]) / 3795
                    print (avgdot)
                    return avgdot
                elif data[0]==str(neuron) and str(neuron)=='6':                        
                    avgdot = int(data[1]) / 4137
                    print (avgdot)
                    return avgdot
                elif data[0]==str(neuron) and str(neuron)=='7':                        
                    avgdot = int(data[1]) / 4401
                    print (avgdot)
                    return avgdot
                elif data[0]==str(neuron) and str(neuron)=='8':                        
                    avgdot = int(data[1]) / 4063
                    print (avgdot)
                    return avgdot
                elif data[0]==str(neuron) and str(neuron)=='9':                        
                    avgdot = int(data[1]) / 4188
                    print (avgdot)
                    return avgdot

    def activaterowrelu(self, array):  

        storelist = array  #self.output

        newoutput = []

        for i in range(784):
            #print (storelist[i])
            newoutput.append (np.maximum(0, storelist[i]))

        print (newoutput)

    def derivativeofrelu(self,array):  

        storelist = array  #self.output
        newoutput = []

        for i in range(784):
            #print (storelist[i])
            if storelist[i] > 0:
                newoutput.append(1)

            elif storelist[i] <= 0:
                newoutput.append(0)

        print (newoutput)


class Firstlayer():
    
    def sumofeachnodeinfirstlayer (self): 
        
        self.allaccumlist= []
        
        bias = [] # also has to learn 

        with open('./L0toL1constantbiases.csv', 'r') as csv_file:
            csvreader = csv.reader(csv_file)
            next(csvreader) # skips pixelnumbers skips 0 
            for data in csvreader:
                bias=data 

            #print (bias)

        with open('./layer0tolayer1nodedweights.csv', 'r') as csv_file:
            csvreader = csv.reader(csv_file)
            next(csvreader) # skips pixelnumbers skips 0 

            for data in csvreader: #0,11,22,33,..110 so row times 11
                for node in range (10):
                    accum=0
                    for index in range(784):
                        accum +=float (data[(11*index)+ node])
                    self.allaccumlist.append(accum + float (bias[node]))                    
       
        return self.allaccumlist
    

    # 784 nodes (1 per pixel) to 1 node(784 pixels) to generate images 
    # 1 node(784 pixels) to 784 nodes(1 per pixel) then compare the dot product to actual real ones

    def RELU(self):

        storelist = self.allaccumlist 
        self.output = []

        for i in range(10):
            self.output.append (np.maximum(0, storelist[i]))
        return self.output
    

    def layer1tolayer2connections(self):
        
        #self.output is relu output 
        self.accumulated = []
        #summedup = 0

        bias = [] # also has to learn 

        with open('./L1toL2constantbiases.csv', 'r') as csv_file:
            csvreader = csv.reader(csv_file)
            next(csvreader) # skips pixelnumbers skips 0 
            for data in csvreader:
                bias=data 

            print (f'this is the bias:::: {bias}')
        
        with open('./L1toL2constantweights.csv', 'r') as csv_file:
            csvreader = csv.reader(csv_file)

            for l2node, weight in enumerate(csvreader):
                for increment in range (10):
                    if l2node==increment:
                        summedup = 0
                        for i in range (10):
                            #print(f'Row{ self.output[i],weight[i], bias[i]}')
                            summedup += float(self.output[i]) * float (weight[i]) 
                        self.accumulated.append (summedup + float (bias [increment]))
                    
            #print (f'Layer2 nodes: {self.accumulated}')

        return self.accumulated            

    def Softmax(self):
        
        reluonsum=self.accumulated
        e_x = np.exp(reluonsum - np.max(reluonsum))
        return e_x / e_x.sum()





#class Secondlayer():
    
  
        
        #return np.exp(x) / np.sum(np.exp(x), axis=0)

#        ex = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
#        self.output = ex / np.sum(ex, axis=1, keepdims=True)

#class PDSoftmax():
#class PDReLU():








        


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
    #class secondoptimizer():

#    def sumofsquareresiduals_gradient(self, inputdata, outputdata, b):
#        res = b[0] + b[1] * inputdata - outputdata  # 7 + 8 (4)-8 = 20 , 31 mean = 25.5 , 
        #print (res)
#        return res.mean(), (res * inputdata).mean()  # .mean() is a method of np.ndarray

#    def sgd(self,gradient, inputdata, outputdata, start, learn_rate=0.1, batch_size=1, n_iter=50,tolerance=1e-06, dtype="float64", random_state=None):
    
        # Checking if the gradient is callable
#        if not callable(gradient):
#            raise TypeError("'gradient' must be callable")

        # Setting up the data type for NumPy arrays
#        dtype_ = np.dtype(dtype)

        # Converting inputdata and outputdata to NumPy arrays
#        inputdata, outputdata = np.array(inputdata, dtype=dtype_), np.array(outputdata, dtype=dtype_)
#        n_obs = inputdata.shape[0]
#        if n_obs != outputdata.shape[0]:
#            raise ValueError("'inputdata' and 'outputdata' lengths do not match")
#        inputdataoutputdata = np.c_[inputdata.reshape(n_obs, -1), outputdata.reshape(n_obs, 1)]

        # Initializing the random number generator
#        seed = None if random_state is None else int(random_state)
#        rng = np.random.default_rng(seed=seed)

        # Initializing the values of the variables
#        vector = np.array(start, dtype=dtype_)

        # Setting up and checking the learning rate
#        learn_rate = np.array(learn_rate, dtype=dtype_)
#        if np.any(learn_rate <= 0):
#            raise ValueError("'learn_rate' must be greater than zero")

        # Setting up and checking the size of minibatches
#        batch_size = int(batch_size)
#        if not 0 < batch_size <= n_obs:
#            raise ValueError(
#                "'batch_size' must be greater than zero and less than "
#                "or equal to the number of observations"
#            )

        # Setting up and checking the maximal number of iterations
#        n_iter = int(n_iter)
#        if n_iter <= 0:
#            raise ValueError("'n_iter' must be greater than zero")

        # Setting up and checking the tolerance
#        tolerance = np.array(tolerance, dtype=dtype_)
#        if np.any(tolerance <= 0):
#            raise ValueError("'tolerance' must be greater than zero")


        # Performing the gradient descent loop
#        for _ in range(n_iter):
            # Shuffle inputdata and outputdata
#            rng.shuffle(inputdataoutputdata)  

            # Performing minibatch moves
#            for start in range(0, n_obs, batch_size):
#                stop = start + batch_size
#                inputdata_batch, outputdata_batch = inputdataoutputdata[start:stop, :-1], inputdataoutputdata[start:stop, -1:]

                # Recalculating the difference
#                grad = np.array(gradient(inputdata_batch, outputdata_batch, vector), dtype_)
#                diff = -learn_rate * grad

                # Checking if the absolute difference is small enough
#                if np.all(np.abs(diff) <= tolerance):
#                    break

                # Updating the values of the variables
#                vector += diff

#            print (vector)

#        return vector if vector.shape else vector.item()







