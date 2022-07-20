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


class firstlayer():

    def firstnodes(self,num_neurons,num_inputs):
        self.nodes = np.random.randint(1,size=(num_neurons, num_inputs)) # not right because it should be reading from train.csv # multiplied by a weight 
        # ceiling is at 1 so it could only be 000000.
        #print (self.nodes)
        return self.nodes

    def replacewithreal(self, rnumber):
        
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



    def replacerowrand(self, rnumber): # for amount of rows. have int as a parameter

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

            #print(insidenest)
        return self.insidenest
            
    
    def replacewithw(self, weight):#, rnumber): # for amount of rows. have int as a parameter
        
        pslist = self.insidenest
        self.output = []
        dot = 0

        for i in range( len(pslist)):
            dot = pslist[i] * weight         #times the weight. 
            self.output.append(dot)
            
        #print (f"self output:   {self.output}") #to debug
        return self.output


    def dotoutput(self,  bias, array): # adds bias 
  
        storelist = array
        listwithweights = 0

        for i in range(len(storelist)):
            listwithweights += storelist[i] 
        
        self.accum = listwithweights + bias # 1 is the bias that # straight line to linear line.

        return self.accum

        # have it read in nested list and insert the input from there  and find dot product and then add biases
    

    def averagedot (self, neuron):

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



    # 784 nodes (1 per pixel) to 1 node(784 pixels) to generate images 
    # 1 node(784 pixels) to 784 nodes(1 per pixel) then compare the dot product to actual real ones

    
#class Softmax():
#    def feed(self, inputs):

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







