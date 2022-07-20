

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
