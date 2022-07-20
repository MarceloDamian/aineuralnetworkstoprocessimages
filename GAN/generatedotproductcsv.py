
import neuralnet as nn
import numpy as np
import pandas as pd
import csv
import random


classfirst= nn.firstlayer()  #layer0 = nn.InputLayer(42000, 784) #layer0 = nn.InputLayer(784, 10) # danwrote
arrayzero = classfirst.firstnodes(1,784)


summeddotproduct = 0
header = ['dotproducts']
alldotprod = []



for i in range (37812,42000):
    realarr = classfirst.replacewithreal(i)# change later on will be added a for loop to map thru all 42000
#    print (realarr)

    realdotprod = classfirst.dotoutput(1,realarr) # bias added and realarray
    #print (f"dotproduct of the real first one with added bias {realdotprod}\n")
    alldotprod.append (realdotprod)
    #print (alldotprod)

    summeddotproduct+=realdotprod
    print (f"index : {i} summeddotprod: {summeddotproduct}")

data=[alldotprod]



with open('tenthdotproduct.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(data)
