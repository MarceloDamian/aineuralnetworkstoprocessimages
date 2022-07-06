
#finished
import numpy as np
from random import randrange
import csv
import random

header = ['pixelnumbers']
tupleobj= ()

for num in range(28):
    tupleobj = tupleobj + (randrange(256),)

#for z in range (3): # for the amount of rows
    #row = () + (randrange(256) ,randrange(256) ,randrange(256))   # can be made to color or hexcolor here.. I made it pink to test out grayscale
    #data = [row * 261 + (123,)]#[ 9 * row + tupleobj  ]  #  times 9 plus 28 and then times 28,000 rows to repicate
    #z+=1
    #data = data + data    

row = ()              # row = tuples 
for x in range(28): # x values 0 to 255 1
    row = row + (randrange(256),randrange(256),randrange(256))   # can be made to color or hexcolor here.. I made it pink to test out grayscale
data = [ 9 * row + tupleobj]  #  times 9 plus 28 and then times 28,000 rows to repicate


for z in range(27999): # add 1 to get the total as the previous for loop added a line
    x = data 
    y = list(x[0]) # changes dataype to change tuples that are immutable
    #print(x[0])
    for i in range(len(y)):
        y[i] = randrange(256) # changes each individual element 
    x = tuple(y) 
    n = [x]
    data = data + n

#print (data)

with open('faketest.csv', 'w', encoding='UTF8', newline='') as f:
#with open('newimage.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(data)

