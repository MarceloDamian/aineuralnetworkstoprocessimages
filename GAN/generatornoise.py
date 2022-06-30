
import png
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from random import randrange
from PIL import Image
import numpy as np
import discriminator as dd
#import os

def grayscalenoise(arrayofnodes) : # (arrayofnodes):  #####(input1,input2,input3..) :
    
    #dd.updateweightsandbiases(arrayofnodes)
    #arrayofnodes = (['0','1','2','3','4','5','6','7','8','9','10']) # sample real one would be effected by weights and biases
    #for node in arrayofnodes: # all 10 nodes
    #    print (node)

    width = 28     # 0 to 255 256 values in total
    height = 28    # 0 to 255 256 values in total
    img = []        # empty array

    for y in range(height):   # y values 0 to 255 1
        row = ()              # row = tuples 
        for x in range(width): # x values 0 to 255 1
            row = row + (randrange(256),randrange(256),randrange(256))   # can be made to color or hexcolor here.. I made it pink to test out grayscale
        img.append(row)
    with open('noisetester.png', 'wb') as func: # file name and writing in binary 
        w = png.Writer(width, height, greyscale=False) # png writer 
        w.write(func, img) # writes img with respect to its function.

    img = Image.open('noisetester.png')
    imgGray = img.convert('L') # L for 8 bits of black and white
    imgGray.save('finalnoisegrayscale.png') # prints grayscale version 
    #imgGray.show() # useless creates a temp

    #os.rename ('noisetestergray.png','finalnoisetester.png')

    imge = mpimg.imread('noisetester.png')
    R, G, B = imge[:,:,0], imge[:,:,1], imge[:,:,2] # tuples are within grayscale range
    imgGrayscale = 0.2989 * R + 0.5870 * G + 0.1140 * B
    plt.imshow(imgGrayscale, cmap='gray')
    plt.show() # very useful for coordinate and debugging

newarray = []
grayscalenoise(newarray)