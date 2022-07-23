
import seaborn as sns
from scipy.stats import norm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import newdiscr as nd


low = (0, 0, 1) # dark blue
color = [low]#, medium, medium, high, low]

lista = []
for i in range(784):
        lista.append(i)

# ! dark blue is fake data

xplot = np.asarray(lista)
yplot = np.asarray([40.804, 34.225, 7.920999999999999, 3.249, 0.0, 0.0, 12.321, 0.196, 0.32399999999999995, 0.08099999999999999, 11.449, 29.241000000000003, 46.225, 6.724, 11.664, 16.129, 44.521, 30.275999999999996, 19.044, 0.8999999999999999, 2.116, 42.849, 27.556, 0.8999999999999999, 3.721, 8.281, 46.656, 12.544, 7.568999999999999, 61.504, 26.244, 41.616, 12.769, 46.225, 2.601, 0.4, 52.441, 49.284, 10.815999999999999, 1.225, 23.409, 10.609, 0.049, 1.089, 55.224999999999994, 1.089, 17.689, 11.881, 54.756, 47.524, 26.569000000000003, 8.649, 37.249, 53.824000000000005, 14.884, 5.625, 63.504, 26.569000000000003, 15.129, 35.721000000000004, 14.399999999999999, 2.916, 0.009000000000000001, 28.900000000000002, 5.329, 25.6, 1.1560000000000001, 63.504, 0.16899999999999998, 20.448999999999998, 38.809000000000005, 2.116, 50.625, 3.721, 63.504, 0.196, 19.321, 12.321, 57.599999999999994, 65.025, 47.089, 25.921, 2.601, 41.616, 1.764, 31.683999999999997, 53.361000000000004, 23.716, 17.161, 48.841, 16.129, 60.516, 4.489, 3.3640000000000003, 23.104, 17.161, 61.504, 16.900000000000002, 0.025, 53.824000000000005, 44.1, 63.001, 55.224999999999994, 0.256, 0.036000000000000004, 61.009, 5.476, 1.444, 12.544, 16.900000000000002, 27.225, 35.721000000000004, 21.025, 63.001, 61.504, 25.281, 0.6759999999999999, 40.401, 8.464, 7.920999999999999, 1.1560000000000001, 8.649, 61.504, 0.729, 32.041, 65.025, 26.896, 36.1, 20.735999999999997, 55.696, 31.328999999999997, 6.241, 47.961, 43.263999999999996, 0.0, 34.969, 62.001, 14.641, 51.076, 27.225, 3.3640000000000003, 7.056, 38.416000000000004, 63.504, 0.196, 6.084, 64.516, 1.1560000000000001, 1.681, 27.556, 0.8999999999999999, 63.504, 21.608999999999998, 3.5999999999999996, 19.6, 19.321, 18.769000000000002, 9.025, 42.849, 5.625, 12.769, 33.856, 50.625, 30.275999999999996, 40.804, 3.249, 19.6, 10.0, 14.884, 9.409, 4.6240000000000006, 1.089, 12.769, 63.504, 2.5, 0.009000000000000001, 19.044, 34.225, 0.1, 0.16899999999999998, 11.235999999999999, 3.5999999999999996, 8.464, 25.6, 32.760999999999996, 6.889, 53.361000000000004, 1.936, 46.656, 3.721, 30.624999999999996, 17.161, 5.625, 0.08099999999999999, 0.729, 35.721000000000004, 57.599999999999994, 44.521, 20.163999999999998, 64.516, 18.496000000000002, 4.9, 50.625, 1.8489999999999998, 14.884, 1.764, 37.249, 9.801, 0.8410000000000001, 44.521, 4.096, 19.044, 16.129, 61.009, 8.1, 29.241000000000003, 37.636, 31.683999999999997, 20.163999999999998, 7.920999999999999, 10.609, 2.916, 43.263999999999996, 32.041, 29.583999999999996, 1.8489999999999998, 6.4, 0.784, 1.681, 11.881, 0.036000000000000004, 42.849, 0.5760000000000001, 40.401, 4.6240000000000006, 46.656, 25.281, 0.529, 0.8410000000000001, 3.3640000000000003, 3.025, 20.163999999999998, 51.076, 2.4010000000000002, 3.481, 0.6759999999999999, 5.0409999999999995, 27.556, 26.569000000000003, 4.489, 11.235999999999999, 21.025, 10.0, 55.224999999999994, 44.1, 27.889000000000003, 0.5760000000000001, 17.161, 46.225, 3.025, 41.209, 33.124, 49.284, 48.841, 0.009000000000000001, 0.14400000000000002, 2.025, 0.256, 0.8999999999999999, 3.721, 0.064, 24.336, 9.604000000000001, 37.249, 19.321, 0.625, 60.516, 3.025, 0.009000000000000001, 0.08099999999999999, 23.716, 31.683999999999997, 12.544, 4.761, 51.529, 5.625, 0.009000000000000001, 15.376, 11.449, 8.464, 38.025, 45.369, 0.8410000000000001, 2.209, 62.5, 26.896, 32.4, 0.729, 28.900000000000002, 44.943999999999996, 19.6, 6.084, 6.241, 19.880999999999997, 4.9, 58.564, 43.263999999999996, 21.025, 12.321, 7.395999999999999, 46.656, 44.943999999999996, 36.1, 44.943999999999996, 21.316, 60.516, 0.484, 17.161, 22.201, 24.336, 59.049, 7.395999999999999, 4.6240000000000006, 21.316, 14.161, 6.084, 27.889000000000003, 3.969, 16.129, 59.536, 3.136, 0.32399999999999995, 10.404, 42.436, 64.009, 24.336, 1.444, 60.516, 38.416000000000004, 16.129, 64.009, 28.561000000000003, 49.729, 45.369, 5.625, 17.424, 0.5760000000000001, 64.009, 3.249, 15.376, 47.089, 64.516, 11.025, 3.721, 22.5, 3.025, 14.884, 27.889000000000003, 16.129, 63.504, 48.841, 63.504, 10.0, 46.656, 30.976, 37.249, 30.624999999999996, 16.384, 15.876, 1.764, 56.169, 0.4, 15.876, 0.121, 26.244, 64.009, 27.889000000000003, 13.225000000000001, 3.844, 43.681, 7.920999999999999, 46.656, 0.009000000000000001, 14.399999999999999, 2.916, 14.641, 59.536, 29.929, 19.880999999999997, 4.489, 13.689, 1.024, 12.544, 6.724, 4.761, 20.163999999999998, 35.344, 46.225, 36.864000000000004, 12.321, 47.524, 32.041, 38.416000000000004, 51.984, 54.756, 45.796, 63.001, 19.321, 3.249, 33.489, 21.904, 5.625, 59.536, 20.735999999999997, 46.656, 52.900000000000006, 1.225, 33.489, 11.235999999999999, 17.161, 12.1, 45.369, 8.836, 56.169, 52.900000000000006, 43.263999999999996, 1.225, 23.104, 0.4, 59.049, 11.449, 0.441, 2.4010000000000002, 20.163999999999998, 6.241, 61.009, 2.025, 38.809000000000005, 26.244, 1.369, 21.316, 3.3640000000000003, 44.1, 9.409, 53.361000000000004, 60.025, 61.504, 9.025, 19.6, 21.608999999999998, 63.001, 32.4, 48.4, 4.489, 51.076, 28.900000000000002, 3.025, 4.6240000000000006, 2.916, 0.196, 0.036000000000000004, 21.316, 17.161, 63.001, 23.716, 37.249, 16.129, 0.121, 23.716, 18.769000000000002, 5.476, 7.056, 64.516, 36.864000000000004, 27.225, 12.996, 37.249, 38.416000000000004, 0.1, 45.369, 3.5999999999999996, 33.124, 27.225, 19.6, 6.4, 3.969, 28.224, 6.084, 64.516, 15.876, 0.1, 33.124, 13.689, 17.956, 25.281, 3.844, 8.464, 0.001, 55.696, 0.6759999999999999, 3.025, 54.289, 11.235999999999999, 58.564, 0.025, 49.284, 51.984, 17.161, 20.735999999999997, 35.344, 11.881, 17.689, 16.129, 57.120999999999995, 1.6, 14.399999999999999, 17.424, 61.009, 42.849, 25.281, 3.969, 36.1, 47.961, 5.476, 3.136, 29.241000000000003, 0.025, 0.08099999999999999, 37.636, 10.404, 51.076, 2.5, 25.6, 37.636, 0.001, 29.583999999999996, 48.4, 9.409, 37.636, 23.716, 8.281, 2.4010000000000002, 24.336, 5.0409999999999995, 7.920999999999999, 6.889, 40.804, 26.244, 30.624999999999996, 6.724, 6.561, 29.929, 36.1, 17.161, 42.849, 26.244, 14.884, 25.281, 11.664, 44.521, 2.3040000000000003, 1.1560000000000001, 3.136, 0.16899999999999998, 24.649, 4.6240000000000006, 0.529, 58.564, 30.275999999999996, 9.604000000000001, 3.969, 21.904, 24.964, 6.241, 52.441, 63.504, 4.761, 0.361, 33.489, 19.044, 21.608999999999998, 0.484, 0.784, 0.016, 8.649, 17.956, 15.129, 33.489, 49.284, 0.8999999999999999, 28.224, 24.964, 16.641000000000002, 10.0, 49.729, 5.0409999999999995, 48.4, 0.025, 8.464, 41.616, 52.900000000000006, 15.376, 9.025, 0.08099999999999999, 39.601, 0.961, 5.0409999999999995, 0.256, 48.841, 41.209, 31.683999999999997, 21.316, 9.025, 2.601, 4.9, 63.001, 42.849, 43.681, 0.625, 0.8410000000000001, 24.964, 51.076, 42.849, 21.904, 56.644, 18.496000000000002, 1.024, 22.201, 2.209, 24.025, 63.001, 26.244, 45.369, 15.876, 5.0409999999999995, 2.3040000000000003, 1.089, 0.5760000000000001, 45.796, 20.163999999999998, 0.025, 0.529, 2.7039999999999997, 62.5, 33.124, 55.224999999999994, 15.625, 30.275999999999996, 0.0, 25.281, 43.681, 7.2250000000000005, 5.183999999999999, 3.025, 8.649, 12.769, 34.596, 4.356, 5.929, 27.556, 0.32399999999999995, 14.884, 40.804, 57.599999999999994, 5.183999999999999, 27.225, 22.801, 2.025, 1.369, 22.201, 11.449, 47.089, 2.8089999999999997, 10.201, 14.399999999999999, 2.5, 5.929, 39.204, 48.841, 0.8999999999999999, 47.524, 57.599999999999994, 51.529, 63.001, 48.4, 36.864000000000004, 13.689, 58.080999999999996, 7.395999999999999, 2.209, 6.084, 27.225, 0.4, 2.3040000000000003, 27.225, 21.025, 64.009, 18.769000000000002, 42.849, 0.8999999999999999, 41.209, 22.5, 33.124, 1.521, 38.809000000000005, 36.1, 34.596, 9.025, 45.796, 6.724, 25.6, 14.161, 1.6, 0.32399999999999995, 17.424, 26.896, 5.625, 18.769000000000002, 7.056, 0.14400000000000002, 1.681, 9.409, 12.1, 28.224, 7.395999999999999, 24.025, 64.516, 60.025, 34.969, 12.1, 54.756, 3.025, 8.1, 63.504, 3.844, 7.920999999999999, 3.136, 62.5, 12.1, 34.225, 0.22499999999999998, 5.625, 24.336, 54.289, 11.025, 0.4, 9.801, 7.395999999999999, 0.784, 3.481, 0.16899999999999998, 6.889, 2.3040000000000003, 33.489, 27.889000000000003, 34.969, 47.089, 26.569000000000003, 0.049, 36.481, 3.969, 8.649, 1.8489999999999998, 35.344, 56.644, 0.5760000000000001, 43.681, 2.601, 5.625, 24.336, 52.900000000000006, 7.2250000000000005, 44.1, 65.025, 51.076, 13.456000000000001, 19.6, 8.649, 7.056, 36.1, 0.32399999999999995, 1.8489999999999998, 0.6759999999999999, 31.683999999999997])

plt.scatter(
    x=xplot,
    y=yplot,
    c=color,
    alpha = 0.5

)


plt.show()
#* light blue is real data






xsecplot = np.asarray(lista)
ysecplot = np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 106, 254, 246, 56, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36, 229, 253, 253, 204, 17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 30, 217, 254, 253, 253, 133, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 253, 254, 253, 253, 190, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 195, 253, 254, 253, 253, 253, 228, 110, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 196, 253, 253, 254, 253, 253, 253, 253, 244, 101, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 72, 247, 253, 253, 254, 253, 253, 253, 253, 253, 253, 94, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 11, 177, 253, 253, 253, 164, 40, 14, 128, 253, 253, 253, 163, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 145, 253, 253, 245, 120, 0, 0, 0, 28, 231, 253, 253, 216, 35, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 223, 253, 253, 208, 0, 0, 0, 0, 0, 224, 253, 253, 163, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 137, 254, 254, 254, 209, 0, 0, 0, 0, 121, 254, 254, 254, 164, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 231, 253, 253, 232, 71, 0, 0, 0, 4, 151, 253, 253, 253, 163, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 120, 253, 253, 253, 66, 0, 0, 0, 0, 113, 253, 253, 253, 253, 163, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 183, 253, 253, 253, 14, 0, 0, 0, 29, 206, 253, 253, 253, 189, 87, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 148, 253, 253, 253, 133, 2, 0, 0, 137, 231, 253, 253, 253, 226, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 175, 253, 253, 214, 35, 29, 57, 179, 254, 253, 253, 253, 226, 109, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 253, 253, 253, 169, 134, 247, 253, 253, 254, 253, 253, 253, 29, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61, 253, 253, 253, 253, 253, 253, 253, 253, 255, 253, 186, 14, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 158, 253, 253, 253, 253, 253, 245, 208, 209, 85, 21, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 104, 236, 253, 253, 200, 86, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

high = (1, 1,0) # light blue

color2 = [high]

plt.scatter(
    x=xsecplot,
    y=ysecplot,
    c=color2,
    alpha = 0.5
)

plt.show()

# Calculating parameters (Here, intercept-theta1 and slope-theta0)
# of the line using the numpy.polyfit() function
#theta = np.polyfit#(xsecplot, ysecplot, 1)

#print(f'The parameters of the line: {theta}')

# Now, calculating the y-axis values against x-values according to
# the parameters theta0, theta1 and theta2
#y_line = theta[1] + theta[0] * xsecplot

# Plotting the data points and the best fit line
#plt.plot(xsecplot, y_line, 'r')
#plt.title('Best fit line using numpy.polyfit()')
#plt.xlabel('x-axis')
#plt.ylabel('y-axis')

#plt.hist(ysecplot, bins=15)
#plt.gca().set(title='Frequency Histogram', ylabel='Frequency')



#data = norm.rvs(5,0.4,size=1000) # you can use a pandas series or a list if you want

#sns.displot(ysecplot)

#plt.show()




#? black is fake with weights data

#xthrplot = np.asarray(lista)
#ythrplot = np.asarray([222.20000000000002, 203.50000000000003, 97.9, 62.7, 15.400000000000002, 28.6, 122.10000000000001, 15.400000000000002, 19.8, 9.9, 117.7, 188.10000000000002, 236.50000000000003, 90.2, 118.80000000000001, 139.70000000000002, 232.10000000000002, 191.4, 151.8, 33.0, 50.6, 227.70000000000002, 182.60000000000002, 33.0, 67.10000000000001, 100.10000000000001, 237.60000000000002, 123.20000000000002, 95.7, 272.8, 178.20000000000002, 224.4, 124.30000000000001, 236.50000000000003, 56.1, 22.0, 251.90000000000003, 244.20000000000002, 114.4, 38.5, 168.3, 113.30000000000001, 7.700000000000001, 36.300000000000004, 258.5, 36.300000000000004, 146.3, 119.9, 257.40000000000003, 239.8, 179.3, 102.30000000000001, 212.3, 255.20000000000002, 134.20000000000002, 82.5, 277.20000000000005, 179.3, 135.3, 207.9, 132.0, 59.400000000000006, 3.3000000000000003, 187.00000000000003, 80.30000000000001, 176.0, 37.400000000000006, 277.20000000000005, 14.3, 157.3, 216.70000000000002, 50.6, 247.50000000000003, 67.10000000000001, 277.20000000000005, 15.400000000000002, 152.9, 122.10000000000001, 264.0, 280.5, 238.70000000000002, 177.10000000000002, 56.1, 224.4, 46.2, 195.8, 254.10000000000002, 169.4, 144.10000000000002, 243.10000000000002, 139.70000000000002, 270.6, 73.7, 63.800000000000004, 167.20000000000002, 144.10000000000002, 272.8, 143.0, 5.5, 255.20000000000002, 231.00000000000003, 276.1, 258.5, 17.6, 6.6000000000000005, 271.70000000000005, 81.4, 41.800000000000004, 123.20000000000002, 143.0, 181.50000000000003, 207.9, 159.5, 276.1, 272.8, 174.9, 28.6, 221.10000000000002, 101.2, 97.9, 37.400000000000006, 102.30000000000001, 272.8, 29.700000000000003, 196.9, 280.5, 180.4, 209.00000000000003, 158.4, 259.6, 194.70000000000002, 86.9, 240.9, 228.8, 0.0, 205.70000000000002, 273.90000000000003, 133.10000000000002, 248.60000000000002, 181.50000000000003, 63.800000000000004, 92.4, 215.60000000000002, 277.20000000000005, 15.400000000000002, 85.80000000000001, 279.40000000000003, 37.400000000000006, 45.1, 182.60000000000002, 33.0, 277.20000000000005, 161.70000000000002, 66.0, 154.0, 152.9, 150.70000000000002, 104.50000000000001, 227.70000000000002, 82.5, 124.30000000000001, 202.4, 247.50000000000003, 191.4, 222.20000000000002, 62.7, 154.0, 110.00000000000001, 134.20000000000002, 106.7, 74.80000000000001, 36.300000000000004, 124.30000000000001, 277.20000000000005, 55.00000000000001, 3.3000000000000003, 151.8, 203.50000000000003, 11.0, 14.3, 116.60000000000001, 66.0, 101.2, 176.0, 199.10000000000002, 91.30000000000001, 254.10000000000002, 48.400000000000006, 237.60000000000002, 67.10000000000001, 192.50000000000003, 144.10000000000002, 82.5, 9.9, 29.700000000000003, 207.9, 264.0, 232.10000000000002, 156.20000000000002, 279.40000000000003, 149.60000000000002, 77.0, 247.50000000000003, 47.300000000000004, 134.20000000000002, 46.2, 212.3, 108.9, 31.900000000000002, 232.10000000000002, 70.4, 151.8, 139.70000000000002, 271.70000000000005, 99.00000000000001, 188.10000000000002, 213.4, 195.8, 156.20000000000002, 97.9, 113.30000000000001, 59.400000000000006, 228.8, 196.9, 189.20000000000002, 47.300000000000004, 88.0, 30.800000000000004, 45.1, 119.9, 6.6000000000000005, 227.70000000000002, 26.400000000000002, 221.10000000000002, 74.80000000000001, 237.60000000000002, 174.9, 25.3, 31.900000000000002, 63.800000000000004, 60.50000000000001, 156.20000000000002, 248.60000000000002, 53.900000000000006, 64.9, 28.6, 78.10000000000001, 182.60000000000002, 179.3, 73.7, 116.60000000000001, 159.5, 110.00000000000001, 258.5, 231.00000000000003, 183.70000000000002, 26.400000000000002, 144.10000000000002, 236.50000000000003, 60.50000000000001, 223.3, 200.20000000000002, 244.20000000000002, 243.10000000000002, 3.3000000000000003, 13.200000000000001, 49.50000000000001, 17.6, 33.0, 67.10000000000001, 8.8, 171.60000000000002, 107.80000000000001, 212.3, 152.9, 27.500000000000004, 270.6, 60.50000000000001, 3.3000000000000003, 9.9, 169.4, 195.8, 123.20000000000002, 75.9, 249.70000000000002, 82.5, 3.3000000000000003, 136.4, 117.7, 101.2, 214.50000000000003, 234.3, 31.900000000000002, 51.7, 275.0, 180.4, 198.00000000000003, 29.700000000000003, 187.00000000000003, 233.20000000000002, 154.0, 85.80000000000001, 86.9, 155.10000000000002, 77.0, 266.20000000000005, 228.8, 159.5, 122.10000000000001, 94.60000000000001, 237.60000000000002, 233.20000000000002, 209.00000000000003, 233.20000000000002, 160.60000000000002, 270.6, 24.200000000000003, 144.10000000000002, 163.9, 171.60000000000002, 267.3, 94.60000000000001, 74.80000000000001, 160.60000000000002, 130.9, 85.80000000000001, 183.70000000000002, 69.30000000000001, 139.70000000000002, 268.40000000000003, 61.60000000000001, 19.8, 112.2, 226.60000000000002, 278.3, 171.60000000000002, 41.800000000000004, 270.6, 215.60000000000002, 139.70000000000002, 278.3, 185.9, 245.3, 234.3, 82.5, 145.20000000000002, 26.400000000000002, 278.3, 62.7, 136.4, 238.70000000000002, 279.40000000000003, 115.50000000000001, 67.10000000000001, 165.0, 60.50000000000001, 134.20000000000002, 183.70000000000002, 139.70000000000002, 277.20000000000005, 243.10000000000002, 277.20000000000005, 110.00000000000001, 237.60000000000002, 193.60000000000002, 212.3, 192.50000000000003, 140.8, 138.60000000000002, 46.2, 260.70000000000005, 22.0, 138.60000000000002, 12.100000000000001, 178.20000000000002, 278.3, 183.70000000000002, 126.50000000000001, 68.2, 229.9, 97.9, 237.60000000000002, 3.3000000000000003, 132.0, 59.400000000000006, 133.10000000000002, 268.40000000000003, 190.3, 155.10000000000002, 73.7, 128.70000000000002, 35.2, 123.20000000000002, 90.2, 75.9, 156.20000000000002, 206.8, 236.50000000000003, 211.20000000000002, 122.10000000000001, 239.8, 196.9, 215.60000000000002, 250.8, 257.40000000000003, 235.4, 276.1, 152.9, 62.7, 201.3, 162.8, 82.5, 268.40000000000003, 158.4, 237.60000000000002, 253.00000000000003, 38.5, 201.3, 116.60000000000001, 144.10000000000002, 121.00000000000001, 234.3, 103.4, 260.70000000000005, 253.00000000000003, 228.8, 38.5, 167.20000000000002, 22.0, 267.3, 117.7, 23.1, 53.900000000000006, 156.20000000000002, 86.9, 271.70000000000005, 49.50000000000001, 216.70000000000002, 178.20000000000002, 40.7, 160.60000000000002, 63.800000000000004, 231.00000000000003, 106.7, 254.10000000000002, 269.5, 272.8, 104.50000000000001, 154.0, 161.70000000000002, 276.1, 198.00000000000003, 242.00000000000003, 73.7, 248.60000000000002, 187.00000000000003, 60.50000000000001, 74.80000000000001, 59.400000000000006, 15.400000000000002, 6.6000000000000005, 160.60000000000002, 144.10000000000002, 276.1, 169.4, 212.3, 139.70000000000002, 12.100000000000001, 169.4, 150.70000000000002, 81.4, 92.4, 279.40000000000003, 211.20000000000002, 181.50000000000003, 125.4, 212.3, 215.60000000000002, 11.0, 234.3, 66.0, 200.20000000000002, 181.50000000000003, 154.0, 88.0, 69.30000000000001, 184.8, 85.80000000000001, 279.40000000000003, 138.60000000000002, 11.0, 200.20000000000002, 128.70000000000002, 147.4, 174.9, 68.2, 101.2, 1.1, 259.6, 28.6, 60.50000000000001, 256.3, 116.60000000000001, 266.20000000000005, 5.5, 244.20000000000002, 250.8, 144.10000000000002, 158.4, 206.8, 119.9, 146.3, 139.70000000000002, 262.90000000000003, 44.0, 132.0, 145.20000000000002, 271.70000000000005, 227.70000000000002, 174.9, 69.30000000000001, 209.00000000000003, 240.9, 81.4, 61.60000000000001, 188.10000000000002, 5.5, 9.9, 213.4, 112.2, 248.60000000000002, 55.00000000000001, 176.0, 213.4, 1.1, 189.20000000000002, 242.00000000000003, 106.7, 213.4, 169.4, 100.10000000000001, 53.900000000000006, 171.60000000000002, 78.10000000000001, 97.9, 91.30000000000001, 222.20000000000002, 178.20000000000002, 192.50000000000003, 90.2, 89.10000000000001, 190.3, 209.00000000000003, 144.10000000000002, 227.70000000000002, 178.20000000000002, 134.20000000000002, 174.9, 118.80000000000001, 232.10000000000002, 52.800000000000004, 37.400000000000006, 61.60000000000001, 14.3, 172.70000000000002, 74.80000000000001, 25.3, 266.20000000000005, 191.4, 107.80000000000001, 69.30000000000001, 162.8, 173.8, 86.9, 251.90000000000003, 277.20000000000005, 75.9, 20.900000000000002, 201.3, 151.8, 161.70000000000002, 24.200000000000003, 30.800000000000004, 4.4, 102.30000000000001, 147.4, 135.3, 201.3, 244.20000000000002, 33.0, 184.8, 173.8, 141.9, 110.00000000000001, 245.3, 78.10000000000001, 242.00000000000003, 5.5, 101.2, 224.4, 253.00000000000003, 136.4, 104.50000000000001, 9.9, 218.9, 34.1, 78.10000000000001, 17.6, 243.10000000000002, 223.3, 195.8, 160.60000000000002, 104.50000000000001, 56.1, 77.0, 276.1, 227.70000000000002, 229.9, 27.500000000000004, 31.900000000000002, 173.8, 248.60000000000002, 227.70000000000002, 162.8, 261.8, 149.60000000000002, 35.2, 163.9, 51.7, 170.5, 276.1, 178.20000000000002, 234.3, 138.60000000000002, 78.10000000000001, 52.800000000000004, 36.300000000000004, 26.400000000000002, 235.4, 156.20000000000002, 5.5, 25.3, 57.2, 275.0, 200.20000000000002, 258.5, 137.5, 191.4, 0.0, 174.9, 229.9, 93.50000000000001, 79.2, 60.50000000000001, 102.30000000000001, 124.30000000000001, 204.60000000000002, 72.60000000000001, 84.7, 182.60000000000002, 19.8, 134.20000000000002, 222.20000000000002, 264.0, 79.2, 181.50000000000003, 166.10000000000002, 49.50000000000001, 40.7, 163.9, 117.7, 238.70000000000002, 58.300000000000004, 111.10000000000001, 132.0, 55.00000000000001, 84.7, 217.8, 243.10000000000002, 33.0, 239.8, 264.0, 249.70000000000002, 276.1, 242.00000000000003, 211.20000000000002, 128.70000000000002, 265.1, 94.60000000000001, 51.7, 85.80000000000001, 181.50000000000003, 22.0, 52.800000000000004, 181.50000000000003, 159.5, 278.3, 150.70000000000002, 227.70000000000002, 33.0, 223.3, 165.0, 200.20000000000002, 42.900000000000006, 216.70000000000002, 209.00000000000003, 204.60000000000002, 104.50000000000001, 235.4, 90.2, 176.0, 130.9, 44.0, 19.8, 145.20000000000002, 180.4, 82.5, 150.70000000000002, 92.4, 13.200000000000001, 45.1, 106.7, 121.00000000000001, 184.8, 94.60000000000001, 170.5, 279.40000000000003, 269.5, 205.70000000000002, 121.00000000000001, 257.40000000000003, 60.50000000000001, 99.00000000000001, 277.20000000000005, 68.2, 97.9, 61.60000000000001, 275.0, 121.00000000000001, 203.50000000000003, 16.5, 82.5, 171.60000000000002, 256.3, 115.50000000000001, 22.0, 108.9, 94.60000000000001, 30.800000000000004, 64.9, 14.3, 91.30000000000001, 52.800000000000004, 201.3, 183.70000000000002, 205.70000000000002, 238.70000000000002, 179.3, 7.700000000000001, 210.10000000000002, 69.30000000000001, 102.30000000000001, 47.300000000000004, 206.8, 261.8, 26.400000000000002, 229.9, 56.1, 82.5, 171.60000000000002, 253.00000000000003, 93.50000000000001, 231.00000000000003, 280.5, 248.60000000000002, 127.60000000000001, 154.0, 102.30000000000001, 92.4, 209.00000000000003, 19.8, 47.300000000000004, 28.6, 195.8])

#medium  = (0, 0, 0) # black

#color3 = [medium]

#plt.scatter(
#    x=xthrplot,
#    y=ythrplot,
#    c=color3,
#    alpha = 0.5
#)






# create one with weights and biases
# do this for all the rows then plot all of them on a graph. for label 0. 










#plt.show()