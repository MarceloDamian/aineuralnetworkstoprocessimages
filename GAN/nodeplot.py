import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import numpy as np

#0823659471

zero = (1, 0, 0) # red
color0= [zero]

one = (0, 1, 0) # green
color1 = [one]

sec = (0, 0, 1) # darkblue
color2 = [sec]

third = (1, 1, 0) # yellow
color3 = [third]

fourth = (1, 0, 1)  # pink
color4 = [fourth]

fifth = (0, 1, 1) #light blue
color5 = [fifth]

sixth = (0, 0, 0) # black
color6 = [sixth]

seventh = (0.18, 0, 1) # purple
color7 = [seventh]

eigth = (0.30, 0.50, 0.30) # light green
color8 = [eigth]

ninth = (0.5, 0.2, 0.30) # brown
color9 = [ninth]


listx = [0]
listx1 = [1] # possibly incorperatte this 
listx2 = [2]
listx3 = [3]
listx4 = [4]
listx5 = [5]
listx6 = [6]
listx7 = [7]
listx8 = [8]
listx9 = [9]




zerolist = [34632.40755082285]
firstlist = [15189.466268146884]
seclist = [29873.099353603066]
thirdlist = [28323.188002757986]
fourthlist = [24236.72249508841]
fifthlist = [25840.920421607378]
sixthlist = [27740.917331399563]
seventhlist = [22938.244262667577]
eigthlist = [30192.148412503077]
ninthlist = [24562.75]


plt.scatter(
    x=listx,
    y=zerolist,
    c=color0,
    alpha = 0.5
)

plt.scatter(
    x=listx1,
    y=firstlist,
    c=color1,
    alpha = 0.5
)

plt.scatter(
    x=listx2,
    y=seclist,
    c=color2,
    alpha = 0.5

)
plt.scatter(
    x=listx3,
    y=thirdlist,
    c=color3,
    alpha = 0.5
)
plt.scatter(
    x=listx4,
    y=fourthlist,
    c=color4,
    alpha = 0.5
)
plt.scatter(
    x=listx5,
    y=fifthlist,
    c=color5,
    alpha = 0.5
)
plt.scatter(
    x=listx6,
    y=sixthlist,
    c=color6,
    alpha = 0.5
)
plt.scatter(
    x=listx7,
    y=seventhlist,
    c=color7,
    alpha = 0.5
)
plt.scatter(
    x=listx8,
    y=eigthlist,
    c=color8,
    alpha = 0.5
)
plt.scatter(
    x=listx9,
    y=ninthlist,
    c=color9,
    alpha = 0.5
)











# Calculating parameters (Here, intercept-theta1 and slope-theta0)
# of the line using the numpy.polyfit() function
#theta = np.polyfit(xplot, yplot, 1)

#print(f'The parameters of the line: {theta}')

# Now, calculating the y-axis values against x-values according to
# the parameters theta0, theta1 and theta2
#y_line = theta[1] + theta[0] * xplot

# Plotting the data points and the best fit line
#plt.scatter(xplot, yplot)
#plt.plot(xplot, y_line, 'r')
plt.title('Best fit line using numpy.polyfit()')
plt.xlabel('x-axis')
plt.ylabel('y-axis')

plt.show()
