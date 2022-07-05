import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.metrics import log_loss

# loads the MNIST Dataset
# class MNIST():
#     def __init__(self, filename):
#         self.raw_data = pd.read_csv(filename)
#         self.l = raw_data['label']
#         self.d = raw_data.drop('label', axis = 1)

class InputLayer():
    def __init__(self, num_inputs, num_neurons):
        self.weights = 0.01 * np. random.randn(num_inputs, num_neurons) # initializes weights as a inputs x neruons size array with numbers close to 0
        self.biases = np.zeros((1, num_neurons)) # initalizes biases as 0s

    def feedforward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def optimize(self):
        self.weights

class Softmax():
    def feed(self, inputs):
        ex = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = ex / np.sum(ex, axis=1, keepdims=True)

class ReLU:
    def feed(self, inputs):
        self.output = np.maximum(0, inputs)

''' Only works for measuring loss for a single image'''
class SingleLossFunction():
    # Categorical cross entropy
    def CCE(self, predicted_value, actual_value):
        self.loss = -np.sum(actual_value * np.log(predicted_value + 10**-100))
        print(self.loss)
        
class MultiLossFunction():
    # Categorical cross entropy
    def CCE(self, predicted_value, actual_value):
        self.loss = -np.sum(actual_value * np.log(predicted_value + 10**-100))
        self.loss = self.loss / len(predicted_value)
        print(self.loss)

class LeakyRelu():

    def back(self,inputs):
        if inputs > 0:
            #self.output = inputs
            #print(self.output)
            return inputs
        else:
            #self.output = inputs * 0.05
            #print (self.output)
            return 0.05*inputs

class secondoptimizer():

    def sumofsquareresiduals_gradient(self, x, y, b):
        res = b[0] + b[1] * x - y  # 7 + 8 (4)-8 = 20 , 31 mean = 25.5 , 
        #print (res)
        return res.mean(), (res * x).mean()  # .mean() is a method of np.ndarray

    def sgd(self,gradient, x, y, start, learn_rate=0.1, batch_size=1, n_iter=50,tolerance=1e-06, dtype="float64", random_state=None):
    
        # Checking if the gradient is callable
        if not callable(gradient):
            raise TypeError("'gradient' must be callable")

        # Setting up the data type for NumPy arrays
        dtype_ = np.dtype(dtype)

        # Converting x and y to NumPy arrays
        x, y = np.array(x, dtype=dtype_), np.array(y, dtype=dtype_)
        n_obs = x.shape[0]
        if n_obs != y.shape[0]:
            raise ValueError("'x' and 'y' lengths do not match")
        xy = np.c_[x.reshape(n_obs, -1), y.reshape(n_obs, 1)]

        # Initializing the random number generator
        seed = None if random_state is None else int(random_state)
        rng = np.random.default_rng(seed=seed)

        # Initializing the values of the variables
        vector = np.array(start, dtype=dtype_)

        # Setting up and checking the learning rate
        learn_rate = np.array(learn_rate, dtype=dtype_)
        if np.any(learn_rate <= 0):
            raise ValueError("'learn_rate' must be greater than zero")

        # Setting up and checking the size of minibatches
        batch_size = int(batch_size)
        if not 0 < batch_size <= n_obs:
            raise ValueError(
                "'batch_size' must be greater than zero and less than "
                "or equal to the number of observations"
            )

        # Setting up and checking the maximal number of iterations
        n_iter = int(n_iter)
        if n_iter <= 0:
            raise ValueError("'n_iter' must be greater than zero")

        # Setting up and checking the tolerance
        tolerance = np.array(tolerance, dtype=dtype_)
        if np.any(tolerance <= 0):
            raise ValueError("'tolerance' must be greater than zero")

        # Performing the gradient descent loop
        for _ in range(n_iter):
            # Shuffle x and y
            rng.shuffle(xy)  

            # Performing minibatch moves
            for start in range(0, n_obs, batch_size):
                stop = start + batch_size
                x_batch, y_batch = xy[start:stop, :-1], xy[start:stop, -1:]

                # Recalculating the difference
                grad = np.array(gradient(x_batch, y_batch, vector), dtype_)
                diff = -learn_rate * grad

                # Checking if the absolute difference is small enough
                if np.all(np.abs(diff) <= tolerance):
                    break

                # Updating the values of the variables
                vector += diff

            print (vector)

        return vector if vector.shape else vector.item()




        


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



