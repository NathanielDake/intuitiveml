from __future__ import print_function, division
from builtins import range

import numpy as np
import pandas as pd

"""
Function to initialize a weight matrix and a bias. M1 is the input size, and M2 is the output size
W is a matrix of size M1 x M2, which is randomized initialy to a gaussian normal
We make the standard deviation of this the sqrt of size in + size out
The bias is initialized as zeros. Each is then turned into float 32s so that they can be used in 
Theano and TensorFlow
"""
def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)


"""
Used for convolutional neural networks
"""
def init_filter(shape, poolsz):
    w = np.random.randn(*shape) * np.sqrt(2) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsz)))
    return w.astype(np.float32)

"""
Rectifier Linear Unit - an activation function that can be used in a neural network 
"""
def relu(x):
    return x * (x > 0)


"""
Sigmoid Function
"""
def sigmoid(A):
    return 1 / (1 + np.exp(-A))

"""
Softmax Function
"""
def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

"""
Calculates the cross entropy from the definition for sigmoid cost (binary classification)
"""
def sigmoid_cost(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()

"""
More general cost function, will work for softmax. Direct from the definition
"""
def cost(T, Y):
    return -(T*np.log(Y)).sum()


"""
Also calculates softmax cross entropy, but it does it in a more complicated way, 
By only calculating cost where targets are non zero. Will return same answer as `cost`
"""
def cost2(T, Y):
    # same as cost(), just uses the targets to index Y
    # instead of multiplying by a large indicator matrix with mostly 0s
    N = len(T)
    return -np.log(Y[np.arange(N), T]).mean()

"""
Gives the error rate between targets and predictions 
"""
def error_rate(targets, predictions):
    return np.mean(targets != predictions)

"""
Creates indicator (N x K), from an input N x 1 y matrix
"""
def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind

"""
Function to get all data from all classes. Start by initializing empty lists for X and Y
Skip the first line, since they are headers. The first column is the label, and the second 
column is space separated pixel values, so we turn them all into integers. We then convert 
these into numpy arrays and normalize this data. Because our classes are imbalanced, we lengthen class 1, 
by repeating it 9 times. 
"""
def getData(balance_ones=True):
    # images are 48x48 = 2304 size vectors
    Y = []
    X = []
    first = True
    for line in open('../../../data/fer/fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)

    if balance_ones:
        # balance the 1 class
        X0, Y0 = X[Y!=1, :], Y[Y!=1]
        X1 = X[Y==1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1]*len(X1)))

    return X, Y

"""
Used for grabbing image data when working with convolutional neural nets
"""
def getImageData():
    X, Y = getData()
    N, D = X.shape
    d = int(np.sqrt(D))
    X = X.reshape(N, 1, d, d)
    return X, Y

"""
Nearly the same function as getData, but now only grabbing class 0 and 1
Dealing with class imbalance in the logistic file itself.
"""
def getBinaryData():
    Y = []
    X = []
    first = True
    for line in open('../../../data/fer/fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            y = int(row[0])
            if y == 0 or y == 1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])
    return np.array(X) / 255.0, np.array(Y)


def crossValidation(model, X, Y, K=5):
    # split data into K parts
    X, Y = shuffle(X, Y)
    sz = len(Y) // K
    errors = []
    for k in range(K):
        xtr = np.concatenate([ X[:k*sz, :], X[(k*sz + sz):, :] ])
        ytr = np.concatenate([ Y[:k*sz], Y[(k*sz + sz):] ])
        xte = X[k*sz:(k*sz + sz), :]
        yte = Y[k*sz:(k*sz + sz)]

        model.fit(xtr, ytr)
        err = model.score(xte, yte)
        errors.append(err)
    print("errors:", errors)
    return np.mean(errors)
