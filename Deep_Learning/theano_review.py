import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt # Used to plot log-likelihood
import seaborn as sns
from modern_dl_util import get_normalized_data, y2indicator # Util to get data and create ind matrix

def error_rate(p, t):
    return np.mean(p != t)


def relu(a):
    return a * (a > 0)


def main():
    # ----------- Step 1: Get data and define usual variables -----------
    X, Y = get_normalized_data()

    max_iter = 20
    print_period = 10

    lr = 0.00004
    reg = 0.01

    Xtrain = X[:-1000]
    Ytrain = Y[:-1000]
    Xtest = X[-1000:]
    Ytest = Y[-1000:]
    Ytrain_ind = y2indicator(Ytrain)
    Ytest_ind = y2indicator(Ytest)

    N, D = Xtrain.shape
    batch_sz = 500
    n_batches = N // batch_sz

    M = 300 # 300 hidden units
    K = 10 # 10 classes
    W1_init = np.random.randn(D, M) / 28
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M, K) / np.sqrt(M)
    b2_init = np.zeros(K)

    # ----------- Step 2: Define theano variables and expressions -----------
    thX = T.matrix('X')
    thT = T.matrix('T')
    W1 = theano.shared(W1_init, 'W1') # All parameters are shared variables
    b1 = theano.shared(b1_init, 'b1')    # Shared variable: first arg is initial value
    W2 = theano.shared(W2_init, 'W2')    # second arg is name
    b2 = theano.shared(b2_init, 'b2')

    thZ = relu(thX.dot(W1) + b1) # Create function to solve for Z using relu activation
    thY = T.nnet.softmax(thZ.dot(W2) + b2) # Create function to solve for Y using softmax

    cost = ( -(thT * T.log(thY)).sum() + reg*((W1*W1).sum() + (b1*b1).sum() + (W2*W2).sum() + (b2*b2).sum()))
    prediction = T.argmax(thY, axis=1)

    # ----------- Step 3: Create training/update expressions -----------
    update_W1 = W1 - lr * T.grad(cost, W1)
    update_b1 = b1 - lr * T.grad(cost, b1)
    update_W2 = W2 - lr * T.grad(cost, W2)
    update_b2 = b2 - lr * T.grad(cost, b2)

    # Create train function. Takes in placeholder for X input matrix and placeholder for target matrix
    train = theano.function(
        inputs=[thX, thT],
        updates=[(W1, update_W1), (b1, update_b1), (W2, update_W2), (b2, update_b2)]
    )

    # Create function to get prediction
    get_prediction = theano.function(
        inputs=[thX, thT],
        outputs=[cost, prediction]
    )

    # Training loop
    costs = []
    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = Xtrain[j*batch_sz:(j*batch_sz + batch_sz),]
            Ybatch = Ytrain_ind[j*batch_sz:(j*batch_sz + batch_sz),]

            train(Xbatch, Ybatch) # Calling the train function we created
            if j % print_period == 0:
                # Calling in prediction function we created to get cost and prediction
                cost_val, prediction_val = get_prediction(Xtest, Ytest_ind)
                err = error_rate(prediction_val, Ytest)
                print(f'Cost / err at iteration i={i}, j={j}: {cost_val / err}')
                costs.append(cost_val)

    fig, ax = plt.subplots(figsize=(12,8))
    plt.plot(costs)
    plt.show()

if __name__ == '__main__':
    main()