from functools import reduce

import numpy as np
from sklearn.metrics import f1_score

from Machine_Learning.hmm.discrete_hmm_scaled import HMM
from Machine_Learning.hmm.baseline_logistic_regression import get_data


def accuracy(T, Y):
    # T: targets, Y: predictions
    # inputs are lists of lists
    n_correct = 0
    n_total = 0
    for t, y in zip(T, Y):
        n_correct += np.sum(t == y)
        n_total += len(y)
    return float(n_correct) / n_total


def total_f1_score(T, Y):
    # inputs are lists of lists
    T = np.concatenate(T)
    Y = np.concatenate(Y)
    return f1_score(T, Y, average=None).mean()


def main(smoothing=1e-1):
    # X = words, Y = POS tags
    Xtrain, Ytrain, Xtest, Ytest, word2idx = get_data(split_sequences=True)
    V = len(word2idx) + 1

    # Find hidden state transition matrix (A) and initial state distribution (pi)
    M = len(set(reduce(lambda x,y: x+y, Ytrain))) + 1
    A = np.ones((M, M)) * smoothing # Add-one smoothing
    pi = np.zeros(M)
    for y in Ytrain:
        # Loop through all hidden states (pos tags)
        if len(y) > 0:
            pi[y[0]] += 1
            for i in range(len(y) - 1):
                A[y[i], y[i+1]] += 1
    # Turn A and pi into probability matrices
    A /= A.sum(axis=1, keepdims=True)
    pi /= pi.sum()

    # Find the observation matrix
    B = np.ones((M, V)) * smoothing
    for x, y in zip(Xtrain, Ytrain):
        for xi, yi in zip(x, y):
            B[yi, xi] += 1
    B /= B.sum(axis=1, keepdims=True)

    hmm = HMM(M)
    hmm.pi = pi
    hmm.A = A
    hmm.B = B

    # get predictions
    Ptrain = []
    for x in Xtrain:
        p = hmm.get_state_sequence(x)
        Ptrain.append(p)

    Ptest = []
    for x in Xtest:
        p = hmm.get_state_sequence(x)
        Ptest.append(p)

    # print results
    print("train accuracy:", accuracy(Ytrain, Ptrain))
    print("test accuracy:", accuracy(Ytest, Ptest))
    print("train f1:", total_f1_score(Ytrain, Ptrain))
    print("test f1:", total_f1_score(Ytest, Ptest))


if __name__ == '__main__':
    main()
