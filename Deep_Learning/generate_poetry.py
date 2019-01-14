import numpy as np
import string

import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from sklearn.utils import shuffle

from .rnn_util import get_robert_frost, init_weight


class SimpleRNN():
    def __init__(self, D, M, V):
        """D: dimensionality of word embedding, M: hidden layer size, V: vocabulary size"""
        self.D = D
        self.M = M
        self.V = V

    def fit(self, X, learning_rate=1, mu=0.99, reg=1.0, activation=T.tanh, epochs=500, show_fig=False):
        N = len(X)
        D = self.D
        M = self.M
        V = self.V
        self.f = activation

        # Initial weights, random
        We = init_weight(V, D)
        Wx = init_weight(D, M)
        Wh = init_weight(M, M)
        bh = np.zeros(M)
        h0 = np.zeros(M)       # Initial hidden weight vector (time 0)
        Wo = init_weight(M, V) # Output is prediction of next word, so it will be a vector of length
        bo = np.zeros(V)       # V, and the argmax will be taken along that vector to predict word

        # Make all of params theano shared variables
        self.We = theano.shared(We)
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)

        # Collect params so gradient descent is easy
        self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

        thX = T.ivector('X') # A sequence of word vectors
        Ei = self.We[thX]    # Word embedding indexed by thX indices. Selecting rows corresponding
        thY = T.ivector('Y') # to words in sequence of word vectors

        def recurrence(x_t, h_t1):
            """Recurrence function that we define, will be passed into Theano scan function.
            Returns: h(t), y(t)"""
            h_t = self.f(x_t.dot())
