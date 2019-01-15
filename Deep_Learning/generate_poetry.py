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

        # thX: A sequence of word index's (with START token)
        # Ei: Word embedding indexed by thX indices. Selecting rows corresponding to words in sequence of word vectors.
        # thY: Sequence of word index's (with END token)
        # Example of thX: [0, 629, 1541, 29, 823] -> 0 is the START token
        # Example of thY: [629, 1541, 29, 823, 1] -> 1 is the END token
        thX = T.ivector('X')
        Ei = self.We[thX]
        thY = T.ivector('Y')

        def recurrence(x_t, h_t1):
            """Recurrence function that we define, will be passed into Theano scan function.
            Returns: h(t), y(t)"""
            h_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t

        # Create scan function -> The scan function allows us to pass in the length of a
        # theano variable as the number of times it will loop. As a reminder, the structure
        # of scan is as follows:
        #   - fn: Function to be applied to every element of sequence passed in, in our case
        #         the function is the recurrence function
        #   - outputs_info: Initial value of recurring variables
        #   - sequences: Actual sequence being passed in
        #   - n_steps: number of things to iterate over, generally len(sequences)
        [h, y], = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0],
            sequences=Ei,
            n_steps=Ei.shape[0]
        )

        # Get output
        py_x = y[:, 0, :] # We only care about first and last dimension
        prediction = T.argmax(py_x, axis=1)

        # thY.shape[0] is (symbolically) the number of rows in thY, i.e. the number of examples
        # in thY. Remember, in this case that is just the number of word index's in the sequence
        # T.arange(thY.shape[0]) is a symbolic vector which will contain [0,1,2,...,n-1]
        # py_x is our prediction matrix (of shape len(thX) x V), where each row corresponds to
        # the probability of each word being the next word in the sequence. py_x will be indexed
        # by [0,1,2,...,n-1],[629, 1541, 29, 823, 1], as an example. Here, the second list holds
        # the targets. We know that the true probability distribution has the target having a
        # probability of 1. Hence, we can see what the probability was that we predicted for the
        # target, in order to determine the cross entropy between the two distributions. We then
        # take the log of that, and finally the mean, as per usual.
        cost = -T.mean(T.log(py_x[T.arange(thY.shape[0]), thY])) # Standard cross entropy cost
        grads = T.grad(cost, self.params) # Calculate all gradients in one step
        dparams = [theano.shared(p.get_value() * 0) for p in self.params] # Set momentum params

        # Update params with gradient descent with momentum. We do this by creating a list of updates
        # which the following shape:
        # updates = [
        #        (dWe, mu *  dWe - learning_rate * gWe), (We, We +  mu *  dWe - learning_rate * gWe),
        #        (...),
        #        (...),
        # ]
        # Each parameter is updated, as is the corresponding parameter momentum.
        updates = []
        for p, dp, g in zip(self.params, dparams, grads):
            new_dp = mu * dp - learning_rate * g
            updates.append((dp, new_dp))

            new_p = p + new_dp
            updates.append((p, new_p))

        # Define predict op and train op
        self.predict_op = theano.function(inputs=[thX], outputs=[prediction])
        self.train_op = theano.function(
            inputs=[thX, thY],
            outputs=[cost, prediction],
            updates=updates,
        )

        # Enter main training loop
        costs = []
        n_total = sum((len(sentence) + 1) for sentence in X)
        for i in range(epochs):
            X = shuffle(X)
            n_correct = 0
            total_cost = 0
            for j in range(N):
                input_sequence = [0] + X[j]  # [0] as start token
                output_sequence = X[j] + [1] # [1] as end token
                cost, predictions, y = self.train_op(input_sequence, output_sequence)
                total_cost += cost # Accumulate cost
                for prediction_j, target_j in zip(predictions, output_sequence): # Loop through all predictions
                    if prediction_j == target_j:
                        n_correct += 1
            if i % 50 == 0:
                plt.plot(total_cost)
                plt.show()

    def save(self, filename):
        np.savez(filename, *[p.get_value() for p in self.params])  # Save multiple arrays at once

    # Static load method
    @staticmethod
    def load(filename, activation):
        npz = np.load(filename)
        We = npz['arr_0']
        Wx = npz['arr_1']
        Wh = npz['arr_2']
        bh = npz['arr_3']
        h0 = npz['arr_4']
        Wo = npz['arr_5']
        bo = npz['arr_6']
        V, D = We.shape
        _, M = Wx.shape
        rnn = SimpleRNN(D, M, V)
        rnn.set(We, Wx, Wh, bh, h0, Wo, bo, activation)
        return rnn

    def set(self, We, Wx, Wh, bh, h0, Wo, bo, activation):
        # Pass in np arrays, turn them into theano variables
        self.f = activation

        # redundant - see how you can improve it
        self.We = theano.shared(We)
        self.Wx = theano.shared(Wx)
        self.Wh = theano.shared(Wh)
        self.bh = theano.shared(bh)
        self.h0 = theano.shared(h0)
        self.Wo = theano.shared(Wo)
        self.bo = theano.shared(bo)
        self.params = [self.We, self.Wx, self.Wh, self.bh, self.h0, self.Wo, self.bo]

        thX = T.ivector('X')
        Ei = self.We[thX]  # will be a TxD matrix
        thY = T.ivector('Y')

        def recurrence(x_t, h_t1):
            # returns h(t), y(t)
            h_t = self.f(x_t.dot(self.Wx) + h_t1.dot(self.Wh) + self.bh)
            y_t = T.nnet.softmax(h_t.dot(self.Wo) + self.bo)
            return h_t, y_t

        [h, y], _ = theano.scan(
            fn=recurrence,
            outputs_info=[self.h0, None],
            sequences=Ei,
            n_steps=Ei.shape[0],
        )

        py_x = y[:, 0, :]
        prediction = T.argmax(py_x, axis=1)
        self.predict_op = theano.function(
            inputs=[thX],
            outputs=prediction,
            allow_input_downcast=True,
        )

    def generate(self, pi, word2idx):
        # Generate poetry given the saved model
        # convert word2idx -> idx2word
        idx2word = {v: k for k, v in word2idx.items()}
        V = len(pi)

        # generate 4 lines at a time (4 line verses)
        n_lines = 0

        # Initial word is randomly sampled from V, with sampling distribution pi
        X = [np.random.choice(V, p=pi)]
        print(idx2word[X[0]], end=" ")

        while n_lines < 4:
            P = self.predict_op(X)[-1]  # Predict based on current sequence X
            X += [P]  # Concact prediction onto sequence X
            if P > 1:
                # it's a real word, not start/end token (start is 0, end is 1)
                word = idx2word[P]
                print(word, end=" ")
            elif P == 1:
                # end token
                n_lines += 1
                print('')
                if n_lines < 4:
                    X = [np.random.choice(V, p=pi)]  # reset to start of line
                    print(idx2word[X[0]], end=" ")

def train_poetry():
    sentences, word2idx = get_robert_frost()
    print(len(word2idx))
    rnn = SimpleRNN(30, 30, len(word2idx))
    rnn.fit(sentences, learning_rate=1e-4, show_fig=True, activation=T.nnet.relu, epochs=5)
    rnn.save('RNN_D30_M30_epochs2000_relu.npz')

def generate_poetry():
    # Can call after training poetry
    sentences, word2idx = get_robert_frost()
    rnn = SimpleRNN.load('RNN_D30_M30_epochs2000_relu.npz', T.nnet.relu)

    # determine initial state distribution for starting sentences
    V = len(word2idx)
    pi = np.zeros(V)
    for sentence in sentences:
        pi[sentence[0]] += 1
    pi /= pi.sum()

    rnn.generate(pi, word2idx)


if __name__ == '__main__':
    train_poetry()
    generate_poetry()
