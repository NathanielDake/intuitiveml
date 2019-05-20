import wave
import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

from Machine_Learning.hmm.utils import random_normalized


class HMM:
    def __init__(self, M, K):
        self.M = M
        self.K = K

    def fit(self, X, learning_rate=10e-3, max_iter=10):
        """Train continuous HMM model using stochastic gradient descent."""

        N = len(X) # Number of sequences
        D = X[0].shape[1] # Dimensionality

        # ---------- 1. Initialize continuous HMM model parameters ----------
        # Original HMM parameters
        pi0 = np.ones(self.M) / self.M # Uniform distribution
        A0 = random_normalized(self.M, self.M)

        # Continuous HMM (GMM) parameters.
        R0 = np.ones((self.M, self.K)) / self.K # Uniform distribution

        # mu is set similar to how it is set in kmeans -> randomly select points from dataset
        mu0 = np.zeros((self.M, self.K, D))
        for i in range(self.M):
            for k in range(self.K):
                # For all states and all gaussians, get a random index, choose a random sequence,
                # get a random time, and set mu[i,k] to be whatever point was at that time
                random_idx = np.random.choice(N)
                x = X[random_idx]
                random_time_idx = np.random.choice(len(x))
                mu0[i,k] = x[random_time_idx]

        # setting sigma so that all gaussians are initialized as spherical
        sigma0 = np.zeros((self.M, self.K, D, D))
        for j in range(self.M):
            for k in range(self.K):
                sigma0[j,k] = np.eye(D)

        thx, cost = self.set(pi0, A0, R0, mu0, sigma0)

        # ---------- 2. Perform updates on continuous HMM Model parameters via stochastic gradient descent ----------
        # This is why theano is so powerful. By linking all variables together via a computational
        # graph, we can define our cost, which is really just the probability of a sequence,
        # and we can then find the gradient of the cost with respect to our parameters, pi, A, B
        # R, mu, and sigma.
        pi_update = self.pi - learning_rate*T.grad(cost, self.pi)
        pi_update = pi_update / pi_update.sum()  # Normalizing to ensure it stays a probability

        A_update = self.A - learning_rate*T.grad(cost, self.A)
        A_update = A_update / A_update.sum(axis=1).dimshuffle(0, 'x')

        R_update = self.R - learning_rate*T.grad(cost, self.R)
        R_update = R_update / R_update.sum(axis=1).dimshuffle(0, 'x')

        updates = [
            (self.pi, pi_update),
            (self.A, A_update),
            (self.R, R_update),
            (self.mu, self.mu - learning_rate*T.grad(cost, self.mu)),
            (self.sigma, self.sigma - learning_rate*T.grad(cost, self.sigma)),
        ]

        train_op = theano.function(
            inputs=[thx],
            updates=updates,
        )

        costs = []
        for it in range(max_iter):
            print("it:", it)

            for n in range(N):
                c = self.log_likelihood_multi(X).sum()
                print("c:", c)
                costs.append(c)
                train_op(X[n])

        print("A:", self.A.get_value())
        print("mu:", self.mu.get_value())
        print("sigma:", self.sigma.get_value())
        print("R:", self.R.get_value())
        print("pi:", self.pi.get_value())

        plt.plot(costs)
        plt.show()

    def set(self, pi, A, R, mu, sigma):
        self.pi = theano.shared(pi)
        self.A = theano.shared(A)
        self.R = theano.shared(R)
        self.mu = theano.shared(mu)
        self.sigma = theano.shared(sigma)
        M, K = R.shape
        self.M = M
        self.K = K

        D = self.mu.shape[2]
        twopiD = (2*np.pi)**D # Needed to calculate gaussian

        thx = T.matrix("X") # Represents T x D matrix of sequential observations

        # ---------- Calculate B emission matrix ----------
        # We need to find B for a particular sequence.
        # Recall that B is an M x T matrix
        # For each hidden state, M, at each tim, there is a probability of observing whatever
        # value was in our sequence
        # We have the following below:
        # - component_pdf -> finds a single value of B, at given hidden state and time
        # - state_pdfs ----> finds a column in B; for all hidden states, and a single time
        # - gmm_pdf -------> finds a full B matrix, M x T. Loops over all time steps and
        #                    determines each column of B, resulting in a final B matrix

        def mvn_pdf(x, mu, sigma):
            """Need to define our own mvn because it does not exist in theano."""
            k  = 1 / T.sqrt(twopiD * T.nlinalg.det(sigma))
            e = T.exp(-0.5*(x - mu).T.dot(T.nlinalg.matrix_inverse(sigma).dot(x-mu)))
            return k * e

        def gmm_pdf(x):
            def state_pdfs(xt):
                """Create the B observation matrix.

                Takes in x at a particular time t. Will perform a theano scan, iterating
                over all hidden states M, finding their respective B(j,t). """
                def component_pdf(j, xt):
                    """Takes in hidden state and x at a particular time t. This
                    function is going to calculate B(j, t). Note, this must be defined as a
                    function that so it can be passed into theano scan.

                    Args:
                        - j: current sequence in sequences pass in from scan below
                        - xt: non sequence argument passed in from scan below
                    """
                    Bj_t = 0
                    for k in range(self.K):
                        Bj_t += self.R[j, k] * mvn_pdf(xt, self.mu[j,k], self.sigma[j,k])
                    return Bj_t

                Bt, _ = theano.scan(
                    fn=component_pdf,
                    sequences=T.arange(self.M),
                    n_steps=self.M,
                    outputs_info=None,
                    non_sequences=[xt]
                )
                return Bt

            # Calculate full B matrix
            B, _ = theano.scan(
                fn=state_pdfs,
                sequences=x,
                n_steps=x.shape[0],
                outputs_info=None
            )

            return B.T

        B = gmm_pdf(thx)

        def recurrence_to_find_alpha(t, old_a, B):
            """Forward algorithm."""
            a = old_a.dot(self.A) * B[:, t]
            s = a.sum()
            return (a / s), s

        [alpha, scale], _ = theano.scan(
            fn=recurrence_to_find_alpha,
            sequences=T.arange(1, thx.shape[0]),
            outputs_info=[self.pi*B[:,0], None],
            n_steps=thx.shape[0] - 1,
            non_sequences=[B]
        )

        cost = -T.log(scale).sum()
        self.cost_op = theano.function(
            inputs=[thx],
            outputs=cost
        )
        return thx, cost

    def log_likelihood_multi(self, X):
        return np.array([self.cost_op(x) for x in X])


def real_signal():
    """
    Extracts raw audio from Wav file.

    Right click on the file and go to "get info", you will see that:
        - sampling rate = 16000 Hz
        - bits per sample = 16
        - The first is quanitization in time
        - The second is quantization in amplitude

    This is also done for images. 2^16 = 65536 is the number of different sound levels
    that we have.
    """
    spf = wave.open("helloworld.wav")

    signal = spf.readframes(-1)
    signal = np.fromstring(signal, "Int16")
    T = len(signal)
    signal = (signal - signal.mean()) / signal.std() # Normalize

    hmm = HMM(5, 3)

    # Signal needs to be of shape N x T(n) x D
    hmm.fit(signal.reshape(1, T, 1), learning_rate=10e-6, max_iter=20)


if __name__ == "__main__":
    real_signal()