import numpy as np
import matplotlib.pyplot as plt
import theano
import theano.tensor as T

from Machine_Learning.hmm.utils import get_obj_s3, random_normalized

class HMM:
    def __init__(self, M):
        self.M = M

    def fit(self, X, learning_rate=0.001, max_iter=10, V=None, p_cost=1.0, print_period=10):
        """Train HMM model using stochastic gradient descent."""
        print("hit")
        # Determine V, the vocabulary size
        if V is None:
            V = max(max(x) for x in X) + 1
        N = len(X)

        # Initialize HMM variables
        pi0 = np.ones(self.M) / self.M          # Initial state distribution
        A0 = random_normalized(self.M, self.M)  # State transition matrix
        B0 = random_normalized(self.M, V)       # Output distribution

        thx, cost = self.set(pi0, A0, B0)

        pi_update = self.pi - learning_rate * T.grad(cost, self.pi)
        pi_update = pi_update / pi_update.sum()

        A_update = self.A - learning_rate*T.grad(cost, self.A)
        A_update = A_update / A_update.sum(axis=1).dimshuffle(0, 'x')

        B_update = self.B - learning_rate*T.grad(cost, self.B)
        B_update = B_update / B_update.sum(axis=1).dimshuffle(0, 'x')

        updates = [
            (self.pi, pi_update),
            (self.A, A_update),
            (self.B, B_update),
        ]

        train_op = theano.function(
            inputs=[thx],
            updates=updates,
            allow_input_downcast=True
        )

        costs = []
        for it in range(max_iter):
            for n in range(N):
                c = self.get_cost_multi(X, p_cost).sum()
                costs.append(c)
                train_op(X[n])

        print("A learned from training: \n", self.A.get_value())
        print("B learned from training: \n", self.B.get_value())
        print("pi learned from training: \n", self.pi.get_value())


        plt.plot(costs)
        plt.xlabel("Iteration Number")
        plt.ylabel("Cost")
        plt.show()

    def get_cost(self, x):
        return self.cost_op(x)

    def get_cost_multi(self, X, p_cost=1.0):
        P = np.random.random(len(X))
        return np.array([self.get_cost(x) for x, p in zip(X, P) if p < p_cost])

    def log_likelihood(self, x):
        return - self.cost_op(x)

    def set(self, pi, A, B):
        # Create theano shared variables
        self.pi = theano.shared(pi)
        self.A = theano.shared(A)
        self.B = theano.shared(B)

        # Define input, a vector
        thx = T.ivector("thx")

        def recurrence(t, old_alpha, x):
            """Scaled version of updates for HMM"""
            alpha = old_alpha.dot(self.A) * self.B[:, x[t]]
            s = alpha.sum()
            return (alpha / s), s

        [alpha, scale], _ = theano.scan(
            fn=recurrence,
            sequences=T.arange(1, thx.shape[0]),
            outputs_info=[self.pi *  self.B[:, thx[0]], None],
            n_steps=thx.shape[0] - 1,
            non_sequences=thx,
        )

        cost = -T.log(scale).sum()      # Negative log likelihood

        self.cost_op = theano.function(
            inputs=[thx],
            outputs=cost,
            allow_input_downcast=True,
        )
        return thx, cost


def fit_coin(file_key):
    """Loads data and trains HMM."""

    X = []
    for line in get_obj_s3(file_key).read().decode("utf-8").strip().split(sep="\n"):
        x = [1 if e == "H" else 0 for e in line.rstrip()]
        X.append(x)

    # Instantiate object of class HMM with 2 hidden states (heads and tails)
    hmm = HMM(2)
    hmm.fit(X)
    L = hmm.get_cost_multi(X).sum()
    print("Log likelihood with fitted params: ", round(L, 3))

    # Try the true values
    pi = np.array([0.5, 0.5])
    A = np.array([
        [0.1, 0.9],
        [0.8, 0.2]
    ])
    B = np.array([
        [0.6, 0.4],
        [0.3, 0.7]
    ])
    hmm.set(pi, A, B)
    L = hmm.get_cost_multi(X).sum()
    print("Log Likelihood with true params: ", round(L, 3))


if __name__ == "__main__":
    key = "coin_data.txt"
    fit_coin(key)