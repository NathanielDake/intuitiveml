import numpy as np
import matplotlib.pyplot as plt

from Machine_Learning.hmm.utils import get_obj_s3, random_normalized


class HMM:
    def __init__(self, M):
        self.M = M  # Number of hidden states

    def fit(self, X, max_iter=30):
        """Fit HMM instance to training date.

            Args:
                X: Training data
                max_iter: Controls number of iterations of expectation maximization
        """

        np.random.seed(123)

        # Get Vocabulary size and number of sequences
        V = max(max(x) for x in X) + 1
        N = len(X)

        # Initialize Matrices, pi is uniform
        self.pi = np.ones(self.M) / self.M
        self.A = random_normalized(self.M, self.M)
        self.B = random_normalized(self.M, V)
        print("Initial A: \n", self.A)
        print("Initial B: \n", self.B)

        # Enter main Expectation Maximization Loop
        costs = []
        for it in range(max_iter):

            # Lists for alpha and beta. Remember, learning updates variables for each
            # individual sequence, and they may be different lengths. Hence, we cannot
            # use a numpy matrix
            alphas = []
            betas = []

            # Probabilties
            P = np.zeros(N)

            # Loop through each observation
            for n in range(N):
                x = X[n]
                T = len(x)
                alpha = np.zeros((T, self.M))  # Alpha is indexed by time and state

                # Set 1st value of alpha. pi*B for all states and first observation.
                # Probability of starting in initial state and observing x[0]
                alpha[0] = self.pi * self.B[:, x[0]]

                # Loop through for each time after the initial time
                for t in range(1, T):
                    alpha[t] = alpha[t-1].dot(self.A)*self.B[:, x[t]]

                # At this point alpha has been calculated, can calculate the probability of the sequence
                P[n] = alpha[-1].sum()
                alphas.append(alpha)

                # Now, do the same thing for beta
                beta = np.zeros((T, self.M))
                beta[-1] = 1

                for t in range(T-2, -1, -1):
                    beta[t] = self.A.dot(self.B[:, x[t+1]] * beta[t+1])

                betas.append(beta)

            # Calculate total log likelihood
            cost = np.sum(np.log(P))
            costs.append(cost)

            # Reestimate pi, A, B
            self.pi = np.sum((alphas[n][0] * betas[n][0]) / P[n] for n in range(N)) / N

            # Keep track of all denominators and numerators for A and B updates
            den_a = np.zeros((self.M, 1))
            den_b = np.zeros((self.M, 1))
            a_num = 0
            b_num = 0

            for n in range(N):
                x = X[n]
                T = len(x)

                # Expand to for loop if this is unclear
                den_a += (alphas[n][:-1] * betas[n][:-1]).sum(axis=0, keepdims=True).T / P[n]
                den_b += (alphas[n] * betas[n]).sum(axis=0, keepdims=True).T / P[n]

                # Numerator for A
                a_num_n = np.zeros((self.M, self.M))
                for i in range(self.M):
                    for j in range(self.M):
                        for t in range(T-1):
                            a_num_n[i, j] += alphas[n][t, i] * self.A[i, j] * self.B[j, x[t+1]] * betas[n][t+1, j]
                a_num += a_num_n / P[n]

                # Numerator for B
                b_num_n = np.zeros((self.M, V))
                for i in range(self.M):  # loop through every state
                    for j in range(V):  # loop through every possible observation
                        for t in range(T):  # loop through every time
                            if x[t] == j:
                                b_num_n[i, j] += alphas[n][t, i] * betas[n][t, i]
                b_num += b_num_n / P[n]

            # Update A and B
            self.A = a_num / den_a
            self.B = b_num / den_b

        print("A learned from training: \n", self.A)
        print("B learned from training: \n", self.B)
        print("pi learned from training: \n", self.pi)

        plt.plot(costs)
        plt.xlabel("Iteration Number")
        plt.ylabel("Cost")
        plt.show()

    def likelihood(self, x):
        """ Forward algorithm. Likelihood function for one observation. Returns P(x | model). """
        T = len(x)
        alpha = np.zeros((T, self.M))
        alpha[0] = self.pi * self.B[:, x[0]]
        for t in range(1, T):
            alpha[t] = alpha[t-1].dot(self.A) * self.B[:, x[t]]

        return alpha[-1].sum()

    def likelihood_multi(self, X):
        """Calculates the likelihoods of all observations."""
        return np.array([self.likelihood(x) for x in X])

    def log_likelihood_multi(self, X):
        """Returns log likelihood of every observation."""
        return np.log(self.likelihood_multi(X))

    def get_state_sequence(self, x):
        """Viterbi Algorithm. Returns the most likely hidden state sequence given observed sequence X."""
        T = len(x)
        delta = np.zeros((T, self.M))
        psi = np.zeros((T, self.M))
        delta[0] = self.pi * self.B[:, x[0]]

        # Loop through every other time, loop through all states
        for t in range(1, T):
            for j in range(self.M):
                delta[t, j] = np.max(delta[t-1]*self.A[:, j] * self.B[j, x[t]])
                psi[t, j] = np.argmax(delta[t-1]*self.A[:, j])

        # Backtrack
        states = np.zeros(T, dtype=np.int32)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states


def fit_coin(file_key):
    """Loads data and trains HMM."""

    X = []
    for line in get_obj_s3(file_key).read().decode("utf-8").strip().split(sep="\n"):
        x = [1 if e == "H" else 0 for e in line.rstrip()]
        X.append(x)

    # Instantiate object of class HMM with 2 hidden states (heads and tails)
    hmm = HMM(2)
    hmm.fit(X)
    L = hmm.log_likelihood_multi(X).sum()
    print("Log likelihood with fitted params: ", round(L, 3))

    # Try the true values
    hmm.pi = np.array([0.5, 0.5])
    hmm.A = np.array([
        [0.1, 0.9],
        [0.8, 0.2]
    ])
    hmm.B = np.array([
        [0.6, 0.4],
        [0.3, 0.7]
    ])
    L = hmm.log_likelihood_multi(X).sum()
    print("Log Likelihood with true params: ", round(L, 3))

    # Viterbi
    print("Best state sequence:\n", str(X[0]).replace(",",""))
    print("", hmm.get_state_sequence(X[0]))


if __name__ == "__main__":
    key = "coin_data.txt"
    fit_coin(key)
