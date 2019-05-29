import numpy as np
import matplotlib.pyplot as plt

from Machine_Learning.hmm.utils import get_obj_s3, random_normalized

class HMM:
    def __init__(self, M):
        self.M = M

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

        # ----------- Updated here with Scaling: Main Expectation Maximization Loop -------------
        costs = []
        for it in range(max_iter):

            # Lists for alpha and beta. Remember, learning updates variables for each
            # individual sequence, and they may be different lengths. Hence, we cannot
            # use a numpy matrix
            alphas = []
            betas = []
            scales = [] # New scales list
            logP = np.zeros(N) # Store log(P) instead of P

            # Loop through all examples
            for n in range(N):
                x = X[n]
                T = len(x)

                # ---- Alpha Steps -------
                # Alpha initialize
                scale = np.zeros(T)
                alpha = np.zeros((T, self.M))  # Alpha prime
                alpha[0] = self.pi * self.B[:, x[0]]
                scale[0] = alpha[0].sum()      # Calculate first scale
                alpha[0] /= scale[0]           # This is alpha hat. In latex equations they are
                                               # referred to as alpha prime and alpha hat, in
                                               # code we will just keep them as alpha

                # Alpha induction
                for t in range(1, T):
                    alpha_t_prime = alpha[t-1].dot(self.A) * self.B[:, x[t]]
                    scale[t] = alpha_t_prime.sum()
                    alpha[t] = alpha_t_prime / scale[t]

                # Finish forward algorithm - Calculate (log) probability of sequence
                logP[n] = np.log(scale).sum()
                alphas.append(alpha)
                scales.append(scale)

                # ---- Beta steps --------
                # Beta initialize
                beta = np.zeros((T, self.M))
                beta[-1] = 1

                # Beta induction
                for t in range(T-2, -1, -1):
                    beta[t] = self.A.dot(self.B[:, x[t+1]] * beta[t+1]) / scale[t+1]
                betas.append(beta)

            cost = np.sum(logP)
            costs.append(cost)

            # ---- Reestimate pi, A, B using new scaled updates --------
            # pi
            self.pi = np.sum((alphas[n][0] * betas[n][0]) for n in range(N)) / N

            # Update A and B directly since they don't depend on probabilities
            den_A = np.zeros((self.M, 1))
            den_B = np.zeros((self.M, 1))
            a_num = np.zeros((self.M, self.M))
            b_num = np.zeros((self.M, V))
            for n in range(N):
                x = X[n]
                T = len(x)

                # ----- A and B Denominator update -----
                den_A += (alphas[n][:-1] * betas[n][:-1]).sum(axis=0, keepdims=True).T
                den_B += (alphas[n] * betas[n]).sum(axis=0, keepdims=True).T

                # ----- A numerator update -------
                for i in range(self.M):
                    for j in range(self.M):
                        for t in range(T-1):
                            a_num[i,j] += (
                                alphas[n][t,i] * betas[n][t+1, j] *
                                self.A[i, j] * self.B[j, x[t+1]] / scales[n][t+1]
                            )
                # ----- B numerator update -------
                for j in range(self.M):
                    for t in range(T):
                        b_num[j, x[t]] += alphas[n][t,i] * betas[n][t,j]

            # ----- Update for A and B -------
            self.A = a_num / den_A
            self.B = b_num / den_B

        print("A learned from training: \n", self.A)
        print("B learned from training: \n", self.B)
        print("pi learned from training: \n", self.pi)

        plt.plot(costs)
        plt.xlabel("Iteration Number")
        plt.ylabel("Cost")
        plt.show()

    def log_likelihood(self, x):
        """UPDATED WITH SCALED VERSION."""
        T = len(x)
        scale = np.zeros(T)  # Define scale
        alpha = np.zeros((T, self.M))  # Define alpha
        alpha[0] = self.pi * self.B[:, x[0]]  # Define initial value of alpha prime
        scale[0] = alpha[0].sum()
        alpha[0] /= scale[0]  # alpha hat
        # Induction steps
        for t in range(1, T):
            alpha_t_prime = alpha[t - 1].dot(self.A) * self.B[:, x[t]]
            scale[t] = alpha_t_prime.sum()
            alpha[t] = alpha_t_prime / scale[t]
        return np.log(scale).sum()

    def log_likelihood_multi(self, X):
        """REMAINS THE SAME. Returns log likelihood of every observation."""
        return np.array([self.log_likelihood(x) for x in X])

    def get_state_sequence(self, x):
        """UPDATED WITH SCALING.
        This is the viterbi algorithm. Returns the most likely
        state sequence given observed sequence x."""
        T = len(x)
        if T == 0:
            states = np.zeros(T, dtype=np.int32)
            return states
        delta = np.zeros((T, self.M))
        psi = np.zeros((T, self.M))
        delta[0] = np.log(self.pi) + np.log(self.B[:,x[0]])

        # Loop through the rest of the times and states
        for t in range(1, T):
            for j in range(self.M):
                delta[t,j] = np.max(delta[t-1] + np.log(self.A[:,j])) + np.log(self.B[j, x[t]])
                psi[t,j] = np.argmax(delta[t-1] + np.log(self.A[:,j]))

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