import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn

from Machine_Learning.hmm.generate_hmm_continuous import get_signals, big_init, simple_init
from Machine_Learning.hmm.utils import random_normalized


class HMM:
    def __init__(self, M, K):
        """
        :param M: Number of hidden states
        :param K: Number of gaussians
        """
        self.M = M
        self.K = K

    def fit(self, X, max_iter=30, eps=10e-1):
        N = len(X)
        D = X[0].shape[1]

        self.pi = np.ones(self.M) / self.M # Uniform distribution
        self.A = random_normalized(self.M, self.M)

        # GMM parameters. mu is set similar to how it is set in kmeans -> randomly select points from dataset
        # R, responsibilities --> Uniform distribution
        self.R = np.ones((self.M, self.K)) / self.K
        self.mu = np.zeros((self.M, self.K, D))
        for i in range(self.M):
            for k in range(self.K):
                # For all states and all gaussians, get a random index, choose a random sequence,
                # get a random time, and set mu[i,k] to be whatever point was at that time
                random_idx = np.random.choice(N)
                x = X[random_idx]
                random_time_idx = np.random.choice(len(x))
                self.mu[i,k] = x[random_time_idx]

        self.sigma = np.zeros((self.M, self.K, D, D))
        for j in range(self.M):
            for k in range(self.K):
                self.sigma[j,k] = np.eye(D)

        costs = []
        for it in range(max_iter):
            if it % 1 == 0:
              print("it: ", it)

            alphas = []
            betas = []
            gammas = []
            Bs = []
            P = np.zeros(N) # Sequence probabilities

            # ----------- Expectation Step -----------
            # Iterate over every sequence
            for n in range(N):
                x = X[n]
                T = len(x)

                B = np.zeros((self.M, T))
                component = np.zeros((self.M, self.K, T))

                # Iterate over every state, every time, and every gaussian
                for j in range(self.M):
                    for t in range(T):
                        for k in range(self.K):
                            p = self.R[j,k] * mvn.pdf(x[t], self.mu[j,k], self.sigma[j,k]) # Component probability
                            component[j,k,t] = p
                            B[j,t] += p
                Bs.append(B)

                # Just like discrete case
                alpha = np.zeros((T, self.M))
                alpha[0] = self.pi * B[:, 0]
                for t in range(1, T):
                    alpha[t] = alpha[t-1].dot(self.A) * B[:,t]
                P[n] = alpha[-1].sum()
                assert(P[n] <= 1)
                alphas.append(alpha)

                beta = np.zeros((T, self.M))
                beta[-1] = 1
                for t in range(T-2, -1, -1):
                    beta[t] = self.A.dot(B[:, t+1] * beta[t+1])
                betas.append(beta)

                # This was not needed in the discrete case
                # TODO: Ask lazy programmer for clarification regarding gamma. Is it responsibility,
                # or is it the gamma variable from discrete update?

                gamma = np.zeros((T, self.M, self.K))
                for t in range(T):
                    # Denominator only depends on t
                    alphabeta = (alphas[n][t,:] * betas[n][t,:]).sum()
                    for j in range(self.M):
                        # Now loop through every state and calculate alpha beta factor
                        factor = alphas[n][t,j] * betas[n][t,j] / alphabeta
                        for k in range(self.K):
                            # loop through all gaussians
                            gamma[t,j,k] = factor * component[j,k,t] / B[j,t]
                gammas.append(gamma)

            cost = np.log(P).sum()
            costs.append(cost)

            # ----------- Maximization Step -----------
            self.pi = np.sum((alphas[n][0] * betas[n][0]) / P[n] for n in range(N)) / N

            # Define numerators and denominators, since all updates formulas involve division
            a_den = np.zeros((self.M, 1))
            a_num = 0
            r_num = np.zeros((self.M, self.K))
            r_den = np.zeros(self.M)
            mu_num = np.zeros((self.M, self.K, D))
            sigma_num = np.zeros((self.M, self.K, D, D))
            # Note the denominator for mu and sigma is just r_num

            for n in range(N):
                # iterate over all sequences
                x = X[n]
                T = len(x)
                B = Bs[n]
                gamma = gammas[n]

                a_den += (alphas[n][:-1] * betas[n][:-1]).sum(axis=0, keepdims=True).T / P[n]

                # Update A -> This is the same update that was performed in the discrete case!
                a_num_n = np.zeros((self.M, self.M))
                for i in range(self.M):
                    for j in range(self.M):
                        for t in range(T-1):
                            a_num_n[i,j] += alphas[n][t,i] * self.A[i,j] * B[j,t+1] * betas[n][t+1,j]
                a_num += a_num_n / P[n]

                # Update mixture components
                r_num_n = np.zeros((self.M, self.K))
                r_den_n = np.zeros(self.M)
                for j in range(self.M):
                    for k in range(self.K):
                        for t in range(T):
                            r_num_n[j,k] += gamma[t,j,k]
                            r_den_n[j] += gamma[t,j,k]
                r_num += r_num_n / P[n]
                r_den += r_den_n / P[n]

                mu_num_n = np.zeros((self.M, self.K, D))
                sigma_num_n = np.zeros((self.M, self.K, D, D))
                for j in range(self.M):
                    for k in range(self.K):
                        for t in range(T):
                            mu_num_n[j,k] += gamma[t, j, k] * x[t]
                            sigma_num_n[j,k] += gamma[t,j,k]*np.outer(x[t] - self.mu[j,k], x[t] - self.mu[j,k])
                mu_num += mu_num_n / P[n]
                sigma_num += sigma_num_n / P[n]

            # Updates
            self.A = a_num / a_den
            for j in range(self.M):
                for k in range(self.K):
                    self.R[j,k] = r_num[j,k] / r_den[j]
                    self.mu[j,k] = mu_num[j,k] / r_num[j,k]
                    self.sigma[j,k] = sigma_num[j,k] / r_num[j,k]

        print("A:", self.A)
        print("mu:", self.mu)
        print("sigma:", self.sigma)
        print("R:", self.R)
        print("pi:", self.pi)

        plt.plot(costs)
        plt.show()

    def likelihood(self, x):
        # returns log P(x | model)
        # using the forward part of the forward-backward algorithm
        T = len(x)
        alpha = np.zeros((T, self.M))

        B = np.zeros((self.M, T))
        for j in range(self.M):
            for t in range(T):
                for k in range(self.K):
                    p = self.R[j,k] * mvn.pdf(x[t], self.mu[j,k], self.sigma[j,k])
                    B[j,t] += p

        alpha[0] = self.pi*B[:,0]
        for t in range(1, T):
            alpha[t] = alpha[t-1].dot(self.A) * B[:,t]
        return alpha[-1].sum()

    def likelihood_multi(self, X):
        return np.array([self.likelihood(x) for x in X])

    def log_likelihood_multi(self, X):
        return np.log(self.likelihood_multi(X))

    def set(self, pi, A, R, mu, sigma):
        self.pi = pi
        self.A = A
        self.R = R
        self.mu = mu
        self.sigma = sigma
        M, K = R.shape
        self.M = M
        self.K = K

    def get_state_sequence(self, x):
        # returns the most likely state sequence given observed sequence x
        # using the Viterbi algorithm
        T = len(x)

        # make the emission matrix B
        B = np.zeros((self.M, T))
        for j in range(self.M):
            for t in range(T):
                for k in range(self.K):
                    p = self.R[j,k] * mvn.pdf(x[t], self.mu[j,k], self.sigma[j,k])
                    B[j,t] += p

        # perform Viterbi as usual
        delta = np.zeros((T, self.M))
        psi = np.zeros((T, self.M))
        delta[0] = self.pi*B[:,0]
        for t in range(1, T):
            for j in range(self.M):
                delta[t,j] = np.max(delta[t-1]*self.A[:,j]) * B[j,t]
                psi[t,j] = np.argmax(delta[t-1]*self.A[:,j])

        # backtrack
        states = np.zeros(T, dtype=np.int32)
        states[T-1] = np.argmax(delta[T-1])
        for t in range(T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        return states

def fake_signal(init=simple_init):
    signals = get_signals(N=10, T=10, init=init)

    hmm = HMM(2, 2)
    hmm.fit(signals)
    L = hmm.log_likelihood_multi(signals).sum()
    print("LL for fitted params:", L)

    # test in actual params
    # _, _, _, pi, A, R, mu, sigma = init()
    # hmm.set(pi, A, R, mu, sigma)
    # L = hmm.log_likelihood_multi(signals).sum()
    # print("LL for actual params:", L)

    # print most likely state sequence
    # print("Most likely state sequence for initial observation:")
    # print(hmm.get_state_sequence(signals[0]))


if __name__ == '__main__':
    # real_signal() # will break
    fake_signal(init=simple_init)

