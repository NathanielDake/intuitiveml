import numpy as np
import matplotlib.pyplot as plt


def simple_init():
    # 1 state, 1 gaussian, 1 dimension
    M = 1
    K = 1
    D = 1

    pi = np.array([1])
    A = np.array([[1]])
    R = np.array([[1]])
    mu = np.array([[[0]]])
    sigma = np.array([[[[1]]]])

    return M, K, D, pi, A, R, mu, sigma


def big_init():
    # 5 hidden states, 3 different gaussians, 2 dimensions
    M = 5
    K = 3
    D = 2

    # initial state distribution
    pi = np.array([1, 0, 0, 0, 0])

    # State transition matrix - likes to stay where it is (0.9 across diagonals)
    A = np.array([
        [0.9, 0.025, 0.025, 0.025, 0.025],
        [0.025, 0.9, 0.025, 0.025, 0.025],
        [0.025, 0.025, 0.9, 0.025, 0.025],
        [0.025, 0.025, 0.025, 0.9, 0.025],
        [0.025, 0.025, 0.025, 0.025, 0.9],
    ])

    # Mixture responsibilities -> Uniform distribution
    R = np.ones((M, K)) / K

    # Gaussian means --> M x K x D
    mu = np.array([
        [[0, 0], [1, 1], [2, 2]],
        [[5, 5], [6, 6], [7, 7]],
        [[10, 10], [11, 11], [12, 12]],
        [[15, 15], [16, 16], [17, 17]],
        [[20, 20], [21, 21], [22, 22]],
    ])

    # Gaussian Covariances
    sigma = np.zeros((M, K, D, D))

    for m in range(M):
        for k in range(K):
            sigma[m, k] = np.eye(D)

    return M, K, D, pi, A, R, mu, sigma


def get_signals(N=20, T=100, init=big_init):
    """
    Get signals from HMM with GMM

    Args:
        - N: number of sequences
        - T: length of sequence
        - init: which init to call

    Pseudocode:
        for n in number_of_sequences:
            for t in length_of_sequence:
                Find next hidden state (sample from A)
                Find next gaussian based on current hidden state (sample from R)
                Calculate probability of x (randomly sample from selected gaussian)
    """

    M, K, D, pi, A, R, mu, sigma = init()

    X = []

    # Loop through every N
    for n in range(N):
        x = np.zeros((T, D))
        s = 0 # initial state must be 0, based on pi definition
        r = np.random.choice(K, p=R[s])
        x[0] = np.random.multivariate_normal(mu[s][r], sigma[s][r])

        # Loop through all time steps after time t = 0
        for t in range(1, T):
            s = np.random.choice(M, p=A[s])  # sample from A[s] to select next state using A
            r = np.random.choice(K, p=R[s])  # sample from R[s] to select next gaussian (choose mixture)
            x[t] = np.random.multivariate_normal(mu[s][r], sigma[s][r])  #  generate data point

        X.append(x)
    return X


if __name__ == "__main__":
    T = 500
    x = get_signals(1, T)[0]
    axis = range(T)
    plt.plot(axis, x[:, 0], axis, x[:, 1])
    plt.show()