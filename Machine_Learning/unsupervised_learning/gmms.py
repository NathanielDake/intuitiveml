import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def gmm(X, K, max_iter=20, smoothing=1e-2):
    N, D = X.shape           # Get number of rows and columns in X
    M = np.zeros((K, D))     # Set means to zeros
    R = np.zeros((N, K))     # Set the responsibilities to zeros.
    C = np.zeros((K, D, D))  # Covariance matrix, 3 dimensional
    pi = np.ones(K) / K      # Uniform distribution

    # Iterate over all K gaussians
    for k in range(K):
        M[k] = X[np.random.choice(N)] # Set the means to random points of X
        C[k] = np.diag(np.ones(D))

    costs = np.zeros(max_iter)
    weighted_pdfs = np.zeros((N, K)) # Store pdf values ---> Numerator of responsibility, gamma

    for i in range(max_iter):

        # --------------- Step 1: Calculate Responsibilities ---------------
        for k in range(K):  # Iterate through all K gaussians
            for n in range(N):   # Iterate through all N data points
                weighted_pdfs[n, k] = pi[k]*multivariate_normal.pdf(X[n], M[k], C[k])

        for k in range(K):
            for n in range(N):
                R[n, k] = weighted_pdfs[n, k] / weighted_pdfs[n, :].sum()

        # ---------- Step 2: Re-Calculate parameters (pi, mu, cov) ----------
        for k in range(K):
            Nk = R[:, k].sum() # sum of all responsibilities for specific gaussian k
            pi[k] = Nk / N
            M[k] = R[:, k].dot(X) / Nk

            # Regularization for covariance
            C[k] = np.sum(R[n, k]*np.outer(X[n] - M[k], X[n] - M[k]) for n in range(N)) / Nk + np.eye(D)*smoothing

            # Calculate log likelihood!!!
            costs[i] = np.log(weighted_pdfs.sum(axis=1)).sum()
            if i > 0:
                if np.abs(costs[i] - costs[i - 1]) < 0.1:
                    break

    fig, ax = plt.subplots(figsize=(12, 8))
    plt.plot(costs)
    plt.title("Costs")
    plt.show()

    random_colors = np.random.random((K, 3))
    colors = R.dot(random_colors)
    fig, ax = plt.subplots(figsize=(12, 8))
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.show()

    print("pi:", pi)
    print("means:", M)
    print("covariances:", C)
    return R



def main():
    # Create 3 Gaussian distributed clusters
    D = 2
    s = 4
    mu1 = np.array([0, 0])
    mu2 = np.array([s, s])
    mu3 = np.array([0, s])

    N = 2000 # Number of samples
    X = np.zeros((N, D))
    X[:1200, :] = np.random.randn(1200, D)*2 + mu1  # Covariance = 2
    X[1200:1800, :] = np.random.randn(600, D) + mu2 # Covariance = 1
    X[1800:, :] = np.random.randn(200, D)*0.5 + mu3 # Covariance = 0.5

    gaussian_1 = X[:1200, :]
    gaussian_2 = X[1200:1800, :]
    gaussian_3 = X[1800:, :]

    fig, ax = plt.subplots(figsize=(12,8))
    plt.scatter(gaussian_1[:, 0], gaussian_1[:, 1], c="red")
    plt.scatter(gaussian_2[:, 0], gaussian_2[:, 1], c="blue")
    plt.scatter(gaussian_3[:, 0], gaussian_3[:, 1], c="green")

    plt.show()

    K = 3
    gmm(X, K)


if __name__ == "__main__":
    main()