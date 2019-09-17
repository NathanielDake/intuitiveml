import numpy as np
from datetime import datetime
from scipy.stats import multivariate_normal as mvn

from Machine_Learning.supervised_learning.utils import get_mnist_data


class Bayes(object):
    def fit(self, X, Y, smoothing=10e-3):
        N, D = X.shape
        self.gaussians = {}
        self.priors = {}
        class_labels = set(Y)

        for c in class_labels:
            # Loop through all training examples belonging to a particular class
            # Find mean and variance
            current_x = X[Y == c]
            self.gaussians[c] = {
                "mean": current_x.mean(axis=0),
                "cov": np.cov(current_x.T) + np.eye(D) * smoothing  # Calculating full covariance
            }
            self.priors[c] = float(len(Y[Y == c])) / len(Y)  # Could calculate log prior

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        N, D = X.shape
        K = len(self.gaussians) # Number of classes
        P = np.zeros((N, K)) # (holds probability of each sample, n, belonging to class, k)

        for class_, gaussian in self.gaussians.items():
            mean, cov = gaussian["mean"], gaussian["cov"]

            # Calculating N different log pdfs at the same time, using full covariance
            P[:, class_] = mvn.logpdf(X, mean=mean, cov=cov) + np.log(self.priors[class_])

        return np.argmax(P, axis=1)


if __name__ == "__main__":
    X, Y = get_mnist_data()

    Ntrain = len(Y) // 2
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]


    nb_model = Bayes()
    t0 = datetime.now()
    nb_model.fit(Xtrain, Ytrain)
    print("Training time:", (datetime.now() - t0))

    t0 = datetime.now()
    print("Train accuracy:", nb_model.score(Xtrain, Ytrain))
    print("Time to compute train accuracy:", (datetime.now() - t0), "Train size:", len(Ytrain))

    t0 = datetime.now()
    print("Test accuracy:", nb_model.score(Xtest, Ytest))
    print("Time to compute test accuracy:", (datetime.now() - t0), "Test size:", len(Ytest))