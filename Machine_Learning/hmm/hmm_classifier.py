import numpy as np

from sklearn.utils import shuffle
from nltk import pos_tag, word_tokenize

from Machine_Learning.hmm.hmm_theano import HMM
from Machine_Learning.hmm.utils import get_obj_s3


class HMMClassifier():
    def __init__(self):
        pass

    def fit(self, X, Y, V):
        K = len(set(Y)) # Number of classes
        N = len(Y)
        self.models = []
        self.priors = [] # Priors are for bayes rule
        for k in range(K):
            # Loop through all classes
            thisX = [x for x, y in zip(X, Y) if y == k] # All x's the belong to class k
            C = len(thisX) # Number of elements of class k
            self.priors.append(np.log(C) - np.log(N)) # Not technically the prior

            hmm = HMM(5) # Create an HMM with 5 hidden states
            hmm.fit(thisX, V=V, p_cost=0.1, print_period=1, learning_rate=10e-5, max_iter=100)
            self.models.append(hmm)

    def score(self, X, Y):
        N = len(Y)
        correct = 0
        for x, y in zip(X, Y):
            posteriors = [hmm.log_likelihood(x) + prior for hmm, prior in zip(self.models, self.priors)]
            p = np.argmax(posteriors)
            if p == y:
                correct += 1

        return float(correct) / N


def get_tags(s):
    """Determines parts of speech tags for a given string."""
    word_tag_tuples = pos_tag(word_tokenize(s))
    return [tag for word, tag in word_tag_tuples]

def get_data():
    """Gather's blocks of text for each author, and determines the POS tags for each."""
    word2idx = {}
    current_idx = 0
    X = [] # Sequences
    Y = [] # Labels

    for file_name, label in zip(("robert_frost.txt", "edgar_allan_poe.txt"), (0,1)):
        count = 0
        for line in get_obj_s3(file_name).read().decode("utf-8").split("\n"):
            line = line.rstrip()
            if line:
                tokens = get_tags(line)
                if len(tokens) > 1:
                    for token in tokens:
                        if token not in word2idx:
                            word2idx[token] = current_idx
                            current_idx += 1
                    sequence = np.array([word2idx[w] for w in tokens])
                    X.append(sequence)
                    Y.append(label)
                    count += 1
                    if count >= 50:
                        break

    print("Vocabulary: ", word2idx.keys())
    return X, Y, current_idx


def main():
    X, Y, V = get_data()

    # We will not be using the words directly because there are so many of them
    # Rather, we will use parts of speech tagging instead
    X, Y = shuffle(X, Y)
    N = 20 # Number of test samples
    Xtrain, Ytrain = X[:-N], Y[:-N]
    Xtest, Ytest = X[-N:], Y[-N:]

    model = HMMClassifier()
    model.fit(Xtrain, Ytrain, V)
    print("Score: ", model.score(Xtest, Ytest))


if __name__ == "__main__":
    main()

