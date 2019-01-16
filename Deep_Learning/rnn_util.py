import numpy as np
import string
import os
import sys
import operator
from nltk import pos_tag, word_tokenize
from datetime import datetime


def init_weight(Mi, Mo):
    """Initializes weights so that they are randomly distributed and have
    small enough values to prevent gradient descent from going crazy. 
    Takes in input size and output size. Returns an Mi x Mo matrix."""
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)


def all_parity_pairs(nbit):
    """Takes in the number of bits, generates all possible combinations of bits."""
    # total number of samples (Ntotal) will be a multiple of 100
    # why did I make it this way? I don't remember.
    N = 2**nbit
    remainder = 100 - (N % 100)
    Ntotal = N + remainder
    X = np.zeros((Ntotal, nbit))
    Y = np.zeros(Ntotal)
    for ii in range(Ntotal):
        i = ii % N
        # now generate the ith sample
        for j in range(nbit):
            if i % (2**(j+1)) != 0:
                i -= 2**j
                X[ii,j] = 1
        Y[ii] = X[ii].sum() % 2
    return X, Y


def all_parity_pairs_with_sequence_labels(nbit):
    X, Y = all_parity_pairs(nbit)
    N, t = X.shape

    # we want every time step to have a label
    Y_t = np.zeros(X.shape, dtype=np.int32)
    for n in range(N):
        ones_count = 0
        for i in range(t):
            if X[n,i] == 1:
                ones_count += 1
            if ones_count % 2 == 1:
                Y_t[n,i] = 1

    X = X.reshape(N, t, 1).astype(np.float32)
    return X, Y_t


def remove_punctuation(s):
    return s.translate(str.maketrans('','',string.punctuation))


def get_robert_frost():
    word2idx = {'START': 0, 'END': 1}
    current_idx = 2
    sentences = []
    for line in open('../../data/poems/robert_frost.txt'):
        line = line.strip()
        if line:
            tokens = remove_punctuation(line.lower()).split()
            sentence = []
            for t in tokens:
                if t not in word2idx:
                    word2idx[t] = current_idx
                    current_idx += 1
                idx = word2idx[t]
                sentence.append(idx)
            sentences.append(sentence)
    return sentences, word2idx


def get_tags(s):
    tuples = pos_tag(word_tokenize(s))
    return [y for x, y in tuples]


def get_poetry_classifier_data(samples_per_class, loaded_cached=True, save_cached=True):
    datafile = 'poetry_classifier.npz'
    if loaded_cached and os.path.exists(datafile):
        npz = np.load(datafile)
        X = npz['arr_0'] # Data
        Y = npz['arr_1'] # Targets, 0 or 1
        V = int(npz['arr_2']) # Vocabulary size
        return X, Y, V

    word2idx = {}
    current_idx = 0
    X = []
    Y = []
    for fn, label in zip(('../../data/poems/robert_frost.txt', '../../data/poems/edgar_allan_poe.txt'), (0,1 )):
        count = 0
        for line in open(fn):
            line = line.rstrip()
            if line:
                print(line)
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
                    print(count)
                    if count >= samples_per_class:
                        break
    if save_cached:
        np.savez(datafile, X, Y, current_idx)
    return X, Y, current_idx
