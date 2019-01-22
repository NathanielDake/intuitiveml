import numpy as np
import string
import os
import sys
import operator
from nltk import pos_tag, word_tokenize
from datetime import datetime

relative_path = '../../data/'

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
    for line in open(f'{relative_path}poems/robert_frost.txt'):
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
    for fn, label in zip((f'{relative_path}poems/robert_frost.txt', f'{relative_path}poems/edgar_allan_poe.txt'), (0,1 )):
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


def my_tokenizer(s):
    s = remove_punctuation(s)
    s = s.lower()
    return s.split()


def get_wikipedia_data(n_files, n_vocab, by_paragraph=False):
    """Converts Wikipedia txt files into correct format for Neural Network

    This function takes in a number of files that is too large to fit into memory if all data is loaded
    at once. 100 or less seems to be ideal. The vocabulary also needs to be limited, since it is a lot
    larger than the poetry dataset. We are going to have ~500,000-1,000,000 words. Note that the output
    target is the next word, so that is 1 million output classes, which is a lot of output classes.
    This makes it hard to get good accuracy, and it will make our output weight very large. To remedy
    this, the vocabulary size will be restricted to n_vocab. This is generally set to ~2000 most
    common words.

    Args:
        n_files: Number of input files taken in
        n_vocab: Vocabulary size
        by_paragraph:

    Returns:
        sentences: list of lists containing sentences mapped to index
        word2idx_small: word2index mapping reduced to size n_vocab

    """
    wiki_relative_path = f'{relative_path}wikipedia/unzipped'
    input_files = [f for f in os.listdir(wiki_relative_path) if f.startswith('enwiki') and f.endswith('txt')]

    # Return Variables
    sentences = []
    word2idx = {'START': 0, 'END': 1}
    idx2word = ['START', 'END']
    current_idx = 2
    word_idx_count = {0: float('inf'), 1: float('inf')}

    if n_files is not None:
        input_files = input_files[:n_files]

    for f in input_files:
        print('Reading: ', f )
        for line in open(f'{wiki_relative_path}/{f}'):
            line = line.strip()
            # Don't count headers, structured data, lists, etc
            if line and line[0] not in ('[', '*', '-', '|', '=', '{', '}'):
                if by_paragraph:
                    sentence_lines = [line]
                else:
                    sentence_lines = line.split('. ')
                for sentence in sentence_lines:
                    tokens = my_tokenizer(sentence)
                    for t in tokens:
                        if t not in word2idx:
                            word2idx[t] = current_idx
                            idx2word.append(t)
                            current_idx += 1
                        idx = word2idx[t]
                        word_idx_count[idx] = word_idx_count.get(idx, 0) + 1
                    sentence_by_idx = [word2idx[t] for t in tokens]
                    sentences.append(sentence_by_idx)

    # Reduce vocabulary size to n_vocab
    sorted_word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
    word2idx_small = {}
    new_idx = 0
    idx_new_idx_map = {}
    for idx, count in sorted_word_idx_count[:n_vocab]:
        word = idx2word[idx]
        print(word, count)
        word2idx_small[word] = new_idx
        idx_new_idx_map[idx] = new_idx
        new_idx += 1
    # Let 'unknown' be last token
    word2idx_small['UNKNOWN'] = new_idx
    unknown = new_idx

    assert('START' in word2idx_small)
    assert('END' in word2idx_small)
    assert('king' in word2idx_small)
    assert('queen' in word2idx_small)
    assert('man' in word2idx_small)
    assert('woman' in word2idx_small)

    # Map old idx to new idx
    sentences_small = []
    for sentence in sentences:
        if len(sentence) > 1:
            new_sentence = [idx_new_idx_map[idx] if idx in idx_new_idx_map else unknown for idx in sentence]
            sentences_small.append(new_sentence)

    return sentences_small, word2idx_small




