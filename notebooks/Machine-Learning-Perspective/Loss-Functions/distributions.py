import numpy as np

import matplotlib.pyplot as plt

import keras


color1 = '#6138ba'
color2 = 'pink'

BINS = np.arange(1, 12)
NUM_CLASSES = len(BINS)

EXPERIMENT_PARAMS = {
    'eps': 0.001,
    'n_samples': 100_000,
    'color': {'p': color1, 'q': color2}
}


class Distribution:

    def __init__(self, prob_dict: dict, eps=0.001):
        self.bins = BINS
        self.num_bins = len(self.bins)
        self.prob_dict = prob_dict
        self.eps = eps
        self.probs = self.prob_dict_to_array(self.prob_dict)

    def prob_dict_to_array(self, prob_dict):

        # Assert values are valid probability distribution
        assert np.isclose(sum(list(prob_dict.values())), 1, atol=0.0001)

        probs = np.zeros(self.num_bins)
        for bin_, p in prob_dict.items():
            probs[bin_ - 1] = p

        # Assert probs is valid distribution

        assert np.isclose(probs.sum(), 1, atol=0.0001)

        # Add a tiny amount of noise to replicate "label smoothing"
        if self.eps != 0:
            bins_with_zero_prob_mass = (probs == 0).sum()
            bins_with_nonzero_prob_mass = (probs != 0).sum()
            eps_remove_from_bins = self.eps / bins_with_nonzero_prob_mass
            eps_add_to_bins = self.eps / bins_with_zero_prob_mass
            probs = np.where(
                probs==0, 
                eps_add_to_bins, 
                probs - eps_remove_from_bins
            )

            # Assert probs is valid distribution
            assert np.isclose(probs.sum(), 1, atol=0.0001)

        return probs


    def sample(self, n_samples):
        # Sample the count of times each bin occurs
        return np.random.multinomial(n=n_samples, pvals=self.probs)


    def sample_ohe(self, n_samples):
        # Sample the count of times each bin occurs
        counts = self.sample(n_samples)

        # One hot encode
        l = []
        for i, x in enumerate(counts):
            l.extend([i] * x)
        one_hot_encoded_samples = keras.utils.to_categorical(l, num_classes=11)
        return one_hot_encoded_samples

    @classmethod
    def normalize_samples(cls, samples):
        return samples / samples.sum()

    @classmethod
    def normalize_one_hot_samples(cls, ohe_samples):
        ohe_samples_summed = ohe_samples.sum(axis=0)
        return ohe_samples_summed / ohe_samples_summed.sum()


def plot_distribution(p, color=color1):
    x = np.arange(1, len(p) + 1)
    plt.bar(x=x, height=p, alpha=0.5, label='p', color=color, edgecolor='black')
    plt.xticks(x)
    plt.xlabel('bin')
    plt.ylabel('Probability')
    plt.ylim(top=1.1)


def plot_distributions(distributions_list):

    # Currently only able to handle 4
    num_plots = len(distributions_list)
    assert num_plots <= 4

    fig, axs = plt.subplots(nrows=1, ncols=num_plots, figsize=(25, 4))

    for i, distribution_dict in enumerate(distributions_list):

        for name, distribution in distribution_dict.items():
            x = np.arange(1, len(distribution.probs) + 1)
            axs[i].bar(x=x, height=distribution.probs, alpha=0.5, label=name, color=EXPERIMENT_PARAMS['color'][name], edgecolor='black')

        axs[i].set_xticks(x)
        axs[i].set_ylim(top=1.1)
        axs[i].set_xlabel('Bin')
        axs[i].legend(fontsize=12)

        if i == 0:
            axs[i].set_ylabel('Probability')

        axs[i].set_title(f'#{i + 1}')

    plt.show()
