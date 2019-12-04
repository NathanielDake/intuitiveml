import numpy as np
from scipy.stats import ks_2samp

sample_size = 100_000

mean1 = 0.98
mean2 = 1
var1 = 1
var2 = 1

samp1 = np.random.normal(mean1, var1, sample_size)
samp2 = np.random.normal(mean2, var2, sample_size)

d, p = ks_2samp(samp1, samp2)


d, p = ks_2samp(samp1, samp2)