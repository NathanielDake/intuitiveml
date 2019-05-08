import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random


def plot_julia():
    data = []
    for i in range(2500):
        data.append(random.randint(0,1))

    plt.imshow(np.array(data).reshape(50, 50), cmap=cm.gray)
    plt.show()