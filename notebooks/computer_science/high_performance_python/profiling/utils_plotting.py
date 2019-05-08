import matplotlib.pyplot as plt
import numpy as np


def plot_julia(output, width):
    output = np.array(output).reshape(width, width)


    plt.imshow(
        output,
        cmap=plt.get_cmap('jet'),
        vmin=0,
        vmax=300
    )
    plt.show()