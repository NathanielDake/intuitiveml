# We can also look at find max from another point of view; how
# many times will biggest by updated?
# https://pdfs.semanticscholar.org/0706/5418a63503910e72f3ec7cf97c991367f133.pdf
# https://en.wikipedia.org/wiki/Harmonic_number


import random
import matplotlib.pyplot as plt
import numpy

def find_max(data):
    biggest = data[0]
    biggest_update_count = 0
    for val in data:
        if val > biggest:
            biggest = val
            biggest_update_count += 1

    return biggest, biggest_update_count


if __name__ == "__main__":
    xs = []
    ys = []
    num_experiments = 2
    for i in range(0, num_experiments):
        x = numpy.array([])
        y = numpy.array([])

        for n in [10, 100, 1000, 10_000, 100_000]:
            data = [random.randrange(1, 50, 1) for i in range(0, n)]
            _, biggest_update_count = find_max(data)
            x = numpy.append(x, n)
            y = numpy.append(y, biggest_update_count)

        xs.append(x)
        ys.append(y)

    for idx, x in numpy.ndenumerate(xs):
        x = x.reshape(5, 1)
        xs[idx] = x



    for y in ys:
        y = y.reshape(5, 1)



    plt.plot(x, y)
    plt.show()