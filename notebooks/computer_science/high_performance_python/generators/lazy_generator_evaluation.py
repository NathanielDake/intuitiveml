from random import normalvariate, random
from itertools import count, groupby, islice
from datetime import date
import math


def read_data(filename):
    with open(filename) as fd:
        for line in fd:
            data = line.strip().split(",")
            yield map(int, data)


def read_fake_data(filename):
    """
    This function returns a generator.
    Example usage:
       gen = read_fake_date("test_file_name")
       a = gen.__next__()
       => a = (1, 0.98234)

    Note: docs on count -> https://docs.python.org/3/library/itertools.html#itertools.count
    Essentially a wrapper around a basic while True loop
    """
    for i in count():
        sigma = random() * 10
        yield (i, normalvariate(0, sigma))

def day_grouper(iterable):
    # lambda takes in a data point of form: time, value
    # Returns itertools groupby, which is an iterator with a next method
    key = lambda timestamp_value: date.fromtimestamp(timestamp_value[0])
    return groupby(iterable, key)


def check_anomaly(day_data_tuple):
    """
    Find mean, std, and maximum values for the day. Using a single pass (online)
    mean/std algorithm allows us to only read through the day's data once.

    Note: M2 = 2nd moment, variance
    """
    (day, day_data) = day_data_tuple

    n = 0
    mean = 0
    M2 = 0
    max_value = 0
    for timestamp, value in day_data:
        n += 1
        delta = value - mean
        mean = mean + (delta / n)
        M2 += delta * (value - mean)
        max_value = max(max_value, value)
    variance = M2 / (n - 1)
    standard_deviation = math.sqrt(variance)

    # Check if day's data is anomalous, if True return day
    if max_value > mean + 6 * standard_deviation:
        return day
    return False


def main():
    data = read_fake_data("test_filename")

    data_day = day_grouper(data)

    anomalous_dates = filter(None, map(check_anomaly, data_day))

    first_anomalous_date = anomalous_dates.__next__()
    print(first_anomalous_date)

    next_10_anomalous_dates = islice(anomalous_dates, 10)
    print(list(next_10_anomalous_dates))


if __name__ == "__main__":
    main()