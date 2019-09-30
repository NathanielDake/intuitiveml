from notebooks.computer_science.high_performance_python.profiling.utils_timing import timefn

@timefn
def range(start, stop, step=1):
    numbers = []
    while start < stop:
        numbers.append(start)
        start += step
    return numbers

@timefn
def xrange(start, stop, step=1):
    while start < stop:
        yield start
        start += step


for i in range(1, 10000):
    pass


for i in xrange(1, 10000):
    pass
