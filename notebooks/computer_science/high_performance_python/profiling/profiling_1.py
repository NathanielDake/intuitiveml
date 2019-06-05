"""
We will dig into profiling techniques by using the Julia Set:
* https://en.wikipedia.org/wiki/Julia_set

We will calculate each pixel based on the Julia set. Each pixel is defined
with a location being a complex number, z, with a corresponding grey-scale
intensity:

----> f(z) = z^2 + c

So, in other words the intensity is the complex number z, squared, plus
a constant c.

The specs of the laptop that this is being run on are as follows:
* MacBook Pro, 2018
* Processor: Intel Core i9
    - Speed:               2.9 GHz
    - Number of cores:     6
    - L2 cache (per core): 256 KB
    - L3 Cache:            12 MB
* Memory:
    - Size:                32 GB
    - Speed:               2400 MHz DDR4
"""
import time

# from notebooks.computer_science.high_performance_python.profiling.utils_plotting import plot_julia
# from notebooks.computer_science.high_performance_python.profiling.utils_timing import timefn

# Area of complex space to investigate
x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
c_real, c_imag = -0.62772, -0.42193


def calc_pure_python(desired_width, max_iterations):
    """
    Create a list of complex coordinates (zs) and complex parameters (cs), build Julia set, and display.

    Notes:
        - The lists, x and y, are created entirely from scratch. Generally this could be accomplished
          with the range function, but the goal is for this to be as transparent as possible.
        - Lists will have shape -> x = [-1.8,...,1.8], y = [-1.8,...,1.8]
        - x and y are converted to the real and imaginary coords of a complex number
        - The initial condition is a constant and could easily be removed, but we use it simulate
          a real-world scenario with several inputs to our function
        - We then time calculate_z_serial_purepython, which actually calculates the Julia output
          for a given complex coordinate
        - Finally, check to see if the output is what we expect. The Julia set is deterministic so
          this check can be made.
    """

    # Create coordinate lists as input to calculation function
    x_step = (float(x2 - x1) / float(desired_width))
    y_step = (float(y1 - y2) / float(desired_width))
    x = []
    y = []
    ycoord = y2
    while ycoord > y1:
        y.append(ycoord)
        ycoord += y_step

    xcoord = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step

    # Build a list of coordinates and the initial condition for each cell.
    zs = []
    cs = []
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))

    print("Length of x: ", len(x))
    print("Total elements: ", len(zs))

    # Time the building of the Julia set
    start_time = time.time()
    output = calculate_z_serial_purepython(max_iterations, zs, cs)
    end_time = time.time()
    secs = end_time - start_time

    print(calculate_z_serial_purepython.__name__ + " took", secs, "seconds")
    assert sum(output) == 33219980 # This sum is expected for a 1000^2 grid with 300 iterations.

    return output, zs


def calculate_z_serial_purepython(maxiter, zs, cs):
    """
    Calculate output list using Julia update rule.

    Args:
        - maxiter: max number of iterations before breaking. This is to prevent iterating to
          infinity, which is possible with the julia set.
        - zs: Complex coordinate grid --> real = [-1.8,...,1.8], imaginary = [-1.8j, 1.8j)
        - cs: list of constants

    This is a CPU bound calculation function. This specifically is a serial implementation.
    We have operations being satisfied one at a time, each one waiting for the previous
    operation to complete.

    For each complex coordinate in list zs, while the condition abs(z) < 0 holds perform update
    rule z = z^2 + c and count the number of times this occurs before the condition breaks.
    """
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while abs(z) < 2 and n < maxiter:
            z = z*z + c
            n += 1
        output[i] = n

    return output

if __name__ == "__main__":
    # Calculate julia set using a pure python solution with
    # reasonable defaults for a laptop
    output, zs = calc_pure_python(desired_width=1000, max_iterations=300)

    # plot_julia(output, 1000)


