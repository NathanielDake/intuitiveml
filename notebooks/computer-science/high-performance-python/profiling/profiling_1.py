"""
We will dig into profiling techniques by using the Julia Set:
* https://en.wikipedia.org/wiki/Julia_set

We will calculate each pixel based on the Julia set. Each pixel is defined
with a location being a complex number, z, with a corresponding grey-scale
intensity:

----> f(z) = z^2 + c

So, in other words the intensity is the complex number z, squared, plus
a constant c.

"""
import time
import pylab

# Area of complex space to investigate
x1, x2, y1, y2 = -1.8, 1.8, -1.8, 1.8
c_real, c_imag = -0.62772, -0.42193

# Create coordinate lists as input to calculation function
def calc_pure_python(desired_width, max_iterations):
    """Create a list of complex coordinates (zs) and complex
    parameters (cs), build Julia set, and display."""
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
    # Note that the initial condition is a constant and could easily be removed.
    # We use it to simulate a real-world scenario with several inputs to our function
    zs = []
    cs = []
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))

    print("Length of x: ", len(x))
    print("Total elements: ", len(zs))

    start_time = time.time()
    output = calculate_z_serial_purepython(max_iterations, zs, cs)
    end_time = time.time()
    secs = end_time - start_time

    print(calculate_z_serial_purepython.__name__ + " took", secs, "seconds")

    # This sum is expected for a 1000^2 grid with 300 iterations. It catches minor errors
    # we might introduce when we're working on a fixed set of input.
    assert sum(output) == 33219980

    return output


def calculate_z_serial_purepython(maxiter, zs, cs):
    """
    Calculate output list using Julia update rule.

    This is a CPU bound calculation function. This specifically is a serial implementation.
    W e have operations being satisfied one at a time, each one waiting for the previous
    operation to complete.
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
    output = calc_pure_python(desired_width=1000, max_iterations=300)

    fig = pylab.subplots(figsize=(15,10))
    pylab.subplot(141)
    pylab.imshow(output, cmap='gray_r')