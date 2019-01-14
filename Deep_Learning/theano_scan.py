import numpy as np
import theano
import theano.tensor as T


def scan_example_a():
    x = T.vector('x') # x is vector, so every element in the loop will be a scalar

    def square(a):
        return a * a

    # Call theano scan
    outputs, updates = theano.scan(
        fn=square,
        sequences=x,
        n_steps=x.shape[0]
    )

    # Create theano function
    square_op = theano.function(
        inputs=[x],
        outputs=[outputs]
    )

    output_value = square_op(np.array([1, 2, 3, 4]))

    print('Output: ', output_value)


def recurrence(n, fn_1, fn_2):
    return fn_1 + fn_2, fn_1


def scan_example_b():
    N = T.iscalar('N')

    outputs, updates = theano.scan(
        fn=recurrence,
        sequences=T.arange(N),
        n_steps=N,
        outputs_info=[1., 1.]
    )

    fibonacci = theano.function(
        inputs=[N],
        outputs=outputs
    )

    output_value = fibonacci(8)

    print('Output: ', output_value)


if __name__ == '__main__':
    scan_example_b()