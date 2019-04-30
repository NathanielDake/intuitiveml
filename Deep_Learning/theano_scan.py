import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt


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


def scan_example_b():
    N = T.iscalar('N')

    def recurrence(n, fn_1, fn_2):
        return fn_1 + fn_2, fn_1

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


def scan_example_c():
    X = 2 * np.random.randn(300) + np.sin(np.linspace(0, 3*np.pi, 300))
    plt.plot(X)
    plt.title('Original')
    plt.show()

    decay = T.scalar('decay')
    sequence = T.vector('sequence')

    def recurrence(x, last, decay):
        print(x, last, decay)
        return (1 - decay)*x + decay*last

    outputs, _ = theano.scan(
        fn=recurrence,
        sequences=sequence,
        n_steps=sequence.shape[0],
        outputs_info=[np.float64(0)],
        non_sequences=[decay]
    )

    lpf = theano.function(
        inputs=[sequence, decay],
        outputs=outputs
    )

    Y = lpf(X, 0.99)
    plt.plot(Y)
    plt.title('Filtered')
    plt.show()


def scan_example_d():
    sequence = T.vector('sequence')

    def recurrence(current_x, current_total):
        print("HIT")
        return current_x + current_total

    outputs, _ = theano.scan(
        fn=recurrence,
        sequences=sequence,
        outputs_info=[np.float64(0)]
    )

    calculate_sum = theano.function(
        inputs=[sequence],
        outputs=outputs
    )

    final_sum = calculate_sum(np.array([1,2,3,4]))

    print(final_sum)



if __name__ == '__main__':
    scan_example_c()