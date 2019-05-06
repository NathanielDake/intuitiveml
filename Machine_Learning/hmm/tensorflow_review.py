import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x = tf.placeholder(tf.int32, shape=(None,), name="x")

def square(last, current):
    """last is never used, but must be included based on interface requirements of tf.scan"""
    return current*current

# Essentially doing what a for loop would normally do
# It applies the square function to every element of x
square_op = tf.scan(
    fn=square,
    elems=x
)

# Run it!
with tf.Session() as session:
    o_val = session.run(
        square_op,
        feed_dict={x: [1, 2, 3, 4, 5]}
    )
    print("Output: ", o_val)


# ----------------------- Fibonacci -----------------------
# N is the number fibonacci numbers that we want
N = tf.placeholder(tf.int32, shape=(), name="N")

def fibonacci(last, current):
    # last[0] is the last value, last[1] is the second last value
    return (last[1], last[0] + last[1])


fib_op = tf.scan(
    fn=fibonacci,
    elems=tf.range(N),
    initializer=(0, 1),
)

with tf.Session() as session:
    o_val = session.run(
        fib_op,
        feed_dict={N: 8}
    )
    print("Output: ", o_val)


# ----------------------- Low Pass Filter -----------------------
original = np.sin(np.linspace(0, 3*np.pi, 300))
X = 2*np.random.randn(300) + original

fig = plt.figure(figsize=(15,5))
plt.subplot(1, 2, 1)
ax = plt.plot(X)
plt.title("Original")

# Setup placeholders
decay = tf.placeholder(tf.float32, shape=(), name="decay")
sequence = tf.placeholder(tf.float32, shape=(None, ), name="sequence")

# The recurrence function and loop
def recurrence(last, x):
    return (1.0 - decay)*x + decay*last

low_pass_filter = tf.scan(
    fn=recurrence,
    elems=sequence,
    initializer=0.0 # sequence[0] to use first value of the sequence
)

# Run it!
with tf.Session() as session:
    Y = session.run(low_pass_filter, feed_dict={sequence: X, decay: 0.97})

    plt.subplot(1, 2, 2)
    ax = plt.plot(Y)
    ax2 = plt.plot(original)
    plt.title("Low pass filter")
    plt.show()