import numpy as np
import tensorflow as tf

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
