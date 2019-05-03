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
    print("output: ", o_val)


