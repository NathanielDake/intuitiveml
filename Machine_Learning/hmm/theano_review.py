import numpy as np
import theano
import theano.tensor as T

# Non Theano Implementation
k = 2 # Raising entire np array to the 2 (squaring)
A = np.array(range(10))
result = 1
for i in range(k):
    result = result * A

# print("Non Theano result: ", result)

# Theano Scan Implementation
k = T.iscalar("k")
A = T.vector("A")

def recurrence(prior_result, A):
    return prior_result * A

result, updates = theano.scan(
    fn=recurrence,
    outputs_info=T.ones_like(A),
    non_sequences=A,
    n_steps=k
)

power = theano.function(
    inputs=[A, k],
    outputs=result,
    updates=updates
)

# print("Theano result, k = 2: ", power(range(10),2))
# print("Theano result: k = 4: ", power(range(10),4))

# -------- Simple Accumulation into a scalar -------------
up_to = T.iscalar("up_to")

# The general order of function parameters to fn in theano.scan() is:
# 1) sequences (if any)
# 2) prior result(s) (if needed)
# 3) non-sequences (if any)
#
# So, in this case first we will have:
# 1) The current sequence value that we are looping over. Our sequence in np.array([1,...,15])
# 2) Our current total/sum (all previously summed values in the array). This takes an initial
#    value of outputs_info (in this case 0)

def accumulate_by_adding(current_arange_val, sum_to_date):
    return current_arange_val + sum_to_date

seq = T.arange(up_to)

outputs_info = T.as_tensor_variable(np.asarray(0, seq.dtype)) # output starting at 0
scan_result, scan_updates = theano.scan(
    fn=accumulate_by_adding,
    outputs_info=outputs_info, # Starting output value (first thing passed to accumulate_by_adding- sum_to_date)
    sequences=seq # Looping over this sequence
)
triangular_sequence = theano.function(
    inputs=[up_to],
    outputs=scan_result
)

some_num = 15
print("Input to triangular sequence: ", T.arange(some_num).eval())
print(triangular_sequence(some_num))
