# Quadratic Time
def prefix_average1(S):
    """Return list such that, for all j, A[j] equals average of S[0],...,S[j]."""

    n = len(S)
    A = [0] * n
    for j in range(n):
        total = 0
        for i in range(j + 1):
            total += S[i]
        A[j] = total / (j + 1)  # Record average
    return A


# Also Quadratic time. While this implementation greatly simplfies the presentation
# of the algorithm, it is worth noting that asymptotically the efficiency is no better!
# Even though the expression sum(S[0:j+1]) seems like a single command, it is a function
# call and an evaluation of that function takes O(j+1) time in this context.
# Technically, the computation of the slice S[0:j+1] also takes O(j+1) time,
# as it constructs a new list instance for storage. So, the running time of
# prefix_average2 is still dominated by a series of steps that take proportional to
# 1 + 2 + 3 + ... + n, and an outer loop of n, hence it is O(n^2)
def prefix_average2(S):

    n = len(S)
    A = [0] * n
    for j in range(n):
        A[j] = sum(S[0:j+1]) / (j + 1)  # Record average
    return A


# A Linear time implementation
# In the two quadratic implementations, the prefix sum is computed fresh for each value
# of j. That contributed O(j) for each j, leading to the quadratic behavior. Now, we
# maintain the current prefix sum dynamically; in other words, we never reset to total
# to be 0. The final implementation is O(n).
def prefix_average3(S):

    n = len(S)
    A = [0] * n
    total = 0
    for j in range(n):
        total += S[j]
        A[j] = total / (j+1)  # Record average
    return A