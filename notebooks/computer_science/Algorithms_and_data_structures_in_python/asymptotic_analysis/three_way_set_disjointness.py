# Cubic implementaion, if each set has size n then we are dealing with O(n^3)
def disjoint1(A, B, C):

    for a in A:
        for b in B:
            for c in C:
                if a == b == c:
                    return False  # Common value was found
    return True  # Reaching this means sets are disjoint


# Quadratic Implementation
# We can improve upon disjoint1 by making the observation that once inside the body
# of the loop over B, if selected elements a and b do not match each other, it is a
# waste of time to iterate through all values of C looking for a matching triple.
#
# We can actually make the claim that this implementation below is O(n^2).
# There are quadratically many pairs (a,b) to consider. However, since we know
# that A and B are each sets of distinct elements, there can be at most O(n)
# such pairs with a equal to b. Therefore, the innermost loop, over C, executes
# at most n times.
def disjoint2(A, B, C):

    for a in A:
        for b in B:
            if a == b:  # Only check C if we find match in A and B
                for c in C:
                    if a == c:  # And thus a == b == c
                        return False  # We found common value
    return True  # Sets are disjoint

