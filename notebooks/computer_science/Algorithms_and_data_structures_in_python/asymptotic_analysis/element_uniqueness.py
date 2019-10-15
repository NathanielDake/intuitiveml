# Quadratic Implementation
def unique1(S):
    for j in range(len(S)):
        for k in range(j+1, len(S)):
            if S[j] == S[k]:
                return False  # Found duplicate pair
    return True  # if we reach this, elements were unique


# By utilizing sorting, we can create an implementation that is O(nlogn)
def unique2(S):

    temp = sorted(S)
    for j in range(1, len(temp)):
        if S[j-1] == S[j]:
            return False
        return True