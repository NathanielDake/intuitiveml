# Linear time algorithm
# Loop executes n (length of data) times
# Remaining operations are all constant
def find_max(data):
    biggest = data[0]
    for val in data:
        if val > biggest:
            biggest = val
    return biggest
