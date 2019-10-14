# Calling len() on a list is a constant time operation, and the simplest
# type of algorithm there is. It runs in O(1) time since the list class
# maintains an instance variable that maintains the length of the list
data = ["a", "b", "c", "d", "f"]
len(data)

# Accessing an element of a python list by index is also a constant time
# operation. Because python's list's are implemented as array based sequences
# references to a list's elements are stored in a consecutive block of memory.
# The index provided can be used as an offset into the underlying array.
# Computer hardware supports constant time access to an element based on its
# memory address.
print(data[2])