# The python for loop
object_list = ["a", "b", "c", "d"]

for i in object_list:
    print(i)

# The above for loop is equivalent to
object_iterator = iter(object_list)

while True:
    try:
        i = object_iterator.__next__()
        print(i)
    except StopIteration:
        break