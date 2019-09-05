import map_reduce
import string

def mapper(input_key, input_value):
    return [(word, 1) for word in remove_punctuation(input_value.lower()).split()]


def remove_punctuation(s):
    return s.translate(str.maketrans('', '', string.punctuation))


def reducer(intermediate_key, intermediate_value_list):
    return (intermediate_key, sum(intermediate_value_list))


if __name__ == "__main__":
    filenames = ["text_a.txt", "text_b.txt", "text_c.txt"]

    i = {}

    for filename in filenames:
        f = open(filename)
        i[filename] = f.read()
        f.close()

    print(map_reduce.map_reduce(i,mapper,reducer))