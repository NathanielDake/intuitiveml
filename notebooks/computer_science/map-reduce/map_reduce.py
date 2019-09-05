import itertools


def key_func(x):
    """
    Used to group data. Takes in a tuple and returns the first value. This is the key that
    is used to group on (in this case a specific word).
    """
    return x[0]


def map_reduce(input, mapper, reducer):
    """
    Map example output:
    [('the', 1),
     ('quick', 1),
     ('brown', 1),
     ('fox', 1),
     ('jumped', 1),
     ('over', 1),
     ('the', 1),
     ('lazy', 1),
     ('grey', 1),
     ('dogs', 1),
     ('mary', 1)]

     Intermediate example output:
    {'a': [1, 1],
     'and': [1],
     'as': [1],
     'brown': [1],
     ............,
     'the': [1, 1, 1],
     'to': [1],
     'was': [1, 1]}

     Reduce example output:
     [('a', 2),
     ('and', 1),
     ('as', 1),
     ('brown', 1),
     ............,
     ('the', 3),
     ('to', 1),
     ('was', 2)]
    """

    # Map phase
    intermediate = []
    for (key, value) in input.items():
        intermediate.extend(mapper(key, value))

    # Intermediate Phase - groupby will return a key, group iterator pair for each word. Each iterator
    # will contain a group that can then be iterated over.
    groups = {}

    for key, group in itertools.groupby(sorted(intermediate), key_func):
        groups[key] = list([y for x, y in group])

    # Reduce Phase
    return [reducer(intermediate_key, intermediate_group) for (intermediate_key, intermediate_group) in groups.items()]

