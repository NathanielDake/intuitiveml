import math
from math import sin

def test1(x):
    return math.sin(x)


def test2(x):
    return sin(x)


def test3(x, sin=math.sin):
    return sin(x)