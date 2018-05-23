import math
import numpy

def polynomial_kernel(x, y):
    return math.pow(numpy.dot(x, y) + 1, 4)