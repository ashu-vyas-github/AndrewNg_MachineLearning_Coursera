import numpy


def sigmoid_function(x_var):

    numr = 1.0
    denr = 1.0 + numpy.exp(-x_var)
    sigmoid_transform = numr/denr
    return sigmoid_transform
