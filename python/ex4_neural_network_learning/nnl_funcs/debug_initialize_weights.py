import numpy


def debug_initialize_weights(fan_out, fan_in):

    # Initialize weights using "sin". This ensures that weights is always of the same values and will be useful for debugging
    weights = numpy.sin(numpy.arange(1, 1 + (1+fan_in)*fan_out))/10.0
    weights = weights.reshape(fan_out, 1+fan_in, order='F')
    return weights
