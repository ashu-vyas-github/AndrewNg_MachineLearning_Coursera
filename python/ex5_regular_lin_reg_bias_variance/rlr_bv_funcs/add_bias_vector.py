import numpy


def add_bias_vector(unbiased_array):

    num_examples = unbiased_array.shape[0]
    bias_vector = numpy.ones((num_examples))
    biased_array = numpy.c_[bias_vector, unbiased_array]

    return biased_array
