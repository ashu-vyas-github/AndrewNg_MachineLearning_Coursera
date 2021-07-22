import numpy

import log_reg_funcs


def gradient_function(theta, x_data, y_data):

    num_examples = y_data.shape[0]
    hypothesis = log_reg_funcs.sigmoid_function(numpy.dot(x_data, theta))
    gradient = numpy.dot((hypothesis - y_data), x_data)/num_examples

    return gradient
