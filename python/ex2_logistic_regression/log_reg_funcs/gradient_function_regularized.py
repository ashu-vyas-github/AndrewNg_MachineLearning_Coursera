import numpy

import log_reg_funcs


def gradient_function_regularized(theta, x_data, y_data, regularization_lambda=1.0):

    num_examples = y_data.shape[0]
    regularized_term = numpy.zeros(theta.shape)

    hypothesis = log_reg_funcs.sigmoid_function(numpy.dot(x_data, theta))
    hypothesis_gradient = numpy.dot((hypothesis - y_data), x_data)/num_examples

    regularized_term[1:] = (regularization_lambda/num_examples)*theta[1:]

    gradient = hypothesis_gradient + regularized_term

    return gradient
