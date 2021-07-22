import numpy

import mcc_nn_funcs


def gradient_function_regularized(theta, x_data, y_data, regularization_lambda=1.0):

    num_examples = y_data.shape[0]
    regularized_term = numpy.zeros(theta.shape)

    hypothesis = mcc_nn_funcs.sigmoid_function(numpy.dot(theta, x_data.T))
    error = hypothesis - y_data
    hypothesis_gradient = numpy.dot(x_data.T, error)/num_examples

    regularized_term[1:] = (regularization_lambda/num_examples)*theta[1:]

    gradient = hypothesis_gradient + regularized_term

    return gradient
