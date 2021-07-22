import numpy

import log_reg_funcs


def cost_function_regularized(theta, x_data, y_data, regularization_lambda=1.0):

    num_examples = y_data.shape[0]
    hypothesis = log_reg_funcs.sigmoid_function(numpy.dot(x_data, theta))

    ones = numpy.where(hypothesis == 1.0)
    hypothesis[ones] = hypothesis[ones] + 1e-9
    zeros = numpy.where(hypothesis == 0.0)
    hypothesis[zeros] = 1e-9

    term1 = y_data*(numpy.log(hypothesis))
    term2 = (1.0 - y_data)*(numpy.log(1.0 - hypothesis))
    # regularization, skip bias theta-zero
    regularization = (regularization_lambda/(2*num_examples))*numpy.sum(numpy.power(theta[1:], 2.0))
    cost_J = numpy.sum(-term1 - term2)/num_examples + regularization

    return cost_J
