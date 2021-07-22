import numpy

import log_reg_funcs


def cost_function(theta, x_data, y_data):

    num_examples = y_data.shape[0]
    hypothesis = log_reg_funcs.sigmoid_function(numpy.dot(x_data, theta))

    term1 = y_data*(numpy.log(hypothesis))
    term2 = (1.0 - y_data)*(numpy.log(1.0 - hypothesis))
    cost_J = numpy.sum(-term1 - term2)/num_examples

    return cost_J
