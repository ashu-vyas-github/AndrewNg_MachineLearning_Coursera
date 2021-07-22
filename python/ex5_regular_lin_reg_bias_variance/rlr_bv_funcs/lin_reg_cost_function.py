import numpy


def lin_reg_cost_function(theta, x_array, y_array, reg_lambda=0.0):

    num_examples = x_array.shape[0]
    denr = 2*num_examples
    grad = numpy.zeros(theta.shape)

    hypothesis = numpy.dot(x_array, theta)
    error = hypothesis - y_array
    least_squares = (1/denr) * numpy.sum(numpy.square(error))
    regularization = (reg_lambda/denr) * numpy.sum(numpy.square(theta[1:]))
    cost_J = least_squares + regularization

    grad = (1/num_examples) * numpy.dot(error, x_array)
    grad[1:] = grad[1:] + (reg_lambda/num_examples) * theta[1:]

    return cost_J, grad
