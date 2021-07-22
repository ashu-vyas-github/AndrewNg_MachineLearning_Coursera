import numpy

import nnl_funcs


def compute_numerical_gradient(cost_func_shorthand, theta, eps=1e-4):

    numgrad = numpy.zeros(theta.shape)
    perturb = numpy.diag(eps * numpy.ones(theta.shape))

    for i in range(theta.size):
        loss1, _ = cost_func_shorthand(theta - perturb[:, i])
        loss2, _ = cost_func_shorthand(theta + perturb[:, i])
        numgrad[i] = (loss2 - loss1)/(2*eps)

    return numgrad
