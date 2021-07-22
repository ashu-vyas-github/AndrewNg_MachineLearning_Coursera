import numpy

import nnl_funcs


def sigmoid_gradient(x_var):

    sigmoid = nnl_funcs.sigmoid_function(x_var)
    sig_grad = sigmoid * (1 - sigmoid)

    return sig_grad
