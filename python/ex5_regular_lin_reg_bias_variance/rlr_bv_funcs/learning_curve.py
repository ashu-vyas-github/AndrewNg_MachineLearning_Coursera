import numpy

import rlr_bv_funcs


def learning_curve(cost_function, x_train, y_train, x_valid, y_valid, reg_lambda=0):

    num_train = y_train.shape[0]
    # num_valid = y_valid.shape[0]

    error_train = numpy.zeros(num_train)
    error_valid = numpy.zeros(num_train)

    convergence = numpy.zeros(num_train)

    for idx in range(1, num_train + 1):
        theta_t, conv_t = rlr_bv_funcs.train_lin_reg(
            cost_function, x_train[:idx], y_train[:idx], reg_lambda=reg_lambda, maxiter=1000)
        convergence[idx - 1] = conv_t
        error_train[idx - 1], _ = rlr_bv_funcs.lin_reg_cost_function(
            theta_t, x_train[:idx], y_train[:idx], reg_lambda=reg_lambda)
        error_valid[idx - 1], _ = rlr_bv_funcs.lin_reg_cost_function(theta_t, x_valid, y_valid, reg_lambda=reg_lambda)

    return error_train, error_valid, convergence
