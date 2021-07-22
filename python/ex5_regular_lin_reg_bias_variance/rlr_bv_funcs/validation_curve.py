import numpy

import rlr_bv_funcs


def validation_curve(cost_func, x_train, y_train, x_valid, y_valid):

    # Selected values of lambda (you should not change this)
    lambda_vec = numpy.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10])

    # You need to return these variables correctly.
    error_train = numpy.zeros((lambda_vec.shape[0]))
    error_valid = numpy.zeros((lambda_vec.shape[0]))

    convergence = numpy.zeros((lambda_vec.shape[0]))

    for idx in range(lambda_vec.shape[0]):
        lambda_try = lambda_vec[idx]
        theta_t, conv_t = rlr_bv_funcs.train_lin_reg(cost_func, x_train, y_train, reg_lambda=lambda_try)
        error_train[idx], _ = rlr_bv_funcs.lin_reg_cost_function(theta_t, x_train, y_train, reg_lambda=lambda_try)
        error_valid[idx], _ = rlr_bv_funcs.lin_reg_cost_function(theta_t, x_valid, y_valid, reg_lambda=lambda_try)
        convergence[idx] = conv_t

    return lambda_vec, error_train, error_valid, convergence
