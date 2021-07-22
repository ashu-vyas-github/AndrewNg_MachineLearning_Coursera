import numpy
import scipy.optimize as optimization

import mcc_nn_funcs


def one_vs_all(x_data, y_data, reg_lambda=1.0):

    y_data = y_data.astype("int")
    labels = numpy.unique(y_data)
    x_data = numpy.c_[numpy.ones((y_data.shape[0])), x_data]
    num_examples, num_features = x_data.shape

    all_theta = numpy.zeros((labels.shape[0], num_features))

    cost_values = numpy.zeros(labels.shape)
    convergence_flag = numpy.zeros(labels.shape)
    options = {'maxiter': 1000}  # , 'gtol': 1e-3, 'norm': numpy.inf}

    for onelabel in labels:

        binary_label = numpy.zeros(y_data.shape, dtype=int)
        onelabel_idx = numpy.where(y_data == onelabel)
        binary_label[onelabel_idx] = 1

        results = optimization.minimize(fun=mcc_nn_funcs.cost_function_regularized, x0=all_theta[onelabel, :],
                                        method="CG", args=(x_data, binary_label, reg_lambda),
                                        jac=True, options=options)

        all_theta[onelabel, :] = results.x
        cost_values[onelabel] = results.fun
        convergence_flag[onelabel] = results.success

    return all_theta, cost_values, convergence_flag

# mcc_nn_funcs.gradient_function_regularized
