import numpy
import scipy.optimize as optimization


def train_lin_reg(cost_function, x_array, y_array, reg_lambda=0.0, maxiter=None):

    initial_theta = numpy.zeros((x_array.shape[1]))  # Initialize theta

    options = {'maxiter': maxiter}

    # Minimize using scipy
    results = optimization.minimize(cost_function, initial_theta, args=(
        x_array, y_array, reg_lambda), jac=True, method='TNC', options=options)

    theta = results.x
    convergence_flag = results.success

    return theta, convergence_flag
