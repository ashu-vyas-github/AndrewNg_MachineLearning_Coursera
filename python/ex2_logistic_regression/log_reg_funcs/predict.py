import numpy

import log_reg_funcs


def predict(theta, x_data):
    """
    Predict whether a student will be admitted.
    Args:
        x: array shape(m, n+1)
        theta: ndarray, the optimal parameters of the cost function
    Returns:
        predicted: array shape(m,) of booleans
    """
    probability = log_reg_funcs.sigmoid_function(numpy.dot(x_data, theta))
    predicted = probability >= 0.5
    return predicted
