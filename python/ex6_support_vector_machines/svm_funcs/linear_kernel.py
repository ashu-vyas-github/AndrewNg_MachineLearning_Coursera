import numpy


def linear_kernel(x1, x2):
    """
    Returns a linear kernel between x1 and x2.

    Parameters
    ----------
    x1 : numpy ndarray
        A 1-D vector.

    x2 : numpy ndarray
        A 1-D vector of same size as x1.

    Returns
    -------
    dot_prod: float
        The scalar amplitude.
    """
    dot_prod = numpy.dot(x1, x2)

    return dot_prod
