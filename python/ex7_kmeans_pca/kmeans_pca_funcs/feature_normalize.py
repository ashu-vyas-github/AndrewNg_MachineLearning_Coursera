import numpy


def feature_normalize(x_array):
    """
    Normalizes the features in x_array returns a normalized version of x_array where the mean value of each
    feature is 0 and the standard deviation is 1. This is often a good preprocessing step to do when
    working with learning algorithms.

    Parameters
    ----------
    x_array : array_like
        An dataset which is a (m x n) matrix, where m is the number of examples,
        and n is the number of dimensions for each example.

    Returns
    -------
    x_norm : array_like
        The normalized input dataset.

    mu : array_like
        A vector of size n corresponding to the mean for each dimension across all examples.

    sigma : array_like
        A vector of size n corresponding to the standard deviations for each dimension across
        all examples.
    """
    mu = numpy.mean(x_array, axis=0)
    sigma = numpy.std(x_array, axis=0, ddof=1)

    x_norm = (x_array - mu)/sigma

    return x_norm, mu, sigma
