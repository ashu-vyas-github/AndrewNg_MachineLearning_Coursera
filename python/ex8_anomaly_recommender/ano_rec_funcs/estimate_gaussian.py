import numpy


def estimate_gaussian(x_array):
    """
    This function estimates the parameters of a Gaussian distribution
    using a provided dataset.

    Parameters
    ----------
    x_array : array_like
        The dataset of shape (m x n) with each n-dimensional
        data point in one row, and each total of m data points.

    Returns
    -------
    mean_mu : array_like
        A vector of shape (n,) containing the means of each dimension.

    sigma2 : array_like
        A vector of shape (n,) containing the computed
        variances of each dimension.

    Instructions
    ------------
    Compute the mean of the data and the variances
    In particular, mean_mu[i] should contain the mean of
    the data for the i-th feature and sigma2[i]
    should contain variance of the i-th feature.
    """
    num_examples, num_features = x_array.shape
    mean_mu = numpy.zeros(num_features)
    sigma2 = numpy.zeros(num_features)

    mean_mu = (1/num_examples)*numpy.sum(x_array, axis=0)
    sigma2 = (1/num_examples)*numpy.sum((x_array - mean_mu)**2, axis=0)

    return mean_mu, sigma2
