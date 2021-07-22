import numpy


def multivariate_gaussian(x_array, mean_mu, corr_sigma):
    """
    Computes the probability density function of the multivariate gaussian distribution.

    Parameters
    ----------
    x_array : array_like
        The dataset of shape (m x n). Where there are m examples of n-dimensions.

    mean_mu : array_like
        A vector of shape (n,) contains the means for each dimension (feature).

    corr_sigma : array_like
        Either a vector of shape (n,) containing the variances of independent features
        (i.e. it is the diagonal of the correlation matrix), or the full
        correlation matrix of shape (n x n) which can represent dependent features.

    Returns
    ------
    p : array_like
        A vector of shape (m,) which contains the computed probabilities at each of the
        provided examples.
    """
    k = mean_mu.size

    # if sigma is given as a diagonal, compute the matrix
    if corr_sigma.ndim == 1:
        corr_sigma = numpy.diag(corr_sigma)

    x_array = x_array - mean_mu
    factor = (2*numpy.pi)**(-k/2)*numpy.linalg.det(corr_sigma)**(-0.5)
    exp_part = numpy.exp(-0.5*numpy.sum(numpy.dot(x_array, numpy.linalg.pinv(corr_sigma))*x_array, axis=1))

    probabilities = factor*exp_part

    return probabilities
