import numpy


def gaussian_kernel(x1, x2, sigma):
    """
    Computes the radial basis function
    Returns a radial basis function kernel between x1 and x2.

    Parameters
    ----------
    x1 :  numpy ndarray
        A vector of size (n, ), representing the first datapoint.

    x2 : numpy ndarray
        A vector of size (n, ), representing the second datapoint.

    sigma : float
        The standard deviation (bandwidth) parameter for the Gaussian kernel.

    Returns
    -------
    rbf : float
        The computed RBF between the two provided data points.

    Instructions
    ------------
    Fill in this function to return the similarity between `x1` and `x2`
    computed using a Gaussian kernel with bandwidth `sigma`.
    """
    rbf = 0
    numr = -numpy.sum((x1 - x2)**2)
    denr = 2*(sigma**2)
    rbf = numpy.exp(numr/denr)

    return rbf
