import numpy

import matplotlib.pyplot as plt

import ano_rec_funcs


def visualize_fit(x_array, mean_mu, corr_sigma):
    """
    Visualize the dataset and its estimated distribution.
    This visualization shows you the  probability density function of the Gaussian distribution.
    Each example has a location (x1, x2) that depends on its feature values.

    Parameters
    ----------
    x_array : array_like
        The dataset of shape (m x 2). Where there are m examples of 2-dimensions. We need at most
        2-D features to be able to visualize the distribution.

    mean_mu : array_like
        A vector of shape (n,) contains the means for each dimension (feature).

    corr_sigma : array_like
        Either a vector of shape (n,) containing the variances of independent features
        (i.e. it is the diagonal of the correlation matrix), or the full
        correlation matrix of shape (n x n) which can represent dependent features.
    """

    X1, X2 = numpy.meshgrid(numpy.arange(0, 35.5, 0.5), numpy.arange(0, 35.5, 0.5))
    Z = ano_rec_funcs.multivariate_gaussian(numpy.stack([X1.ravel(), X2.ravel()], axis=1), mean_mu, corr_sigma)
    Z = Z.reshape(X1.shape)

    plt.plot(x_array[:, 0], x_array[:, 1], 'bx', mec='b', mew=2, ms=8)

    if numpy.all(abs(Z) != numpy.inf):
        plt.contour(X1, X2, Z, levels=10**(numpy.arange(-20., 1, 3)), zorder=100)

    return plt
