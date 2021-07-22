import numpy

import matplotlib.pyplot as plt

import svm_funcs


def visualize_boundary(x_array, y_array, model):
    """
    Plots a non-linear decision boundary learned by the SVM and overlays the data on it.

    Parameters
    ----------
    x_array : array_like
        (m x 2) The training data with two features (to plot in a 2-D plane).

    y_array : array_like
        (m, ) The data labels.

    model : dict
        Dictionary of model variables learned by SVM.
    """

    # make classification predictions over a grid of values
    x1plot = numpy.linspace(min(x_array[:, 0]), max(x_array[:, 0]), 100)
    x2plot = numpy.linspace(min(x_array[:, 1]), max(x_array[:, 1]), 100)
    X1, X2 = numpy.meshgrid(x1plot, x2plot)
    vals = numpy.zeros(X1.shape)

    for i in range(X1.shape[1]):
        this_X = numpy.stack((X1[:, i], X2[:, i]), axis=1)
        vals[:, i] = svm_funcs.svm_predict(model, this_X)

    plt.contour(X1, X2, vals, colors='y', linewidths=1.0)
    plt.pcolormesh(X1, X2, vals, cmap='YlGnBu', alpha=0.25, edgecolors='None', linewidth=0, shading='auto')

    return plt
