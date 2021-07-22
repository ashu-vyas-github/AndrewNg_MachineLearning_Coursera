import numpy

import matplotlib.pyplot as plt


def visualize_boundary_linear(x_array, y_array, model, reg_C=0):
    """
    Plots a linear decision boundary learned by the SVM.

    Parameters
    ----------
    x_array : array_like
        (m x 2) The training data with two features (to plot in a 2-D plane).

    y_array : array_like
        (m, ) The data labels.

    model : dict
        Dictionary of model variables learned by SVM.
    """
    w = model['w']
    b = model['b']
    xp = numpy.linspace(min(x_array[:, 0]), max(x_array[:, 0]), 100)
    yp = -(w[0] * xp + b)/w[1]

    plt.plot(xp, yp, linestyle=':', color='black', linewidth=1.0, label="Linear decision C=%.f" % reg_C)

    return plt
