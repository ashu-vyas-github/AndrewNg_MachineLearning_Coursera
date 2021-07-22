import numpy
import matplotlib
import matplotlib.pyplot as plt

import svm_funcs

matplotlib.rcParams.update({'font.size': 8})


def plot_data(x_array, y_array, model=None, reg_C=0, linear_boundary=False, nonlinear_boundary=False):

    # Find indices of positive and negative examples
    positives = numpy.where(y_array == 1)
    negatives = numpy.where(y_array == 0)

    # Plot Examples
    plt.figure(dpi=120)
    plt.title("Ex.6 Training data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    # plt.xlim(0, 5)
    # plt.ylim(0, 5)
    plt.scatter(x_array[positives, 0], x_array[positives, 1], marker='+', s=26,
                linewidths=1.0, color='red', label="Positives - label 1")
    plt.scatter(x_array[negatives, 0], x_array[negatives, 1], marker='o', s=12,
                linewidths=0.75, color='blue', label="Negatives - label 0")

    if linear_boundary is True:
        svm_funcs.visualize_boundary_linear(x_array, y_array, model, reg_C=reg_C)
    else:
        print("No linear boundary plotted")

    if nonlinear_boundary is True:
        svm_funcs.visualize_boundary(x_array, y_array, model)
    else:
        print("No nonlinear boundary plotted")

    plt.grid(b=True, which='major', axis='both', linestyle='--', linewidth=0.5)
    plt.legend(bbox_to_anchor=(1.025, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()
    # pyplot.plot(X[pos, 0], X[pos, 1], 'X', mew=1, ms=10, mec='k')
    # pyplot.plot(X[neg, 0], X[neg, 1], 'o', mew=1, mfc='y', ms=10, mec='k')

    return 0
