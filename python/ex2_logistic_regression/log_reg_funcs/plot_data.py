# Reference 1: https://scikit-learn.org/stable/auto_examples/ensemble/plot_voting_decision_regions.html
# Reference 2: https://gtraskas.github.io/post/ex2/
# Reference 3: https://medium.com/analytics-vidhya/python-implementation-of-andrew-ngs-machine-learning-course-part-2-2-dceff1a12a12

import log_reg_funcs
import numpy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 8})


def plot_data(x_data, y_data, xaxis_text="Test 1", yaxis_text="Test 2", xaxis_lim=(0, 120), yaxis_lim=(0, 120)):

    idx_0 = numpy.where(y_data == 0)
    idx_1 = numpy.where(y_data == 1)
    plt.figure(dpi=120)
    plt.xlabel(xaxis_text)
    plt.ylabel(yaxis_text)
    plt.xlim(xaxis_lim)
    plt.ylim(yaxis_lim)
    plt.title("Training data")
    plt.scatter(x_data[idx_0, 0], x_data[idx_0, 1], s=12, marker='o', color='red', linewidths=0.75, label="QC Failed")
    plt.scatter(x_data[idx_1, 0], x_data[idx_1, 1], s=12, marker='+', color='blue', linewidths=0.75, label="QC Passed")
    # plt.plot(x_data, line_data, label="Linear Regression", color='blue', linewidth=1.0)
    plt.grid(b=True, which='major', axis='both', linestyle='--', linewidth=0.5)
    plt.legend(bbox_to_anchor=(1.025, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()
    return 0


def plot_decision_boundary_line(theta, x_data, y_data):

    # Only 2 points are required to define a line, e.g. min and max.
    idx_0 = numpy.where(y_data == 0)
    idx_1 = numpy.where(y_data == 1)
    plot_x = numpy.array([numpy.min(x_data[:, 1]) - 2, numpy.max(x_data[:, 1] + 2)])
    plot_y = -(theta[0] + theta[1] * plot_x) / theta[2]

    plt.figure(dpi=120)
    plt.xlabel('Exam 1 scores')
    plt.ylabel('Exam 2 scores')
    plt.xlim(0, 120)
    plt.ylim(0, 120)
    plt.title('Training data and linear decision boundary')
    plt.scatter(x_data[idx_0, 1], x_data[idx_0, 2], s=12, marker='o', color='red', linewidths=0.75, label="QC Failed")
    plt.scatter(x_data[idx_1, 1], x_data[idx_1, 2], s=12, marker='+', color='blue', linewidths=0.75, label="QC Passed")
    plt.plot(plot_x, plot_y, linestyle='-', color='black', linewidth=1.0, label='Decision Boundary')
    plt.plot(45, 85, 'kx', ms=6)
    plt.annotate('(45, 85)', xy=(45, 85), xytext=(47, 84))
    plt.grid(b=True, which='major', axis='both', linestyle='--', linewidth=0.5)
    plt.legend(bbox_to_anchor=(1.025, 1.0), loc='upper left')
    plt.tight_layout()
    plt.show()

    return 0


def plot_decision_boundary(theta, x_data, y_data):

    if x_data.shape[1] <= 3:
        plot_decision_boundary_line(theta, x_data, y_data)
    else:
        u = numpy.linspace(-1.0, 1.5, 50)
        v = numpy.linspace(-1.0, 1.5, 50)
        z = numpy.zeros((len(u), len(v)))
        for i in range(len(u)):
            for j in range(len(v)):
                z[i, j] = numpy.dot(log_reg_funcs.map_feature(u[i], v[j], 1), theta)

        idx_0 = numpy.where(y_data == 0)
        idx_1 = numpy.where(y_data == 1)
        plt.figure(dpi=120)
        plt.xlabel("Microchip test 1")
        plt.ylabel("Microchip test 2")
        plt.xlim(-1.5, 1.5)
        plt.ylim(-1.5, 1.5)
        plt.title('Training data and nonlinear decision boundary')
        plt.scatter(x_data[idx_0, 1], x_data[idx_0, 2], s=12, marker='o',
                    color='red', linewidths=0.75, label="QC Failed")
        plt.scatter(x_data[idx_1, 1], x_data[idx_1, 2], s=12, marker='+',
                    color='blue', linewidths=0.75, label="QC Passed")
        print(numpy.sum(z))

        plt.contour(u, v, z, 0)

        plt.grid(b=True, which='major', axis='both', linestyle='--', linewidth=0.5)
        plt.legend(bbox_to_anchor=(1.025, 1.0), loc='upper left')
        plt.tight_layout()
        plt.show()

    return 0
