import numpy
import matplotlib.pyplot as plt

import rlr_bv_funcs


def plot_fit(poly_features_func, min_x, max_x, mu, sigma, theta, poly_deg=8):

    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x_axis = numpy.arange(min_x - 15, max_x + 25, 0.05).reshape(-1)

    # Map the x_train values
    x_poly = poly_features_func(x_axis, poly_deg)
    x_poly = (x_poly - mu)/sigma

    # Add ones
    x_poly = rlr_bv_funcs.add_bias_vector(x_poly)

    # Plot
    plt.plot(x_axis, numpy.dot(x_poly, theta), '--', linewidth=1.0, label="Polynomial fit")

    return plt
