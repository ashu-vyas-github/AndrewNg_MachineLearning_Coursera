import numpy

import matplotlib.pyplot as plt


def display_data(x_array, example_width=None, figsize=(10, 10)):
    """
    Displays 2D data in a nice grid.

    Parameters
    ----------
    x_array : array_like
        The input data of size (m x n) where m is the number of examples and n is the number of
        features.

    example_width : int, optional
        THe width of each 2-D image in pixels. If not provided, the image is assumed to be square,
        and the width is the floor of the square root of total number of pixels.

    figsize : tuple, optional
        A 2-element tuple indicating the width and height of figure in inches.
    """
    # Compute rows, cols
    if x_array.ndim == 2:
        num_examples, num_features = x_array.shape
    elif x_array.ndim == 1:
        num_features = x_array.size
        num_examples = 1
        x_array = x_array[None]  # Promote to a 2 dimensional array
    else:
        raise IndexError('Input x_array should be 1 or 2 dimensional.')

    example_width = example_width or int(numpy.round(numpy.sqrt(num_features)))
    example_height = int(num_features / example_width)

    # Compute number of items to display
    display_rows = int(numpy.floor(numpy.sqrt(num_examples)))
    display_cols = int(numpy.ceil(num_examples / display_rows))

    fig, ax_array = plt.subplots(display_rows, display_cols, figsize=figsize)
    fig.subplots_adjust(wspace=0.025, hspace=0.025)

    ax_array = [ax_array] if num_examples == 1 else ax_array.ravel()

    for i, ax in enumerate(ax_array):
        ax.imshow(x_array[i].reshape(example_height, example_width, order='F'), cmap='gray')
        ax.axis('off')
    fig.show()

    return 0
