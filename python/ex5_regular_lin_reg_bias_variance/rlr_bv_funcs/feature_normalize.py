import numpy


def feature_normalize(x_data, num_examples):

    x_mean = numpy.mean(x_data, axis=0)
    x_std = numpy.std(x_data, axis=0, ddof=1)
    x_norm = (x_data - x_mean)/x_std

    return x_norm, x_mean, x_std
