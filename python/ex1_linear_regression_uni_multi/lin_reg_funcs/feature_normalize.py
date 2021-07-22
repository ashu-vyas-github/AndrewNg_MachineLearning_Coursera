import numpy


def feature_normalize(x_data, y_data, num_examples):

    x_mean = numpy.mean(x_data, axis=0)
    x_std = numpy.std(x_data, axis=0)
    x_norm = (x_data - x_mean)/x_std

    y_mean = numpy.mean(y_data)
    y_std = numpy.std(y_data)
    y_norm = (y_data - y_mean)/y_std

    return x_norm, y_norm, x_mean, y_mean, x_std, y_std
