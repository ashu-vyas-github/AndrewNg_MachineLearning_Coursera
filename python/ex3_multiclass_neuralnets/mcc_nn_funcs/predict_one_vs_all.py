import numpy

import mcc_nn_funcs


def predict_one_vs_all(all_theta, x_data):

    num_classes = all_theta.shape[0]
    x_data = numpy.c_[numpy.ones((x_data.shape[0])), x_data]
    num_examples, num_features = x_data.shape
    predictions = numpy.zeros((num_examples))
    predictions = numpy.argmax(mcc_nn_funcs.sigmoid_function(x_data.dot(all_theta.T)), axis=1)

    return predictions
