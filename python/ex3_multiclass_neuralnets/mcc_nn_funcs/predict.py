import numpy

import mcc_nn_funcs


def predict(theta1, theta2, x_data):

    x_data = numpy.c_[numpy.ones((x_data.shape[0])), x_data]
    num_examples, num_features = x_data.shape
    num_labels = theta2.shape[0]

    # input features to hidden layer sigmoid activation mapping by given weights theta1
    input_to_hidden_layer = mcc_nn_funcs.sigmoid_function(numpy.dot(x_data, numpy.transpose(theta1)))

    # add bias to hidden layer outputs
    input_to_hidden_layer = numpy.c_[numpy.ones((input_to_hidden_layer.shape[0])), input_to_hidden_layer]

    # hidden layer to output sigmoid activation mapping by given weights theta2
    hidden_to_output = mcc_nn_funcs.sigmoid_function(numpy.dot(input_to_hidden_layer, numpy.transpose(theta2)))

    # generate predictions
    predictions = numpy.argmax(hidden_to_output, axis=1)

    return predictions
