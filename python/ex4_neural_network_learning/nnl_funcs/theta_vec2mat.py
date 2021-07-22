import numpy


def theta_vec2mat(theta_vector, input_layer_size, hidden_layer_size, num_labels):

    theta1 = numpy.reshape(theta_vector[:hidden_layer_size*(input_layer_size + 1)],
                           (hidden_layer_size, (input_layer_size + 1)))

    theta2 = numpy.reshape(theta_vector[(hidden_layer_size*(input_layer_size + 1)):],
                           (num_labels, (hidden_layer_size + 1)))

    return theta1, theta2
