import numpy

import nnl_funcs


def backpropagation_nn(theta1, theta2, x_data, y_encoded, input_to_hidden_map, hidden_to_output_map, reg_lambda=1.0):

    num_examples, num_features = x_data.shape
    theta1_grad = numpy.zeros(theta1.shape)
    theta2_grad = numpy.zeros(theta2.shape)

    small_delta_out = hidden_to_output_map - y_encoded

    sdht1 = numpy.dot(small_delta_out, theta2)[:, 1:]
    sdht2 = nnl_funcs.sigmoid_gradient(numpy.dot(x_data, numpy.transpose(theta1)))
    small_delta_hidden = sdht1 * sdht2

    cap_delta_input = numpy.dot(numpy.transpose(small_delta_hidden), x_data)
    cap_delta_hidden = numpy.dot(numpy.transpose(small_delta_out), input_to_hidden_map)

    # Add regularization to gradient
    theta1_grad = (1/num_examples)*cap_delta_input
    theta1_grad[:, 1:] = theta1_grad[:, 1:] + (reg_lambda/num_examples)*theta1[:, 1:]

    theta2_grad = (1/num_examples)*cap_delta_hidden
    theta2_grad[:, 1:] = theta2_grad[:, 1:] + (reg_lambda/num_examples)*theta2[:, 1:]

    gradient = numpy.concatenate([theta1_grad.ravel(), theta2_grad.ravel()])

    return gradient
