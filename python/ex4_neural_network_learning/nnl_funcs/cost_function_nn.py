import numpy

import nnl_funcs


def cost_function_nn(theta_vector, x_data, y_data, input_layer_size, hidden_layer_size, num_labels, reg_lambda=0):

    num_examples = y_data.shape[0]
    x_data = numpy.c_[numpy.ones((num_examples)), x_data]

    theta1, theta2 = nnl_funcs.theta_vec2mat(theta_vector, input_layer_size, hidden_layer_size, num_labels)

    input_to_hidden_map = nnl_funcs.sigmoid_function(numpy.dot(x_data, numpy.transpose(theta1)))
    input_to_hidden_map = numpy.c_[numpy.ones((input_to_hidden_map.shape[0])), input_to_hidden_map]

    hidden_to_output_map = nnl_funcs.sigmoid_function(numpy.dot(input_to_hidden_map, numpy.transpose(theta2)))

    y_encoded = y_data.reshape(-1)
    y_encoded = numpy.eye(num_labels)[y_encoded]

    term1 = y_encoded*(numpy.log(hidden_to_output_map))
    term2 = (1.0 - y_encoded)*(numpy.log(1.0 - hidden_to_output_map))
    # regularization, skip bias theta-zero
    sum_sqd_theta1 = numpy.sum(numpy.power(theta1[:, 1:], 2.0))
    sum_sqd_theta2 = numpy.sum(numpy.power(theta2[:, 1:], 2.0))
    regularization = (reg_lambda/(2*num_examples))*(sum_sqd_theta1 + sum_sqd_theta2)
    cost_J = numpy.sum(-term1 - term2)/num_examples + regularization

    gradient = nnl_funcs.backpropagation_nn(theta1, theta2, x_data, y_encoded,
                                            input_to_hidden_map, hidden_to_output_map, reg_lambda=reg_lambda)

    return cost_J, gradient
