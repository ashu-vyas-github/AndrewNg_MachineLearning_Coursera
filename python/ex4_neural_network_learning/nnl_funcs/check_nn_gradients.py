import numpy

import nnl_funcs


def check_nn_gradients(cost_function_nn, reg_lambda=0.0):

    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    num_examples = 5

    # We generate some 'random' test data
    theta1 = nnl_funcs.debug_initialize_weights(hidden_layer_size, input_layer_size)
    theta2 = nnl_funcs.debug_initialize_weights(num_labels, hidden_layer_size)

    # Reusing debugInitializeWeights to generate X
    x_data = nnl_funcs.debug_initialize_weights(num_examples, input_layer_size - 1)
    y_data = numpy.arange(1, 1+num_examples) % num_labels

    # Unroll parameters
    nn_params = numpy.concatenate([theta1.ravel(), theta2.ravel()])

    def cost_func_shorthand(nn_params):
        return nnl_funcs.cost_function_nn(nn_params, x_data, y_data, input_layer_size, hidden_layer_size, num_labels, reg_lambda)

    cost, grad = cost_func_shorthand(nn_params)
    numgrad = nnl_funcs.compute_numerical_gradient(cost_func_shorthand, nn_params)

    # Visually examine the two gradient computations.The two columns you get should be very similar.
    print(numpy.stack([numgrad, grad], axis=1))
    print("The above two columns you get should be very similar.")
    print("(Left-Your Numerical Gradient, Right-Analytical Gradient)\n")

    # Evaluate the norm of the difference between two the solutions. If you have a correct
    # implementation, and assuming you used e = 0.0001 in computeNumericalGradient, then diff
    # should be less than 1e-9.
    diff = numpy.linalg.norm(numgrad - grad)/numpy.linalg.norm(numgrad + grad)

    print('If your backpropagation implementation is correct, then \n'
          'the relative difference will be small (less than 1e-9). \n'
          'Relative Difference: %g' % diff)

    return 0
