import numpy

import ano_rec_funcs


def check_cost_function(cofi_cost_func, reg_lambda=0.0):
    """
    Creates a collaborative filtering problem to check your cost function and gradients.
    It will output the  analytical gradients produced by your code and the numerical gradients
    (computed using computeNumericalGradient). These two gradient computations should result
    in very similar values.

    Parameters
    ----------
    cofi_cost_func: func
        Implementation of the cost function.

    lambda_ : float, optional
        The regularization parameter.
    """
    # Create small problem
    X_t = numpy.random.rand(4, 3)
    Theta_t = numpy.random.rand(5, 3)

    # Zap out most entries
    Y = numpy.dot(X_t, numpy.transpose(Theta_t))
    Y[numpy.random.rand(*Y.shape) > 0.5] = 0
    R = numpy.zeros(Y.shape)
    R[Y != 0] = 1

    # Run Gradient Checking
    X = numpy.random.randn(*X_t.shape)
    Theta = numpy.random.randn(*Theta_t.shape)
    num_movies, num_users = Y.shape
    num_features = Theta_t.shape[1]

    params = numpy.concatenate([X.ravel(), Theta.ravel()])
    num_grad = ano_rec_funcs.compute_numerical_gradient(
        lambda x: cofi_cost_func(x, Y, R, num_users, num_movies, num_features, reg_lambda), params)

    cost, grad = cofi_cost_func(params, Y, R, num_users, num_movies, num_features, reg_lambda)

    print(numpy.stack([num_grad, grad], axis=1))
    print("\nThe above two columns you get should be very similar."
          "(Left-Your Numerical Gradient, Right-Analytical Gradient)")

    diff = numpy.linalg.norm(num_grad-grad)/numpy.linalg.norm(num_grad+grad)
    print("If your cost function implementation is correct, then "
          "the relative difference will be small (less than 1e-9).")
    print('\nRelative Difference: %g' % diff)

    return 0
