import numpy


def cofi_cost_func(params, Y, R, num_users, num_movies, num_features, reg_lambda=0.0):
    """
    Collaborative filtering cost function.

    Parameters
    ----------
    params : array_like
        The parameters which will be optimized. This is a one
        dimensional vector of shape (num_movies x num_users, 1). It is the
        concatenation of the feature vectors X and parameters Theta.

    Y : array_like
        A matrix of shape (num_movies x num_users) of user ratings of movies.

    R : array_like
        A (num_movies x num_users) matrix, where R[i, j] = 1 if the
        i-th movie was rated by the j-th user.

    num_users : int
        Total number of users.

    num_movies : int
        Total number of movies.

    num_features : int
        Number of features to learn.

    reg_lambda : float, optional
        The regularization coefficient.

    Returns
    -------
    cost_J : float
        The value of the cost function at the given params.

    grad : array_like
        The gradient vector of the cost function at the given params.
        grad has a shape (num_movies x num_users, 1)

    Instructions
    ------------
    Compute the cost function and gradient for collaborative filtering.
    Concretely, you should first implement the cost function (without
    regularization) and make sure it is matches our costs. After that,
    you should implement thegradient and use the checkCostFunction routine
    to check that the gradient is correct. Finally, you should implement
    regularization.

    Notes
    -----
    - The input params will be unraveled into the two matrices:
        x_array : (num_movies  x num_features) matrix of movie features
        Theta : (num_users  x num_features) matrix of user features

    - You should set the following variables correctly:

        x_grad : (num_movies x num_features) matrix, containing the
                 partial derivatives w.r.t. to each element of x_array
        Theta_grad : (num_users x num_features) matrix, containing the
                     partial derivatives w.r.t. to each element of Theta

    - The returned gradient will be the concatenation of the raveled
      gradients x_grad and Theta_grad.
    """
    # Unfold the U and W matrices from params
    cost_J = 0
    prod_mov_feat = num_movies*num_features
    x_array = params[:prod_mov_feat].reshape(num_movies, num_features)
    x_grad = numpy.zeros(x_array.shape)
    Theta = params[prod_mov_feat:].reshape(num_users, num_features)
    Theta_grad = numpy.zeros(Theta.shape)

    hypothesis = numpy.dot(x_array, numpy.transpose(Theta))
    error = (1/2)*numpy.sum(numpy.square((hypothesis - Y) * R))
    reg_term = (reg_lambda/2)*(numpy.sum(numpy.square(x_array)) + numpy.sum(numpy.square(Theta)))
    cost_J = error + reg_term

    for i in range(R.shape[0]):
        idx = numpy.where(R[i, :] == 1)[0]
        Theta_temp = Theta[idx, :]
        Y_temp = Y[i, idx]
        term1 = numpy.dot(numpy.dot(x_array[i, :], Theta_temp.T) - Y_temp, Theta_temp)
        term2 = reg_lambda*x_array[i, :]
        x_grad[i, :] = term1 + term2

    for j in range(R.shape[1]):
        idx = numpy.where(R[:, j] == 1)[0]
        X_temp = x_array[idx, :]
        Y_temp = Y[idx, j]
        term1 = numpy.dot(numpy.dot(X_temp, Theta[j, :]) - Y_temp, X_temp)
        term2 = reg_lambda * Theta[j, :]
        Theta_grad[j, :] = term1 + term2

    grad = numpy.concatenate([x_grad.ravel(), Theta_grad.ravel()])

    return cost_J, grad
