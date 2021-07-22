import numpy


def compute_numerical_gradient(cost_func_J, theta, eps=1e-4):
    """
    Computes the gradient using "finite differences" and gives us a numerical estimate of the gradient.

    Parameters
    ----------
    cost_func_J : func
        The cost function which will be used to estimate its numerical gradient.

    theta : array_like
        The one dimensional unrolled network parameters. The numerical gradient is computed at those given parameters.

    eps : float (optional)
        The value to use for epsilon for computing the finite difference.

    Returns
    -------
    num_grad : array_like
        The numerical gradient with respect to theta. Has same shape as theta.

    Notes
    -----
    The following code implements numerical gradient checking, and returns the numerical gradient.
    It sets `numgrad[i]` to (a numerical approximation of) the partial derivative of J with respect to the
    i-th input argument, evaluated at theta. (i.e., `numgrad[i]` should be the (approximately)
    partial derivative of J with respect to theta[i].)
    """
    num_grad = numpy.zeros(theta.shape)
    perturb = numpy.diag(eps*numpy.ones(theta.shape))
    for i in range(theta.size):
        loss1, _ = cost_func_J(theta - perturb[:, i])
        loss2, _ = cost_func_J(theta + perturb[:, i])
        num_grad[i] = (loss2 - loss1)/(2*eps)

    return num_grad
