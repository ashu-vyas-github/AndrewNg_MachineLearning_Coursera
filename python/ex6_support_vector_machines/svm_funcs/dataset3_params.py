import numpy

import svm_funcs


def dataset3_params(x_train, y_train, x_valid, y_valid):
    """
    Returns your choice of reg_C and sigma for Part 3 of the exercise
    where you select the optimal (reg_C, sigma) learning parameters to use for SVM
    with RBF kernel.

    Parameters
    ----------
    x_train : array_like
        (m x n) matrix of training data where m is number of training examples, and
        n is the number of features.

    y_train : array_like
        (m, ) vector of labels for ther training data.

    x_valid : array_like
        (mv x n) matrix of validation data where mv is the number of validation examples
        and n is the number of features

    y_valid : array_like
        (mv, ) vector of labels for the validation data.

    Returns
    -------
    reg_C, sigma : float, float
        The best performing values for the regularization parameter C and
        RBF parameter sigma.

    Instructions
    ------------
    Fill in this function to return the optimal C and sigma learning
    parameters found using the cross validation set.
    You can use `svmPredict` to predict the labels on the cross
    validation set. For example,

        predictions = svmPredict(model, x_valid)

    will return the predictions on the cross validation set.

    Note
    ----
    You can compute the prediction error using

        numpy.mean(predictions != y_valid)
    """
    reg_C = 0
    reg_C_array = numpy.linspace(0.01, 0.5, 11, True)
    # reg_C_array = numpy.concatenate((reg_C_array, 0.25*reg_C_array, 0.5*reg_C_array, 0.75*reg_C_array))
    sigma_array = numpy.linspace(-0.05, 0.1, 11, True)
    # sigma_array = numpy.concatenate((sigma_array, 0.25*sigma_array, 0.5*sigma_array, 0.75*sigma_array))
    num_C = reg_C_array.shape[0]
    num_sigma = sigma_array.shape[0]
    err_array = numpy.zeros((num_C, num_sigma))
    print("\n\nRunning {v1} models\n\n".format(v1=num_C*num_sigma))
    for i in numpy.arange(num_C):
        for j in numpy.arange(num_sigma):
            model = svm_funcs.svm_train(svm_funcs.gaussian_kernel, x_train, y_train,
                                        reg_C_array[i], tol=1e-3, max_passes=5, args=(sigma_array[j],))
            predictions = svm_funcs.svm_predict(model, x_valid)
            pred_error = numpy.mean(predictions != y_valid)
            err_array[i, j] = pred_error

    ind = numpy.unravel_index(numpy.argmin(err_array, axis=None), err_array.shape)
    reg_C = reg_C_array[ind[0]]
    sigma = sigma_array[ind[1]]

    return reg_C, sigma
