import numpy

import svm_funcs


def svm_predict(model, x_array):

    # check if we are getting a vector. If so, then assume we only need to do predictions
    # for a single example
    if x_array.ndim == 1:
        x_array = x_array[numpy.newaxis, :]

    num_examples = x_array.shape[0]
    p = numpy.zeros(num_examples)
    pred = numpy.zeros(num_examples)

    if model['kernel_function'].__name__ == 'linear_kernel':
        # we can use the weights and bias directly if working with the linear kernel
        p = numpy.dot(x_array, model['w']) + model['b']
    elif model['kernel_function'].__name__ == 'gaussian_kernel':
        # vectorized RBF Kernel
        # This is equivalent to computing the kernel on every pair of examples
        x_array1 = numpy.sum(x_array**2, 1)
        x_array2 = numpy.sum(model['X']**2, 1)
        K = x_array2 + x_array1[:, None] - 2 * numpy.dot(x_array, model['X'].T)

        if len(model['args']) > 0:
            K /= 2*model['args'][0]**2

        K = numpy.exp(-K)
        p = numpy.dot(K, model['alphas']*model['y']) + model['b']
    else:
        # other non-linear kernel
        for i in range(num_examples):
            predictions = 0
            for j in range(model['X'].shape[0]):
                predictions += model['alphas'][j] * model['y'][j] \
                    * model['kernel_function'](x_array[i, :], model['X'][j, :])
            p[i] = predictions

    pred[p >= 0] = 1

    return pred
