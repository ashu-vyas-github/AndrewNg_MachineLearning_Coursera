import numpy

import svm_funcs


def svm_train(kernel_function, x_array, y_array, reg_C, tol=1e-3, max_passes=5, args=()):

    b = 0
    passes = 0
    y_array = y_array.astype(int)  # make sure labels are signed int
    y_array[y_array == 0] = -1  # Map 0 to -1
    num_examples, num_features = x_array.shape
    E = numpy.zeros(num_examples)
    alphas = numpy.zeros(num_examples)

    if kernel_function.__name__ == 'linear_kernel':
        K = numpy.dot(x_array, numpy.transpose(x_array))

    elif kernel_function.__name__ == 'gaussian_kernel':
        x_array_sqd = numpy.sum(x_array**2, axis=1)
        K = x_array_sqd + x_array_sqd[:, None] - 2*numpy.dot(x_array, numpy.transpose(x_array))

        if len(args) > 0:
            K /= 2*args[0]**2

        K = numpy.exp(-K)

    else:
        K = numpy.zeros((num_examples, num_examples))
        for i in range(num_examples):
            for j in range(i, num_examples):
                K[i, j] = kernel_function(x_array[i, :], x_array[j, :])
                K[j, i] = K[i, j]

    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(num_examples):
            E[i] = b + numpy.sum(alphas*y_array*K[:, i]) - y_array[i]

            if (y_array[i]*E[i] < -tol and alphas[i] < reg_C) or (y_array[i]*E[i] > tol and alphas[i] > 0):
                # select the alpha_j randomly
                j = numpy.random.choice(list(range(i)) + list(range(i+1, num_examples)), size=1)[0]
                E[j] = b + numpy.sum(alphas*y_array*K[:, j]) - y_array[j]
                alpha_i_old = alphas[i]
                alpha_j_old = alphas[j]

                if y_array[i] == y_array[j]:
                    L = max(0, alphas[j] + alphas[i] - reg_C)
                    H = min(reg_C, alphas[j] + alphas[i])
                else:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(reg_C, reg_C + alphas[j] - alphas[i])

                if L == H:
                    continue

                eta = 2*K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                alphas[j] = alphas[j] - y_array[j] * (E[i] - E[j])/eta
                alphas[j] = max(L, min(H, alphas[j]))

                if abs(alphas[j] - alpha_j_old) < tol:
                    alphas[j] = alpha_j_old
                    continue

                alphas[i] = alphas[i] + y_array[i]*y_array[j]*(alpha_j_old - alphas[j])
                b1 = b - E[i] - y_array[i]*(alphas[i] - alpha_i_old) * K[i, j] \
                     - y_array[j] * (alphas[j] - alpha_j_old) * K[i, j]
                b2 = b - E[j] - y_array[i]*(alphas[i] - alpha_i_old) * K[i, j] \
                     - y_array[j] * (alphas[j] - alpha_j_old) * K[j, j]

                if 0 < alphas[i] < reg_C:
                    b = b1
                elif 0 < alphas[j] < reg_C:
                    b = b2
                else:
                    b = (b1 + b2)/2

                num_changed_alphas = num_changed_alphas + 1

        if num_changed_alphas == 0:
            passes = passes + 1
        else:
            passes = 0

    idx = alphas > 0
    model = {'X': x_array[idx, :],
             'y': y_array[idx],
             'kernel_function': kernel_function,
             'b': b,
             'args': args,
             'alphas': alphas[idx],
             'w': numpy.dot(alphas*y_array, x_array)}

    return model
