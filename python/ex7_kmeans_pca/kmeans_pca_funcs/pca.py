import numpy


def pca(x_array):
    """
    Run principal component analysis.

    Parameters
    ----------
    x_array : array_like
        The dataset to be used for computing PCA. It has dimensions (m x n)
        where m is the number of examples (observations) and n is
        the number of features.

    Returns
    -------
    U : array_like
        The eigenvectors, representing the computed principal components
        of x_array. U has dimensions (n x n) where each column is a single
        principal component.

    S : array_like
        A vector of size n, contaning the singular values for each
        principal component. Note this is the diagonal of the matrix we
        mentioned in class.

    Instructions
    ------------
    You should first compute the covariance matrix. Then, you
    should use the "svd" function to compute the eigenvectors
    and eigenvalues of the covariance matrix.

    Notes
    -----
    When computing the covariance matrix, remember to divide by m (the
    number of examples).
    """
    num_examples, num_features = x_array.shape

    mat_U = numpy.zeros(num_features)
    mat_S = numpy.zeros(num_features)

    dot_prod = numpy.dot(numpy.transpose(x_array), x_array)
    covariance_Sigma = (1/num_examples)*(dot_prod)
    mat_U, mat_S, mat_V = numpy.linalg.svd(covariance_Sigma)

    return mat_U, mat_S
