import numpy


def recover_data(projection_Z, mat_U, ndims_K):
    """
    Recovers an approximation of the original data when using the
    projected data.

    Parameters
    ----------
    projection_Z : array_like
        The reduced data after applying PCA. This is a matrix
        of shape (m x K).

    mat_U : array_like
        The eigenvectors (principal components) computed by PCA.
        This is a matrix of shape (n x n) where each column represents
        a single eigenvector.

    ndims_K : int
        The number of principal components retained
        (should be less than n).

    Returns
    -------
    recovered_X : array_like
        The recovered data after transformation back to the original
        dataset space. This is a matrix of shape (m x n), where m is
        the number of examples and n is the dimensions (number of
        features) of original datatset.

    Instructions
    ------------
    Compute the approximation of the data by projecting back
    onto the original space using the top K eigenvectors in mat_U.
    For the i-th example projection_Z[i,:], the (approximate)
    recovered data for dimension j is given as follows:

        v = projection_Z[i, :]
        recovered_j = np.dot(v, mat_U[j, :ndims_K])

    Notice that mat_U[j, :ndims_K] is a vector of size K.
    """
    recovered_X = numpy.dot(projection_Z, numpy.transpose(mat_U[:, :ndims_K]))

    return recovered_X
