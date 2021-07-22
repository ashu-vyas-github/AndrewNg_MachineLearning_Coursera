import numpy


def project_data(x_array, mat_U, ndims_K):
    """
    Computes the reduced data representation when projecting only
    on to the top K eigenvectors.

    Parameters
    ----------
    x_array : array_like
        The input dataset of shape (m x n). The dataset is assumed to be
        normalized.

    mat_U : array_like
        The computed eigenvectors using PCA. This is a matrix of
        shape (n x n). Each column in the matrix represents a single
        eigenvector (or a single principal component).

    ndims_K : int
        Number of dimensions to project onto. Must be smaller than n.

    Returns
    -------
    projection_Z : array_like
        The projects of the dataset onto the top K eigenvectors.
        This will be a matrix of shape (m x k).

    Instructions
    ------------
    Compute the projection of the data using only the top K
    eigenvectors in mat_U (first K columns).
    For the i-th example x_array[i,:], the projection on to the k-th
    eigenvector is given as follows:

        x_array = x_array[i, :]
        projection_k = np.dot(x_array,  mat_U[:, k])

    """
    projection_Z = numpy.dot(x_array,  mat_U[:, :ndims_K])

    return projection_Z
