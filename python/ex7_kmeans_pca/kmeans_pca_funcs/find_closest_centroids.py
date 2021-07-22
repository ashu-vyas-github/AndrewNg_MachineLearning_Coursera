import numpy


def find_closest_centroids(x_array, centroids):
    """
    Computes the centroid memberships for every example.

    Parameters
    ----------
    x_array : array_like
        The dataset of size (m, n) where each row is a single example.
        That is, we have m examples each of n dimensions.

    centroids : array_like
        The k-means centroids of size (K, n). K is the number
        of clusters, and n is the the data dimension.

    Returns
    -------
    idx : array_like
        A vector of size (m, ) which holds the centroids assignment for each
        example (row) in the dataset x_array.

    Instructions
    ------------
    Go over every example, find its closest centroid, and store
    the index inside `idx` at the appropriate location.
    Concretely, idx[i] should contain the index of the centroid
    closest to example i. Hence, it should be a value in the
    range 0..K-1

    Note
    ----
    You can use a for-loop over the examples to compute this.
    """
    num_centroids_K = centroids.shape[0]
    num_examples, num_features = x_array.shape
    idx = numpy.zeros((num_examples), dtype=int)
    temp = numpy.zeros((num_centroids_K))

    for i in numpy.arange(num_examples):
        for j in range(num_centroids_K):
            difference = x_array[i, :] - centroids[j, :]
            distance = numpy.sum(difference**2)
            temp[j] = distance
        idx[i] = numpy.argmin(temp) + 1  # for indexing from 1 rather than 0

    return idx
