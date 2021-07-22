import numpy


def compute_centroids(x_array, idx, num_centroids_K):
    """
    Returns the new centroids by computing the means of the data points
    assigned to each centroid.

    Parameters
    ----------
    x_array : array_like
        The datset where each row is a single data point. That is, it
        is a matrix of size (m, n) where there are m datapoints each
        having n dimensions.

    idx : array_like
        A vector (size m) of centroid assignments (i.e. each entry in range [0 ... K-1])
        for each example.

    num_centroids_K : int
        Number of clusters

    Returns
    -------
    centroids : array_like
        A matrix of size (K, n) where each row is the mean of the data
        points assigned to it.

    Instructions
    ------------
    Go over every centroid and compute mean of all points that
    belong to it. Concretely, the row vector centroids[i, :]
    should contain the mean of the data points assigned to
    cluster i.

    Note:
    -----
    You can use a for-loop over the centroids to compute this.
    """
    num_examples, num_features = x_array.shape
    centroids = numpy.zeros((num_centroids_K, num_features))
    count = numpy.zeros((num_centroids_K, 1))

    reindexing = idx - 1

    for i in numpy.arange(num_examples):
        index = int(reindexing[i])
        centroids[index, :] = centroids[index, :] + x_array[i, :]
        count[index] = count[index] + 1

    computed_centroids = centroids/count

    return computed_centroids
