import numpy


def kmeans_init_centroids(x_array, num_centroids_K):
    """
    This function initializes K centroids that are to be used in K-means on the dataset x_array.

    Parameters
    ----------
    x_array : array_like
        The dataset of size (m x n).

    num_centroids_K : int
        The number of clusters.

    Returns
    -------
    rand_init_centroids : array_like
        Centroids of the clusters. This is a matrix of size (K x n).

    Instructions
    ------------
    You should set centroids to randomly chosen examples from the dataset x_array.
    """
    numpy.random.seed(seed=42)
    num_examples, num_features = x_array.shape
    rand_init_centroids = numpy.zeros((num_centroids_K, num_features))

    randidx = numpy.random.permutation(num_examples)
    # Take the first K examples as centroids
    rand_init_centroids = x_array[randidx[:num_centroids_K], :]

    return rand_init_centroids
