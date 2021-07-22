import numpy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import kmeans_pca_funcs


def run_kmeans(find_closest_centroids_func, compute_centroids_func, x_array, centroids, max_iters=10, plot_progress=False):
    """
    Runs the K-means algorithm.

    Parameters
    ----------
    x_array : array_like
        The data set of size (m, n). Each row of x_array is a single example of n dimensions. The
        data set is a total of m examples.

    centroids : array_like
        Initial centroid location for each clusters. This is a matrix of size (K, n). K is the total
        number of clusters and n is the dimensions of each data point.

    findClosestCentroids : func
        A function (implemented by student) reference which computes the cluster assignment for
        each example.

    computeCentroids : func
        A function(implemented by student) reference which computes the centroid of each cluster.

    max_iters : int, optional
        Specifies the total number of interactions of K-Means to execute.

    plot_progress : bool, optional
        A flag that indicates if the function should also plot its progress as the learning happens.
        This is set to false by default.

    Returns
    -------
    centroids : array_like
        A (K x n) matrix of the computed (updated) centroids.
    idx : array_like
        A vector of size (m,) for cluster assignment for each example in the dataset. Each entry
        in idx is within the range [0 ... K-1].

    anim : FuncAnimation, optional
        A matplotlib animation object which can be used to embed a video within the jupyter
        notebook. This is only returned if `plot_progress` is `True`.
    """
    num_centroids_K, ncols = centroids.shape
    num_examples, num_features = x_array.shape
    idx = None
    idx_history = numpy.zeros((max_iters, num_examples))
    centroid_history = numpy.zeros((max_iters, num_centroids_K, ncols))

    for i in range(max_iters):
        idx = find_closest_centroids_func(x_array, centroids)
        idx_history[i, :] = idx
        centroid_history[i, :, :] = centroids
        centroids = compute_centroids_func(x_array, idx, num_centroids_K)

    if plot_progress is True:
        fig = plt.figure(dpi=120)
        anim = FuncAnimation(fig, kmeans_pca_funcs.plot_progress_kmeans, frames=max_iters,
                             interval=500, repeat_delay=2, fargs=(x_array, centroid_history, idx_history))
        return centroids, idx, anim
    else:
        return centroids, idx
