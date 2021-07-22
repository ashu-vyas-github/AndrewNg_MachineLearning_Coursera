import numpy
import matplotlib
import matplotlib.pyplot as plt


def plot_progress_kmeans(iteration, x_array, centroid_history, idx_history):
    """
    A helper function that displays the progress of k-Means as it is running. It is intended for use
    only with 2D data. It plots data points with colors assigned to each centroid. With the
    previous centroids, it also plots a line between the previous locations and current locations
    of the centroids.

    Parameters
    ----------
    iteration : int
        Current iteration number of k-means. Used for matplotlib animation function.

    x_array : array_like
        The dataset, which is a matrix (m x n). Note since the plot only supports 2D data, n should
        be equal to 2.

    centroid_history : list
        A list of computed centroids for all iteration.

    idx_history : list
        A list of computed assigned indices for all iterations.
    """
    max_iters, num_centroids_K, ncols = centroid_history.shape
    plt.gcf().clf()
    cmap = plt.cm.rainbow
    norm = matplotlib.colors.Normalize(vmin=0, vmax=2)

    for k in range(num_centroids_K):
        current = numpy.stack([c[k, :] for c in centroid_history[:iteration+1]], axis=0)
        plt.plot(current[:, 0], current[:, 1], '-Xk', mec='k', lw=2, ms=10, mfc=cmap(norm(k)), mew=2)
        plt.scatter(x_array[:, 0], x_array[:, 1], c=idx_history[iteration], cmap=cmap, marker='o', s=8**2, linewidths=1)
    plt.grid(b=True, which='major', axis='both', linestyle='--', linewidth=0.5)
    plt.title('Iteration number %d' % (iteration+1))
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.tight_layout()

    return plt
