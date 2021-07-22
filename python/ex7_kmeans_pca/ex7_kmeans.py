# Machine Learning Online Class
#  Exercise 7 | Principle Component Analysis and K-Means Clustering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     pca.m
#     projectData.m
#     recoverData.m
#     computeCentroids.m
#     findClosestCentroids.m
#     kMeansInitCentroids.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import numpy
import matplotlib.pyplot as plt

from skimage.io import imread
from scipy.io import loadmat

import kmeans_pca_funcs

# ================= Part 1: Find Closest Centroids ====================
print("Finding closest centroids...\n")

data = loadmat("ex7data2.mat")  # Load an example dataset that we will be using
x_data = data['X']
num_examples, num_features = x_data.shape
print("Ex 7.1 #training examples:", num_examples, "and #features:", num_features, "\n")

# Select an initial set of centroids
num_centroids_K = 3  # 3 Centroids
initial_centroids = numpy.array([[3, 3], [6, 2], [8, 5]])

# Find the closest centroids for the examples using the initial_centroids
idx = kmeans_pca_funcs.find_closest_centroids(x_data, initial_centroids)

print("Closest centroids for the first 3 examples")
print("Computed centroids:", idx[:3])
print("Expected centroids: [1 3 2]")

# ===================== Part 2: Compute Means =========================
print("\nComputing centroids means.\n\n")

# Compute means based on the closest centroids found in the previous part.
centroids = kmeans_pca_funcs.compute_centroids(x_data, idx, num_centroids_K)

print("Centroids computed after initial finding of closest centroids:")
print(centroids)
print("\nThe centroids should be")
print("[[ 2.428301 3.157924 ]")
print(" [ 5.813503 2.633656 ]")
print(" [ 7.119387 3.616684 ]]")


# =================== Part 3: K-Means Clustering ======================
print("\nRunning K-Means clustering on example dataset...\n")

max_iters = 10
initial_centroids = numpy.array([[3, 3], [6, 2], [8, 5]])

# Run K-Means algorithm. The 'true' at the end tells our function to plot the progress of K-Means
# centroids, idx, anim = kmeans_pca_funcs.run_kmeans(
# kmeans_pca_funcs.find_closest_centroids, kmeans_pca_funcs.compute_centroids, x_data, initial_centroids, max_iters, True)

# anim

print("\nK-Means Done.\n\n")

# ============= Part 4: K-Means Clustering on Pixels ===============
print("\nRunning K-Means clustering on pixels from an image.\n")

num_centroids_K = 8
img = imread("./bird_small.png")  # Load an image of a bird
# img_mat = loadmat("./bird_small.mat") # same image in MATLAB matrix
img = img/255  # normalize pixels between 0 to 1
nrows, ncols, nchannels = img.shape
img_2D = img.reshape(-1, nchannels)  # reshape images as MxN=p, #total_pixels x #channels

# When using K-Means, it is important to randomly initialize centroids
# You should complete the code in kMeansInitCentroids above before proceeding
initial_centroids = kmeans_pca_funcs.kmeans_init_centroids(img_2D, num_centroids_K)

# Run K-Means
centroids, idx = kmeans_pca_funcs.run_kmeans(
    kmeans_pca_funcs.find_closest_centroids, kmeans_pca_funcs.compute_centroids, img_2D, initial_centroids, max_iters, False)

# ================= Part 5: Image Compression ======================
idx = idx - 1  # reindexing from 0 onwards, rather than 1 onwards indexing

img_recovered = centroids[idx, :]
img_recovered = img_recovered.reshape(img.shape)

img = img*255
img = img.astype(int)

img_recovered = img_recovered*255
img_recovered = img_recovered.astype(int)

# Display the original image, rescale back by 255
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(img)
ax[0].set_title('Original')
ax[0].grid(False)

# Display compressed image, rescale back by 255
ax[1].imshow(img_recovered)
ax[1].set_title('Compressed, with %d colors' % num_centroids_K)
ax[1].grid(False)
# fig.tight_layout()
fig.show()

print("\n\nDone\n\n")
