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

# ================== Part 1: Load Example Dataset  ===================
print("Visualizing example dataset for PCA.\n\n")

#  The following command loads the dataset. You should now have the variable X in your environment
data = loadmat("./ex7data1.mat")
x_data = data['X']
num_examples, num_features = x_data.shape
print("Ex.7.2 PCA #training examples:", num_examples, "#features:", num_features, "\n")

#  Visualize the example dataset
plt.figure(dpi=120)
plt.title("PCA dataset 1")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.scatter(x_data[:, 0], x_data[:, 1], marker='o', s=16, linewidths=1.0, color='blue')
plt.xlim(0.5, 6.5)
plt.ylim(2, 8)
plt.gca().set_aspect('equal')
plt.grid(b=True, which='major', axis='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()

# =============== Part 2: Principal Component Analysis ===============
#  Before running PCA, it is important to first normalize X
x_norm, mu, sigma = kmeans_pca_funcs.feature_normalize(x_data)
mat_U, mat_S = kmeans_pca_funcs.pca(x_norm)  # Run PCA

#  Draw the eigenvectors centered at mean of data. These lines show the directions of maximum variations in the dataset.
fig, ax = plt.subplots()
plt.title("Eigenvectors")
ax.scatter(x_data[:, 0], x_data[:, 1], marker='o', s=16, linewidths=1.0, color='blue')

for i in range(num_features):
    ax.arrow(mu[0], mu[1], 1.5*mat_S[i]*mat_U[0, i], 1.5*mat_S[i]*mat_U[1, i],
             head_width=0.25, head_length=0.2, fc='k', ec='k', lw=2, zorder=1000)

ax.axis([0.5, 6.5, 2, 8])
ax.set_aspect('equal')
ax.grid(b=True, which='major', axis='both', linestyle='--', linewidth=0.5)
fig.tight_layout()
fig.show()

print("Computed eignevector = [{:.6f} {:.6f}]".format(mat_U[0, 0], mat_U[1, 0]))
print("Expected eigenvector = [-0.707107 -0.707107]")


# =================== Part 3: Dimension Reduction ===================
#  Project the data onto K = 1 dimension
ndims_K = 1
projection_Z = kmeans_pca_funcs.project_data(x_norm, mat_U, ndims_K)
print("\nComputed projection value: {:.6f}".format(projection_Z[0, 0]))
print("Expected projection value: 1.481274")

recovered_X = kmeans_pca_funcs.recover_data(projection_Z, mat_U, ndims_K)
print("\nComputed recovered value of the first example: [{:.6f} {:.6f}]".format(recovered_X[0, 0], recovered_X[0, 1]))
print("Expected recovered value of the first example: [-1.047419 -1.047419]")

#  Plot the normalized dataset (returned from featureNormalize)
fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
plt.title("Recovered data")
ax.scatter(x_norm[:, 0], x_norm[:, 1], marker='o', s=16, linewidths=1.0, color='blue')
ax.set_aspect('equal')
plt.axis([-3, 2.75, -3, 2.75])
ax.grid(b=True, which='major', axis='both', linestyle='--', linewidth=0.5)

# Draw lines connecting the projected points to the original points
ax.scatter(recovered_X[:, 0], recovered_X[:, 1], marker='o', s=16, linewidths=1.0, color='red')
for xnorm, xrec in zip(x_norm, recovered_X):
    ax.plot([xnorm[0], xrec[0]], [xnorm[1], xrec[1]], '--k', lw=1)

fig.tight_layout()
fig.show()


# =============== Part 4: Loading and Visualizing Face Data =============
#  Load Face dataset
data = loadmat('./ex7faces.mat')
x_faces = data['X']
num_examples, num_features = x_faces.shape
print("\nEx.7.3 Faces PCA #training examples:", num_examples, "#features:", num_features, "\n")

#  Display the first 100 faces in the dataset
kmeans_pca_funcs.display_data(x_faces[:100, :], figsize=(8, 8))

# =========== Part 5: PCA on Face Data: Eigenfaces  ===================
# normalize X by subtracting the mean value from each feature
x_faces_norm, mu, sigma = kmeans_pca_funcs.feature_normalize(x_faces)

faces_U, faces_S = kmeans_pca_funcs.pca(x_faces_norm)  # Run PCA

# Visualize the top 36 eigenvectors found
kmeans_pca_funcs.display_data(numpy.transpose(faces_U[:, :36]), figsize=(8, 8))

# ============= Part 6: Dimension Reduction for Faces =================
#  Project images to the eigen space using the top k eigenvectors if you are applying a machine learning algorithm
ndims_K = 100

faces_Z = kmeans_pca_funcs.project_data(x_faces_norm, faces_U, ndims_K)

print("The projected data Z has a shape of:", faces_Z.shape)

# # ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
#  Project images to the eigen space using the top K eigen vectors and visualize only using those K dimensions
#  Compare to the original input, which is also displayed
ndims_K = 100
recovered_faces_X = kmeans_pca_funcs.recover_data(faces_Z, faces_U, ndims_K)

# Display normalized data
kmeans_pca_funcs.display_data(x_faces_norm[:100, :], figsize=(6, 6))
plt.gcf().suptitle('Original faces')

# Display reconstructed data from only k eigenfaces
kmeans_pca_funcs.display_data(recovered_faces_X[:100, :], figsize=(6, 6))
plt.gcf().suptitle('Recovered faces')

# === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===

# # If imread does not work for you, you can try instead
img = imread("./bird_small.png")
img = img/255
img_2D = img.reshape(-1, 3)

# perform the K-means clustering again here
num_centroids_K = 16
max_iters = 10
initial_centroids = kmeans_pca_funcs.kmeans_init_centroids(img_2D, num_centroids_K)
centroids, idx = kmeans_pca_funcs.run_kmeans(
    kmeans_pca_funcs.find_closest_centroids, kmeans_pca_funcs.compute_centroids, img_2D, initial_centroids, max_iters, False)

#  Sample 1000 random indexes (since working with all the data is too expensive. If you have a fast computer, you may increase this.
sel = numpy.random.choice(img_2D.shape[0], size=1000)

fig = plt.figure(figsize=(6, 6), dpi=120)
ax = fig.add_subplot(111, projection='3d')

ax.scatter(img_2D[sel, 0], img_2D[sel, 1], img_2D[sel, 2], cmap='rainbow', c=idx[sel], s=8**2)
ax.set_title('Pixel dataset plotted in 3D.\nColor shows centroid memberships')
fig.show()

# === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
# Subtract the mean to use PCA
img_2D_norm, mu, sigma = kmeans_pca_funcs.feature_normalize(img_2D)

# PCA and project the data to 2D
U, S = kmeans_pca_funcs.pca(img_2D_norm)
Z = kmeans_pca_funcs.project_data(img_2D_norm, U, 2)

fig = plt.figure(figsize=(6, 6), dpi=120)
ax = fig.add_subplot(111)

ax.scatter(Z[sel, 0], Z[sel, 1], cmap='rainbow', c=idx[sel], s=64)
ax.set_title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
ax.grid(b=True, which='major', axis='both', linestyle='--', linewidth=0.5)
fig.show()

print("\n\n\nDone\n\n\n")
