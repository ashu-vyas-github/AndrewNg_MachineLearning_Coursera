# Machine Learning Online Class
#  Exercise 8 | Anomaly Detection and Collaborative Filtering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     estimateGaussian.m
#     selectThreshold.m
#     cofiCostFunc.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import numpy

import matplotlib.pyplot as plt

from scipy.io import loadmat
from scipy import optimize

import ano_rec_funcs

# ================== Part 1: Load Example Dataset  ===================
print("Visualizing example dataset for outlier detection...\n")

data = loadmat("./ex8data1.mat")
x_train = data["X"]
x_valid = data["Xval"]
y_valid = data["yval"]
y_valid = y_valid.reshape(-1)
num_examples, num_features = x_train.shape
print("Ex.8.1 Anomaly #training examples:", num_examples, "#features:", num_features, "\n")

# Visualize the example dataset
plt.figure(dpi=120)
plt.title("Anomaly detection training set")
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput Mb/s")
plt.axis([0, 30, 0, 30])
plt.scatter(x_train[:, 0], x_train[:, 1], marker="o", s=50, linewidths=1.0, color="red")
plt.grid(b=True, which="major", axis="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

# ================== Part 2: Estimate the dataset statistics ===================
print("Visualizing Gaussian fit...\n")

mu, sigma2 = ano_rec_funcs.estimate_gaussian(x_train)  # Estimate mu and sigma2

# Returns the density of the multivariate normal at each data point (row) of X
probabilities_train = ano_rec_funcs.multivariate_gaussian(x_train, mu, sigma2)

#  Visualize the fit
plt.figure(dpi=120)
plt.title("Ex.8.1 Anomaly density Gaussian contours")
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput Mb/s")
plt.axis([0, 30, 0, 30])
ano_rec_funcs.visualize_fit(x_train,  mu, sigma2)
plt.grid(b=True, which="major", axis="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

# ================== Part 3: Find Outliers ===================
probabilities_valid = ano_rec_funcs.multivariate_gaussian(x_valid, mu, sigma2)

epsilon, f1 = ano_rec_funcs.select_threshold(y_valid, probabilities_valid)
print("Computed best epsilon: %.2e" % epsilon)
print("Expected best epsilon: 8.99e-05")
print("Computed best F1: %f" % f1)
print("Expected best F1: 0.875000")

#  Find the outliers in the training set and plot the
outliers = probabilities_train < epsilon

#  Visualize the fit
plt.figure(dpi=120)
plt.title("Ex.8.1 Anomaly probable outliers")
plt.xlabel("Latency (ms)")
plt.ylabel("Throughput Mb/s")
plt.axis([0, 30, 0, 30])
ano_rec_funcs.visualize_fit(x_train,  mu, sigma2)
plt.plot(x_train[outliers, 0], x_train[outliers, 1], "ro", ms=10,
         mfc="None", mew=2)  # Draw a red circle around those outliers
plt.grid(b=True, which="major", axis="both", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

# ================== Part 4: Multidimensional Outliers ===================
#  Loads the second dataset. You should now have the
data = loadmat("./ex8data2.mat")
x_train = data["X"]
x_valid = data["Xval"]
y_valid = data["yval"]
y_valid = y_valid.reshape(-1)
num_examples, num_features = x_train.shape
print("\nEx.8.2 Anomaly #training examples:", num_examples, "#features:", num_features, "\n")


mu, sigma2 = ano_rec_funcs.estimate_gaussian(x_train)
probabilities_train = ano_rec_funcs.multivariate_gaussian(x_train, mu, sigma2)
probabilities_valid = ano_rec_funcs.multivariate_gaussian(x_valid, mu, sigma2)
epsilon, f1 = ano_rec_funcs.select_threshold(y_valid, probabilities_valid)

num_outliers = numpy.sum(probabilities_train < epsilon)

print("Computed best epsilon: %.2e" % epsilon)
print("Expected best epsilon: 1.38e-18")
print("Computed best F1: %f" % f1)
print("Expected best F1: 0.615385")
print("Total outliers found: %d" % num_outliers)

print("\n\n\nDone\n\n\n")
