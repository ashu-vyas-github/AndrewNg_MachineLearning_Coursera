# Machine Learning Online Class
#  Exercise 6 | Support Vector Machines
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     gaussianKernel.m
#     dataset3Params.m
#     processEmail.m
#     emailFeatures.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

import numpy
from scipy.io import loadmat
from scipy import optimize

import svm_funcs

numpy.random.seed(seed=42)  # make randomization predictable for debugging and reproducibility

# =============== Part 1: Loading and Visualizing Data ================
print("Loading and Visualizing Dataset 1...")

data = loadmat("./ex6data1.mat")  # Load from ex6data1
x_data = data['X']
y_data = data['y']
y_data = y_data.reshape(-1)
num_examples, num_features = x_data.shape
print("Example 6.1. #examples:", num_examples, "#features:", num_features)
svm_funcs.plot_data(x_data, y_data)  # Plot training data

# ==================== Part 2: Training Linear SVM ====================
print("\nTraining Linear SVM...\n")

reg_C = 1
model = svm_funcs.svm_train(svm_funcs.linear_kernel, x_data, y_data, reg_C, tol=1e-3, max_passes=20)
svm_funcs.plot_data(x_data, y_data, model=model, linear_boundary=True)

# # Try this loop to visualize different values of C and effects on decision boundary
# c_list = numpy.logspace(-4, 4, num=9, endpoint=True)
# for oneC in c_list:
#     model = svm_funcs.svm_train(svm_funcs.linear_kernel, x_data, y_data, oneC, tol=1e-3, max_passes=20)
#     svm_funcs.plot_data(x_data, y_data, model=model, reg_C=oneC, boundary=True)

# =============== Part 3: Implementing Gaussian Kernel ===============
print("\nEvaluating the Gaussian Kernel...\n")

x1 = numpy.array([1, 2, 1])
x2 = numpy.array([0, 4, -1])
sigma = 2
rbf = svm_funcs.gaussian_kernel(x1, x2, sigma)

print("Gaussian Kernel between x1 = {v1}, x2 = {v2}, sigma = {v3},".format(v1=x1, v2=x2, v3=sigma))
print("Computed kernel:", rbf)
print("Expected kernel: 0.324652\n")

# =============== Part 4: Visualizing Dataset 2 ================
print("Loading and Visualizing Dataset 2...\n")

data = loadmat("./ex6data2.mat")  # Load from ex6data2
x_data = data['X']
y_data = data['y']
y_data = y_data.reshape(-1)
num_examples, num_features = x_data.shape
print("Example 6.2. #examples:", num_examples, "#features:", num_features)
svm_funcs.plot_data(x_data, y_data)  # Plot training data

# ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
print("\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes)...\n")
# SVM Parameters
reg_C = 1
sigma = 0.1

model = svm_funcs.svm_train(svm_funcs.gaussian_kernel, x_data, y_data, reg_C, tol=1e-3, max_passes=20, args=(sigma,))
svm_funcs.plot_data(x_data, y_data, model=model, reg_C=reg_C, linear_boundary=False, nonlinear_boundary=True)

# =============== Part 6: Visualizing Dataset 3 ================
print("Loading and Visualizing Dataset 3...\n")

data = loadmat("./ex6data3.mat")  # Load from ex6data2
x_train = data['X']
y_train = data['y']
y_train = y_train.reshape(-1)
num_examples_train, num_features_train = x_train.shape
print("Example 6.3. training #examples:", num_examples_train, "#features:", num_features_train)
x_valid = data['Xval']
y_valid = data['yval']
y_valid = y_valid.reshape(-1)
num_examples_valid, num_features_valid = x_train.shape
print("Example 6.3. validation #examples:", num_examples_valid, "#features:", num_features_valid)
svm_funcs.plot_data(x_train, y_train)  # Plot training data

# ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========
# Try different SVM Parameters here
reg_C, sigma = svm_funcs.dataset3_params(x_train, y_train, x_valid, y_valid)

# Train the SVM on cross validated and optimum C, sigma
model = svm_funcs.svm_train(svm_funcs.gaussian_kernel, x_train, y_train, reg_C, tol=1e-3, max_passes=20, args=(sigma,))
svm_funcs.plot_data(x_train, y_train, model=model, reg_C=reg_C, linear_boundary=False, nonlinear_boundary=True)

print("C:", reg_C)
print("Sigma:", sigma)
