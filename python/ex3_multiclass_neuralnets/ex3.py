# Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     lrCostFunction.m (logistic regression cost function)
#     oneVsAll.m
#     predictOneVsAll.m
#     predict.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

import numpy
import scipy.io

import mcc_nn_funcs

# Setup the parameters you will use for this part of the exercise (note that we have mapped "0" to label 10)
input_layer_size = 400  # 20x20 Input Images of Digits
num_labels = 10  # 10 labels, from 1 to 10

# =========== Part 1: Loading and Visualizing Data =============

# Load Training Data
print("Loading data...")

mat = scipy.io.loadmat("./ex3data1.mat")
x_data = mat['X']
y_data = mat['y']
ten_idx = numpy.where(y_data == 10)
y_data[ten_idx] = 0  # setting 10s to 0s
num_examples, num_features = x_data.shape
y_data = numpy.reshape(y_data, (num_examples))
print("\nTraining data #examples:", num_examples, "#features:", num_features)

# ============ Part 2a: Vectorize Logistic Regression ============

# Test case for lrCostFunction
print("\nTesting regularized cost funciton...")

theta_t = numpy.array([-2.0, -1.0, 1.0, 2.0])
X_t = numpy.arange(1.0, 16.0, 1.0)/10.0
X_t = numpy.c_[numpy.ones((5)), numpy.reshape(X_t, (5, 3), order='F')]
y_t = numpy.array([1.0, 0.0, 1.0, 0.0, 1.0])  # >= 0.5
lambda_t = 3
cost_J, gradient = mcc_nn_funcs.cost_function_regularized(theta_t, X_t, y_t, lambda_t)
# gradient = mcc_nn_funcs.gradient_function_regularized(theta_t, X_t, y_t, lambda_t)

print("Expected cost: 2.534819")
print("Computed cost:", cost_J)
print("Expected gradients: [0.146561 -0.548558 0.724722 1.398003]")
print("Computed gradients:", gradient)

# ============ Part 2b: One-vs-All Training ============
print("\nTraining One-vs-All Logistic Regression...\n")

lambda_train = 1e-1
all_theta, cost_values, convergence_flag = mcc_nn_funcs.one_vs_all(x_data, y_data, reg_lambda=lambda_train)

# ================ Part 3: Predict for One-Vs-All ================

predictions = mcc_nn_funcs.predict_one_vs_all(all_theta, x_data)

print('Training Set Accuracy: {:.2f}%'.format(numpy.mean(predictions == y_data) * 100))
