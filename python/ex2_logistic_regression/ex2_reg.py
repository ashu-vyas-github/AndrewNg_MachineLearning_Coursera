# Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import copy
import numpy
import scipy.optimize as optimization

import log_reg_funcs

# Load Data
#  The first two columns contains the X values and the third column
#  contains the label (y).

data = numpy.loadtxt("./ex2data2.txt", delimiter=",", dtype="float64")
x_data = data[:, 0:2]
y_data = data[:, 2]
num_examples, num_features = x_data.shape

print("Few examples of the dataset:")
print(data[:4, :])

print("\nPlotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n")
log_reg_funcs.plot_data(x_data, y_data, xaxis_text="Microchip Test 1", yaxis_text="Microchip Test 2",
                        xaxis_lim=(-1.5, 1.5), yaxis_lim=(-1.5, 1.5))


# =========== Part 1: Regularized Logistic Regression ============
# Add Polynomial Features
# Note that mapFeature also adds a column of ones for us, so the intercept term is handled
new_data_matrix = log_reg_funcs.map_feature(x_data[:, 0], x_data[:, 1], num_examples, degree=6)

initial_theta = numpy.zeros((new_data_matrix.shape[1]))  # Initialize fitting parameters
regularization_lambda = 1.0  # Set regularization parameter lambda to 1

# Compute and display initial cost and gradient for regularized logistic regression
cost_J = log_reg_funcs.cost_function_regularized(
    initial_theta, new_data_matrix, y_data, regularization_lambda=regularization_lambda)

print("Cost at initial theta (zeros):", cost_J)
print("Expected cost (approx): 0.693\n")

gradient = log_reg_funcs.gradient_function_regularized(
    initial_theta, new_data_matrix, y_data, regularization_lambda=regularization_lambda)

print("Gradient at initial theta (zeros) - first five values only:", gradient[:5])
print("Expected gradients (approx) - first five values only: [0.0085 0.0188 0.0001 0.0503 0.0115]")

# Compute and display cost and gradient with all-ones theta and lambda = 10
test_theta = numpy.ones((new_data_matrix.shape[1]))
regularization_lambda = 10.0

cost_J = log_reg_funcs.cost_function_regularized(
    test_theta, new_data_matrix, y_data, regularization_lambda=regularization_lambda)

gradient = log_reg_funcs.gradient_function_regularized(
    test_theta, new_data_matrix, y_data, regularization_lambda=regularization_lambda)

print("\nCost at test theta (with lambda = 10):", cost_J)
print("Expected cost (approx): 3.16\n")
print("Gradient at test theta - first five values only:", gradient[:5])
print("Expected gradients (approx) - first five values only: [1.3460 0.1614 0.1948 0.2269 0.0922]")


# ============= Part 2: Regularization and Accuracies =============int
#  Optional Exercise:int
#  Try the following values of lambda (0, 1, 10, 100).

regularization_lambda = 1e-12
initial_theta = numpy.zeros((new_data_matrix.shape[1]))  # Initialize fitting parameters

# norm_data = copy.deepcopy(new_data_matrix.astype('float64'))
# x_norm, y_norm, x_mean, y_mean, x_std, y_std = log_reg_funcs.feature_normalize(norm_data[:, 3:], y_data, num_examples)
# norm_data[:, 3:] = x_norm
norm_data = new_data_matrix

options = {'maxiter': 10000, 'gtol': 1e-9, 'norm': numpy.inf}
results = optimization.minimize(fun=log_reg_funcs.cost_function_regularized, x0=initial_theta,
                                method="BFGS", args=(norm_data.astype('float64'), y_data, regularization_lambda), options=options)

print("")
print(results.fun)
print(results.message)
print(results.success)
print(results.x)

# Plot Boundary
log_reg_funcs.plot_decision_boundary(results.x, new_data_matrix, y_data)

# Compute accuracy on our training set
train_predicted_labels = log_reg_funcs.predict(results.x, norm_data)
correct = numpy.sum(train_predicted_labels.astype(int) == y_data)
train_accuracy = correct/num_examples

print("\nTrain Accuracy:", train_accuracy*100)
print("Expected accuracy (approx): 83.1%\n")
