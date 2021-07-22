# Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the logistic
#  regression exercise. You will need to complete the following functions
#  in this exericse:
#
#     sigmoid.m
#     costFunction.m
#     predict.m
#     costFunctionReg.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
import numpy
import scipy.optimize as optimization

import log_reg_funcs

import warnings
# invalid value encountered in multiply, divide by zero encountered in log
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

data = numpy.loadtxt('./ex2data1.txt', delimiter=',', dtype='float')
x_data = data[:, 0:2]
y_data = data[:, 2]

print("Few examples of the dataset:")
print(data[:4, :])

# ==================== Part 1: Plotting ====================
#  We start the exercise by first plotting the data to understand the
#  the problem we are working with.

print("\nPlotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.\n")
log_reg_funcs.plot_data(x_data, y_data)

# ============ Part 2: Compute Cost and Gradient ============
#  In this part of the exercise, you will implement the cost and gradient
#  for logistic regression. You neeed to complete the code in
#  costFunction.m

num_examples, num_features = x_data.shape
x_data = numpy.c_[numpy.ones((num_examples, 1)), x_data]  # Add intercept term to x_data
initial_theta = numpy.zeros((num_features+1))  # Initialize fitting parameters

# Compute and display initial cost and gradient
cost_J = log_reg_funcs.cost_function(initial_theta, x_data, y_data)
gradient = log_reg_funcs.gradient_function(initial_theta, x_data, y_data)

print("Cost at initial theta (zeros):", cost_J)
print("Expected cost (approx): 0.693\n")
print("Gradient at initial theta (zeros):", gradient)
print("Expected gradients (approx): [-0.1000 -12.0092 -11.2628]")

# Compute and display cost and gradient with non-zero theta
test_theta = numpy.array([-24, 0.2, 0.2])
cost_J = log_reg_funcs.cost_function(test_theta, x_data, y_data)
gradient = log_reg_funcs.gradient_function(test_theta, x_data, y_data)
print("\nCost at test theta:", cost_J)
print("Expected cost (approx): 0.218\n")
print("Gradient at test theta:", gradient)
print("Expected gradients (approx): [0.043 2.566 2.647]")

# ============= Part 3: Optimizing using fminunc  =============
#  In this exercise, you will use a built-in function (fminunc) to find the
#  optimal parameters theta.

#  Set options for fminunc
results = optimization.fmin_bfgs(f=log_reg_funcs.cost_function, x0=initial_theta,
                                 fprime=log_reg_funcs.gradient_function, args=(x_data, y_data), maxiter=1000, full_output=True)

# Print theta to screen
print("Cost at theta found by fminunc equivalent fmin_bfgs:", results[1])
print("Expected cost (approx): 0.203\n")
print("theta:", results[0])
print("Expected theta (approx): [-25.161 0.206 0.201]\n")

# Plot decision boundary
log_reg_funcs.plot_decision_boundary(results[0], x_data, y_data)


# ============== Part 4: Predict and Accuracies ==============
#  After learning the parameters, you'll like to use it to predict the outcomes
#  on unseen data. In this part, you will use the logistic regression model
#  to predict the probability that a student with score 45 on exam 1 and
#  score 85 on exam 2 will be admitted.
#
#  Furthermore, you will compute the training and test set accuracies of
#  our model.
#
#  Your task is to complete the code in predict.m

#  Predict probability for a student with score 45 on exam 1
#  and score 85 on exam 2

test_observation = numpy.array([1.0, 45.0, 85.0])
test_probability = log_reg_funcs.sigmoid_function(numpy.dot(test_observation, results[0]))
print("For a student with scores 45 and 85, we predict an admission probability of ", test_probability)
print("Expected value: 0.775 +/- 0.002\n")

# Compute accuracy on our training set
train_predicted_labels = log_reg_funcs.predict(results[0], x_data)
correct = numpy.sum(train_predicted_labels.astype(int) == y_data)
train_accuracy = correct/num_examples

print("Train Accuracy:", train_accuracy*100)
print("Expected accuracy (approx): 89.0%\n")
