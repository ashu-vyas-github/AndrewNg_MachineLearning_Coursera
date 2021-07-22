# Machine Learning Online Class
#  Exercise 1: Linear regression with multiple variables
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear regression exercise.
#
#  You will need to complete the following functions in this
#  exericse:
#
#     warmUpExercise.m
#     plotData.m
#     gradientDescent.m
#     computeCost.m
#     gradientDescentMulti.m
#     computeCostMulti.m
#     featureNormalize.m
#     normalEqn.m
#
#  For this part of the exercise, you will need to change some
#  parts of the code below for various experiments (e.g., changing
#  learning rates).
#
import numpy
import matplotlib.pyplot as plt

import lin_reg_funcs

# ================ Part 1: Feature Normalization ================

print('Loading data ...\n')

# Load Data
data = numpy.loadtxt('./ex1data2.txt', dtype="float", delimiter=",")
x_data = data[:, 0:2]
y_data = data[:, 2]
num_examples = y_data.shape[0]  # number of training examples

# Print out some data points
print('First 4 examples from the dataset:')
print(data[0:4, :])

# Scale features and set them to zero mean
print('\nNormalizing Features ...\n')

x_norm, y_norm, x_mean, y_mean, x_std, y_std = lin_reg_funcs.feature_normalize(x_data, y_data, num_examples)
x_norm = numpy.c_[numpy.ones((num_examples, 1)), x_norm]  # Add intercept term to X


# ================ Part 2: Gradient Descent ================

print('Running gradient descent ...')

# Choose some alpha value
alpha = 0.01
num_iters = 50000

# Init Theta and Run Gradient Descent
theta = 1.0*numpy.random.rand(3)
theta, J_history = lin_reg_funcs.gradient_descent_multi(x_norm, y_norm, theta, num_examples, alpha, num_iters)
theta1 = numpy.random.rand(3)
alpha_list = numpy.logspace(start=0, stop=-6, num=7, endpoint=True)

print("Optimum theta values:", theta)

plt.figure(dpi=120)
plt.xlabel("Number of iterations")
plt.ylabel("Cost J(theta)")
plt.xlim((0, num_iters))
plt.ylim((0.0, 1.0))

for alpha in alpha_list:
    theta = theta1
    theta, J_history = lin_reg_funcs.gradient_descent_multi(x_norm, y_norm, theta, num_examples, alpha, num_iters)
    plt.plot(numpy.arange(num_iters), J_history, label="alpha = {}".format(alpha), linewidth=0.75)

plt.legend(bbox_to_anchor=(1.025, 1.0), loc='upper left')
plt.tight_layout()
plt.grid(b=True, which='major', axis='both', linestyle='--', linewidth=0.5)
plt.show()

predict1 = numpy.dot(numpy.array([1.0, (1650-x_mean[0])/x_std[0], (3-x_mean[1])/x_std[1]]), theta)
predict1 = (predict1*y_std) + y_mean
print("\nPredicted price of a 1650 sq-ft, 3 br house {}\n".format(predict1))


# ================ Part 3: Normal Equations ================

print("Solving with normal equations...")
theta_neqn = lin_reg_funcs.normal_equation(x_norm, y_norm)
print("Optimum theta values:", theta_neqn)

# Estimate the price of a 1650 sq-ft, 3 br house
predict2 = numpy.dot(numpy.array([1.0, (1650-x_mean[0])/x_std[0], (3-x_mean[1])/x_std[1]]), theta_neqn)
predict2 = (predict2*y_std) + y_mean
print("\nUsing normal equation, predicted price of a 1650 sq-ft, 3 br house ${}\n".format(predict2))

print("Difference between Gradient Descent and Normal Equation predicted prices...")
print("GD-price - NE-price = ${}".format(predict1 - predict2))
print("\n\nDone")
