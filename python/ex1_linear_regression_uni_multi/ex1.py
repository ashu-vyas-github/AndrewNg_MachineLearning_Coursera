# Machine Learning Online Class - Exercise 1: Linear Regression

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     warmup_exercise.py
#     plotData.m
#     computeCost.m
#     gradientDescent.m
#     gradientDescentMulti.m
#     computeCostMulti.m
#     featureNormalize.m
#     normalEqn.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
# x refers to the population size in 10,000s
# y refers to the profit in $10,000s
#
import numpy
import matplotlib.pyplot as plt

import lin_reg_funcs

# ==================== Part 1: Basic Function ====================
# Complete warmup_exercise.py
print('Running warmup_exercise ... \n')
print('5x5 Identity Matrix: \n')
print(lin_reg_funcs.warmup_exercise())


# ======================= Part 2: Plotting =======================
print("\nPlotting Data...\n")
data = numpy.loadtxt('./ex1data1.txt', dtype="float", delimiter=",")
x_data = data[:, 0]
y_data = data[:, 1]
num_examples = y_data.shape[0]  # number of training examples

# Plot Data
# Note: You have to complete the code in plotData.m
lin_reg_funcs.plot_data(x_data, y_data)


# =================== Part 3: Cost and Gradient descent ===================
x_data = numpy.c_[numpy.ones((num_examples, 1)), data[:, 0]]  # Add a column of ones to x
theta = numpy.zeros((2, 1))  # initialize fitting parameters

# Some gradient descent settings
iterations = 10000
alpha = 0.0001

print('\nTesting the cost function ...\n')
# compute and display initial cost
cost_J = lin_reg_funcs.compute_cost(x_data, y_data, theta, num_examples)
print('With theta = [0, 0]\nCost computed = {:.2f}'.format(cost_J))
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
theta = numpy.array([-1.0, 2.0])
cost_J = lin_reg_funcs.compute_cost(x_data, y_data, theta, num_examples)
print('\nWith theta = [-1, 2]\nCost computed = {:.2f}'.format(cost_J))
print('Expected cost value (approx) 54.24\n')

print('\nRunning Gradient Descent ...\n')
# run gradient descent
theta, J_history = lin_reg_funcs.gradient_descent(x_data, y_data, theta, num_examples, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:', theta)
print('Expected theta values (approx)', numpy.array([-3.6303, 1.1664]), "\n")

# Plot the linear fit
line_data = numpy.reshape(numpy.dot(x_data, theta), y_data.shape)
lin_reg_funcs.plot_data_line(x_data[:, 1], y_data, line_data)

# Predict values for population sizes of 35,000 and 70,000
predict1 = numpy.dot(numpy.array([1.0, 3.5]), theta)
print('For population = 35,000, we predict a profit of {}'.format(predict1*10000))
predict2 = numpy.dot(numpy.array([1.0, 7.0]), theta)
print('For population = 70,000, we predict a profit of {}'.format(predict2*10000))


# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')

# Grid over which we will calculate J
theta0_vals = numpy.linspace(-10, 10, 100)
theta1_vals = numpy.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = numpy.zeros((theta0_vals.shape[0], theta1_vals.shape[0]))

# Fill out J_vals
for ix in range(theta0_vals.shape[0]):
    for jx in range(theta1_vals.shape[0]):
        t = numpy.array([theta0_vals[ix], theta1_vals[jx]])
        J_vals[ix, jx] = lin_reg_funcs.compute_cost(x_data, y_data, t, num_examples)


# Because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
# J_vals = J_vals'
# Surface plot
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals, cmap='viridis', edgecolor='none')
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.show()


# Contour plot
fig = plt.figure()
ax = plt.axes()
# Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
ax.contour(theta0_vals, theta1_vals, J_vals, numpy.logspace(-2, 3, 50))
plt.xlabel('theta_0')
plt.ylabel('theta_1')
ax.plot(theta[0], theta[1], marker='x', markersize=10, linewidth=2)
plt.show()
