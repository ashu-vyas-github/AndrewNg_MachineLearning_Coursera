# Machine Learning Online Class - Exercise 4 Neural Network Learning

#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions
#  in this exericse:
#
#     sigmoidGradient.m
#     randInitializeWeights.m
#     nnCostFunction.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

import numpy
import scipy.io
import scipy.optimize as optimization

import nnl_funcs

# Setup the parameters you will use for this exercise (note that we have mapped "0" to label 10)
input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10

# =========== Part 1: Loading and Visualizing Data =============

# Load Training Data
print("Loading and Visualizing Data...")

mat = scipy.io.loadmat("./ex4data1.mat")
x_data = mat['X']
y_data = mat['y']
ten_idx = numpy.where(y_data == 10)
y_data[ten_idx] = 0  # setting 10s to 0s
num_examples, num_features = x_data.shape
y_data = numpy.reshape(y_data, (num_examples))
print("\nTraining data #examples:", num_examples, "#features:", num_features)

# Randomly select 100 data points to display
rand_indices = numpy.random.choice(num_examples, 100, replace=False)
sel = x_data[rand_indices, :]

nnl_funcs.display_data(sel)

# ================ Part 2: Loading Parameters ================

print("\nLoading Saved Neural Network Parameters...")

# Load the weights into variables Theta1 and Theta2
thetas_weights = scipy.io.loadmat("./ex4weights.mat")
theta1 = thetas_weights['Theta1']
theta2 = thetas_weights['Theta2']

# swap first and last columns of theta2, due to legacy from MATLAB indexing,
# since the weight file ex3weights.mat was saved based on MATLAB indexing
theta2 = numpy.roll(theta2, 1, axis=0)

# Unroll parameters
theta_vector = numpy.concatenate([theta1.ravel(), theta2.ravel()])

# ================ Part 3: Compute Cost (Feedforward) ================
print("\nFeedforward Using Neural Network...")

# Weight regularization parameter (we set this to 0 here).
reg_lambda = 0

cost_J, gradient = nnl_funcs.cost_function_nn(
    theta_vector, x_data, y_data, input_layer_size, hidden_layer_size, num_labels, reg_lambda=reg_lambda)

print("Computed cost:", cost_J)
print("Expected cost: 0.287629")

# =============== Part 4: Implement Regularization ===============
print("\nChecking Cost Function (w/ Regularization)...")

# Weight regularization parameter (we set this to 1 here).
reg_lambda = 1

cost_J, gradient = nnl_funcs.cost_function_nn(
    theta_vector, x_data, y_data, input_layer_size, hidden_layer_size, num_labels, reg_lambda=reg_lambda)

print("Computed cost:", cost_J)
print("Expected cost: 0.383770")

# ================ Part 5: Sigmoid Gradient  ================
print("\nEvaluating sigmoid gradient...")

sg = nnl_funcs.sigmoid_gradient(numpy.array([-1, -0.5, 0, 0.5, 1]))
print("Sigmoid gradient evaluated at [-1 -0.5 0 0.5 1]:")
print(sg)

# ================ Part 6: Initializing Pameters ================
print("\nInitializing Neural Network Parameters...")

initial_theta1 = nnl_funcs.random_initialize_weight(input_layer_size, hidden_layer_size)
initial_theta2 = nnl_funcs.random_initialize_weight(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = numpy.concatenate([initial_theta1.ravel(), initial_theta2.ravel()])
print(initial_nn_params.shape)

# =============== Part 7: Implement Backpropagation ===============
print("\nChecking Backpropagation...")

nnl_funcs.check_nn_gradients(nnl_funcs.cost_function_nn, reg_lambda=reg_lambda)

# =============== Part 8: Implement Regularization ===============
print("\nChecking Backpropagation (w/ Regularization)...")

#  Check gradients by running checkNNGradients
reg_lambda = 3
nnl_funcs.check_nn_gradients(nnl_funcs.cost_function_nn, reg_lambda=reg_lambda)

# Also output the costFunction debugging values
debug_J, _ = nnl_funcs.cost_function_nn(theta_vector, x_data, y_data, input_layer_size,
                                        hidden_layer_size, num_labels, reg_lambda=reg_lambda)

print("\n\nCost at (fixed) debugging parameters (w/ lambda = {v1}): {v2}".format(v1=reg_lambda, v2=debug_J))
print("Expected cost for lambda = 3, this value should be about 0.576051")

# =================== Part 8: Training NN ===================
print("\nTraining Neural Network...")

reg_lambda = 1
options = {'maxiter': None}

# Create "short hand" for the cost function to be minimized


def cost_func_sh(params): return nnl_funcs.cost_function_nn(params, x_data, y_data,
                                                            input_layer_size, hidden_layer_size, num_labels, reg_lambda)


# Now, cost_func_sh is a function that takes in only one argument (the neural network parameters)
results = optimization.minimize(cost_func_sh, initial_nn_params, jac=True, method='CG', options=options)

# get the solution of the optimization
nn_params = results.x

print("Convergence success:", results.success)
print("Computed cost:", results.fun)

# Obtain Theta1 and Theta2 back from nn_params
theta1 = numpy.reshape(nn_params[:hidden_layer_size*(input_layer_size + 1)],
                       (hidden_layer_size, (input_layer_size + 1)))
theta2 = numpy.reshape(nn_params[(hidden_layer_size*(input_layer_size + 1)):], (num_labels, (hidden_layer_size + 1)))

# ================= Part 9: Visualize Weights =================
print("\nVisualizing Neural Network...\n")

# displayData(Theta1(:, 2: end))
nnl_funcs.display_data(theta1[:, 1:])
# nnl_funcs.display_data(theta2[:, 1:])

# ================= Part 10: Implement Predict =================
predictions = nnl_funcs.predict(theta1, theta2, x_data)
accuracy = numpy.mean(predictions == y_data) * 100
print("Training Set Accuracy:", accuracy)
