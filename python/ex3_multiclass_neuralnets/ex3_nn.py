# Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

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

# Setup the parameters you will use for this exercise (note that we have mapped "0" to label 10)
input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10


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

# Randomly select 100 data points to display
rand_indices = numpy.random.choice(num_examples, 100, replace=False)
sel = x_data[rand_indices, :]

mcc_nn_funcs.display_data(sel)

# ================ Part 2: Loading Pameters ================

print("\nLoading Saved Neural Network Parameters...\n")

# Load the weights into variables Theta1 and Theta2
thetas_weights = scipy.io.loadmat("./ex3weights.mat")
theta1 = thetas_weights['Theta1']
theta2 = thetas_weights['Theta2']

# swap first and last columns of theta2, due to legacy from MATLAB indexing,
# since the weight file ex3weights.mat was saved based on MATLAB indexing
theta2 = numpy.roll(theta2, 1, axis=0)

# ================= Part 3: Implement Predict =================

predictions = mcc_nn_funcs.predict(theta1, theta2, x_data)
accuracy = numpy.mean(predictions == y_data) * 100

print("Training Set Accuracy:", accuracy)

#  To give you an idea of the network's output, you can also run through the examples one at the a time to see what it is predicting.
indices = numpy.random.permutation(num_examples)
if indices.size > 0:
    i, indices = indices[0], indices[1:]
    oneexample = numpy.reshape(x_data[i, :], (1, x_data[i, :].shape[0]))
    onelabel = y_data[i]
    pred = mcc_nn_funcs.predict(theta1, theta2, oneexample)
    print("Predicted label:", pred[0])
    print("True label", onelabel)
else:
    print('No more images to display!')
