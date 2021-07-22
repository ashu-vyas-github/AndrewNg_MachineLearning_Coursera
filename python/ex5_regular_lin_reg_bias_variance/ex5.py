# Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     linearRegCostFunction.m
#     learningCurve.m
#     validationCurve.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

import numpy
import scipy.io
import matplotlib
import matplotlib.pyplot as plt

import rlr_bv_funcs

matplotlib.rcParams.update({'font.size': 8})

# =========== Part 1: Loading and Visualizing Data =============
# Load Training Data
print("Loading and Visualizing Data...")

# You will have X, y, Xval, yval, Xtest, ytest in your environment
mat = scipy.io.loadmat("./ex5data1.mat")
x_train = mat['X']
y_train = mat['y']
x_valid = mat['Xval']
y_valid = mat['yval']
x_test = mat['Xtest']
y_test = mat['ytest']

num_train, num_features = x_train.shape
num_valid = x_valid.shape[0]
num_test = x_test.shape[0]

x_train = numpy.reshape(x_train, (num_train))
y_train = numpy.reshape(y_train, (num_train))
x_valid = numpy.reshape(x_valid, (num_valid))
y_valid = numpy.reshape(y_valid, (num_valid))
x_test = numpy.reshape(x_test, (num_test))
y_test = numpy.reshape(y_test, (num_test))

print("\nTraining data #examples:", num_train, "#features:", num_features)
print("Validation data #examples:", num_valid, "#features:", num_features)
print("Testing data #examples:", num_test, "#features:", num_features)

# =========== Part 2: Regularized Linear Regression Cost =============
reg_lambda = 1.0
theta = numpy.array([1, 1])
biased_x_train = rlr_bv_funcs.add_bias_vector(x_train)
cost_J, gradient = rlr_bv_funcs.lin_reg_cost_function(theta, biased_x_train, y_train, reg_lambda=reg_lambda)

print("\nComputed cost:", cost_J)
print("Expected cost: 303.993192")

# =========== Part 3: Regularized Linear Regression Gradient =============
print("\nComputed gradient:", gradient)
print("Expected gradient: [-15.303016; 598.250744]")

# =========== Part 4: Train Linear Regression =============
# Train linear regression with lambda = 0
reg_lambda = 0.0
theta, convergence_flag = rlr_bv_funcs.train_lin_reg(
    rlr_bv_funcs.lin_reg_cost_function, biased_x_train, y_train, reg_lambda=reg_lambda, maxiter=1000)
print("\nTheta parameters:", theta)
print("Convergence success:", convergence_flag)

fit_line = numpy.dot(biased_x_train, theta)

# Plot training data
plt.figure(dpi=120)
plt.title("Ex.5 Training data")
plt.xlabel("Change in water level (x)")
plt.ylabel("Water flowing out of the dam (y)")
plt.xlim(-60, 60)
plt.ylim(0, 40)
plt.scatter(x_train, y_train, marker='x', s=12, linewidths=0.75, color='red', label="Training data")
plt.plot(x_train, fit_line, linestyle=':', linewidth=1.5, label="Linear fit")
plt.grid(b=True, which='major', axis='both', linestyle='--', linewidth=0.5)
plt.legend(bbox_to_anchor=(1.025, 1.0), loc='upper left')
plt.tight_layout()
plt.show()

# =========== Part 5: Learning Curve for Linear Regression =============
reg_lambda = 0
biased_x_valid = rlr_bv_funcs.add_bias_vector(x_valid)
error_train, error_valid, convergence = rlr_bv_funcs.learning_curve(
    rlr_bv_funcs.lin_reg_cost_function, biased_x_train, y_train, biased_x_valid, y_valid, reg_lambda=reg_lambda)

print("")
print(convergence)

x_axis = numpy.arange(1, num_train + 1)

plt.figure(dpi=120)
plt.title("Learning curve for linear regression")
plt.xlabel("Number of training examples")
plt.ylabel("Error")
plt.xlim(0, 13)
plt.ylim(0, 150)
plt.plot(x_axis, error_train, linewidth=1.0, color='red', marker='o', markersize=4, label="Training Error")
plt.plot(x_axis, error_valid, linewidth=1.0, color='blue', marker='o', markersize=4, label="Validation Error")
plt.grid(b=True, which='major', axis='both', linestyle='--', linewidth=0.5)
plt.legend(bbox_to_anchor=(1.025, 1.0), loc='upper left')
plt.tight_layout()
plt.show()

print("#Training\tTrain\t\tCross Validation")
print(" Examples\tError\t\tError")
for i in range(num_train):
    print("  \t%d\t\t%f\t%f" % (i+1, error_train[i], error_valid[i]))

# =========== Part 6: Feature Mapping for Polynomial Regression =============
poly_deg = 8

# Map X onto Polynomial Features and Normalize
x_train_poly = rlr_bv_funcs.poly_features(x_train, poly_deg)
x_train_poly, x_train_mean, x_train_std = rlr_bv_funcs.feature_normalize(x_train_poly, num_train)
x_train_poly = rlr_bv_funcs.add_bias_vector(x_train_poly)

# Map X_poly_val and normalize (using mu and sigma)
x_valid_poly = rlr_bv_funcs.poly_features(x_valid, poly_deg)
x_valid_poly = (x_valid_poly - x_train_mean)/x_train_std
x_valid_poly = rlr_bv_funcs.add_bias_vector(x_valid_poly)

# Map X_poly_test and normalize (using mu and sigma)
x_test_poly = rlr_bv_funcs.poly_features(x_test, poly_deg)
x_test_poly = (x_test_poly - x_train_mean)/x_train_std
x_test_poly = rlr_bv_funcs.add_bias_vector(x_test_poly)

print("\nNormalized training example 1:")
print(x_train_poly[0, :])

# =========== Part 7: Learning Curve for Polynomial Regression =============
reg_lambda = 1e2
poly_deg = 8

theta, convergence_flag = rlr_bv_funcs.train_lin_reg(
    rlr_bv_funcs.lin_reg_cost_function, x_train_poly, y_train, reg_lambda=reg_lambda, maxiter=None)


fit_line = numpy.dot(x_train_poly[:, :2], theta[:2])
min_x = numpy.min(x_train)
max_x = numpy.max(x_train)

# Plot training data and fit
plt.figure(dpi=120)
plt.title("Polynomial Regression Fit (lambda = %.2f)" % reg_lambda)
plt.xlabel("Change in water level (x)")
plt.ylabel("Water flowing out of the dam (y)")
# plt.xlim(0, 13)
# plt.ylim(-20, 50)
plt.scatter(x_train, y_train, marker='x', s=12, linewidths=0.75,
            color='red', label="Training data normalized")
# plt.plot(x_train_norm[:, 1], fit_line, linestyle=':', linewidth=1.5, label="Linear fit")
rlr_bv_funcs.plot_fit(rlr_bv_funcs.poly_features, min_x, max_x, x_train_mean, x_train_std, theta, poly_deg)
plt.grid(b=True, which='major', axis='both', linestyle='--', linewidth=0.5)
plt.legend(bbox_to_anchor=(1.025, 1.0), loc='upper left')
plt.tight_layout()
plt.show()


error_train, error_valid, convergence = rlr_bv_funcs.learning_curve(
    rlr_bv_funcs.lin_reg_cost_function, x_train_poly, y_train, x_valid_poly, y_valid, reg_lambda=reg_lambda)

plt.figure(dpi=120)
plt.title("Learning curve for polynomial regression (lambda = %.2f)" % reg_lambda)
plt.xlabel("Number of training examples")
plt.ylabel("Error")
plt.xlim(0, 13)
plt.ylim(0, 150)
plt.plot(x_axis, error_train, linewidth=1.0, color='red', marker='o', markersize=4, label="Training Error")
plt.plot(x_axis, error_valid, linewidth=1.0, color='blue', marker='o', markersize=4, label="Validation Error")
plt.grid(b=True, which='major', axis='both', linestyle='--', linewidth=0.5)
plt.legend(bbox_to_anchor=(1.025, 1.0), loc='upper left')
plt.tight_layout()
plt.show()

print("\nPolynomial Regression (lambda = %.2f)\n" % reg_lambda)
print("#Training\tTrain\tValidation")
print(" Examples\tError\tError")
for idx in range(num_train):
    print('  \t%d\t\t%f\t%f' % (idx+1, error_train[idx], error_valid[idx]))

# =========== Part 8: Validation for Selecting Lambda =============
lambda_vec, error_train, error_valid, convergence = rlr_bv_funcs.validation_curve(
    rlr_bv_funcs.lin_reg_cost_function, x_train_poly, y_train, x_valid_poly, y_valid)


plt.figure(dpi=120)
plt.title("Validation curve for polynomial regression")
plt.xlabel("Regularization parameter")
plt.ylabel("Error")
plt.xlim(-2, 13)
plt.ylim(0, 50)
plt.plot(lambda_vec, error_train, linewidth=1.0, color='red', marker='o', markersize=4, label="Training Error")
plt.plot(lambda_vec, error_valid, linewidth=1.0, color='blue', marker='o', markersize=4, label="Validation Error")
plt.grid(b=True, which='major', axis='both', linestyle='--', linewidth=0.5)
plt.legend(bbox_to_anchor=(1.025, 1.0), loc='upper left')
plt.tight_layout()
plt.show()

print("\n lambda\t\tTrain Error\tValidation Error")
for idx in range(len(lambda_vec)):
    print(' %f\t%f\t%f' % (lambda_vec[idx], error_train[idx], error_valid[idx]))
