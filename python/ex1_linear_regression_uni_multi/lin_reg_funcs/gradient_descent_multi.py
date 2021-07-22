#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Ashutosh Vyas

A function to compute gradient descent algorithm on ex1data1.txt as shown in ex1.py.

"""

import numpy

import lin_reg_funcs


def gradient_descent_multi(x_data, y_data, theta, num_examples, alpha, num_iters):
    """
    Gradient descent algorithm.

    Parameters
    ----------
    x_data : ndarray
        X 2D array dataset with appended theta0 initalized as 1.
    y_data : ndarray
        A vector of actual values of y from the dataset.
    theta : ndarray
        theta0 (y-intercept) and theta1 (slope) of the linear model.
    num_examples : scalar
        Number of training examples.
    alpha : float
        Constant learning rate alpha.
    num_iters : int
        Number of iterations to run gradient descent.

    Returns
    -------
    theta : ndarray
        1D array of optimum theta parameters.
    J_history : ndarray
        1D array of historical trend of gradient descent after every iteration.

    """
    J_history = numpy.zeros((num_iters))

    for oneiteration in range(num_iters):
        x_dot_theta = numpy.reshape(numpy.dot(x_data, theta), y_data.shape)  # calculating hypothesis
        difference = x_dot_theta - y_data  # error = hypothesis - actual
        theta = theta - (alpha/num_examples)*numpy.dot(difference, x_data)  # updating theta
        J_history[oneiteration] = lin_reg_funcs.compute_cost(x_data, y_data, theta, num_examples)  # saving cost values

    return theta, J_history
