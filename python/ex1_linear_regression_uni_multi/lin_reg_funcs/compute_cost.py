#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Ashutosh Vyas

A function to compute cost for linear regression.

This function is used for the Part 3 of the Exercise 1: Linear Regression.
cost_J = compute_cost(x_data, y_data, theta) computes the cost of using theta
as the parameter for linear regression to fit the data points in x_data, y_data.
"""

import numpy


def compute_cost(x_data, y_data, theta, num_examples):
    """
    Compute cost for single iteration of linear regression.

    Given the hypothesis h(x) = theta0 + theta1*x = theta^transpose.x,
    this function calculates the cost incurred for the given value of theta.
    Thus computes J(theta)

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

    Returns
    -------
    cost_J : scalar
        Cost incurred as difference between hypothesis and actual value.
        Represented as sum -of squared errors

    """

    x_dot_theta = numpy.reshape(numpy.dot(x_data, theta), y_data.shape)  # calculating hypothesis
    difference = x_dot_theta - y_data  # error = hypothesis - actual
    squared = numpy.power(difference, 2.0)  # squared error
    factor = (1.0/(2.0*num_examples))
    cost_J = factor*numpy.sum(squared)  # sum of squared error
    return cost_J
