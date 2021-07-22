#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Ashutosh Vyas

A function to plot the loaded data from ex1data1.txt as shown in ex1.py.

This function is used for the Part 2 of the Exercise 1: Linear Regression.
It imports the python package NumPy, and Matplotlib and them to generate a
scatter plot of the dataset.
"""

import numpy
import matplotlib.pyplot as plt


def plot_data(x_data, y_data):
    """
    Plot and display the loaded dataset.

    Returns
    -------
    Plot of the dataset.

    """
    plt.figure(dpi=120)
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.scatter(x_data, y_data, s=12, marker='x', color='red', linewidths=0.75)
    plt.grid(b=True, which='major', axis='both', linestyle='--', linewidth=0.5)
    plt.show()
    return 0


def plot_data_line(x_data, y_data, line_data):
    """
    Plot and display the fitted model to the dataset.

    Returns
    -------
    Plot of the dataset.

    """
    plt.figure(dpi=1200)
    plt.xlabel("Population of City in 10,000s")
    plt.ylabel("Profit in $10,000s")
    plt.scatter(x_data, y_data, s=12, marker='x', color='red', linewidths=0.75, label="Training data")
    plt.plot(x_data, line_data, label="Linear Regression", color='blue', linewidth=1.0)
    plt.grid(b=True, which='major', axis='both', linestyle='--', linewidth=0.5)
    plt.legend()
    plt.show()
    return 0
