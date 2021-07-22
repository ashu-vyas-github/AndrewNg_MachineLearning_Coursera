#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: Ashutosh Vyas

A simple warmup exercise to create an identity matrix of (5x5).

This function is used for the Part 1 of the Exercise 1: Linear Regression.
It imports the python package NumPy - Numerical Python and uses the function
'eye' to generate a 2D Identity matrix (array) of dimension (5x5).
"""

import numpy


def warmup_exercise():
    """
    Create an identity matrix of (5x5).

    Returns
    -------
    identity_matrix : ndarray
        Identity matrix of 5x5.

    """
    identity_matrix = numpy.eye(N=5, M=5, k=0, dtype="float")
    return identity_matrix
