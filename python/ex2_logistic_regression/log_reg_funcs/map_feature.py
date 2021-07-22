import numpy


def map_feature(X1, X2, num_examples, degree=6):
    """
    Feature mapping function to polynomial features.
    Maps the features to quadratic features.
    Returns a new df with more features, comprising of
    x1, x2, x1^2, x2^2, x1*x2, x1*x2^2, etc...
    Args:
        X1, X2: vectors of original features
        features: int, the number of initial features
        degree: int, the polynomial degree
    Returns:
        mapped_features: a matrix with the new features
    """
    mapped_features = numpy.ones((num_examples))

    for idx_i in range(1, degree+1):
        for idx_j in range(idx_i + 1):
            polynomial_features = numpy.multiply(numpy.power(X1, idx_i - idx_j), numpy.power(X2, idx_j))
            mapped_features = numpy.c_[mapped_features, polynomial_features]

    return mapped_features
