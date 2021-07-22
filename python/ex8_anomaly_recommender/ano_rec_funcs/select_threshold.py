import numpy


def select_threshold(yval, pval):
    """
    Find the best threshold (epsilon) to use for selecting outliers based
    on the results from a validation set and the ground truth.

    Parameters
    ----------
    yval : array_like
        The validation dataset of shape (m x n) where m is the number
        of examples an n is the number of dimensions(features).

    pval : array_like
        The ground truth labels of shape (m, ).

    Returns
    -------
    bestEpsilon : array_like
        A vector of shape (n,) corresponding to the threshold value.

    bestF1 : float
        The value for the best F1 score.

    Instructions
    ------------
    Compute the F1 score of choosing epsilon as the threshold and place the
    value in F1. The code at the end of the loop will compare the
    F1 score for this choice of epsilon and set it to be the best epsilon if
    it is better than the current choice of epsilon.

    Notes
    -----
    You can use predictions = (pval < epsilon) to get a binary vector
    of 0's and 1's of the outlier predictions
    """
    best_epsilon = 0
    best_f1 = 0
    f1 = 0

    for epsilon in numpy.linspace(1.01*min(pval), max(pval), 1000):
        predictions = (pval < epsilon)
        tp = numpy.sum((predictions == yval) & (yval == 1))
        fp = numpy.sum((predictions == 1) & (yval == 0))
        fn = numpy.sum((predictions == 0) & (yval == 1))

        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1 = 2*prec*rec/(prec + rec)

        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = epsilon

    return best_epsilon, best_f1
