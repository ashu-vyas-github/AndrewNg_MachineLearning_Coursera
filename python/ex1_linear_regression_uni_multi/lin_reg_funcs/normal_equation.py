import numpy


def normal_equation(x_data, y_data):

    x_transpose = numpy.transpose(x_data)
    term1 = numpy.dot(x_transpose, y_data)
    term2 = numpy.dot(x_transpose, x_data)
    term3 = numpy.linalg.inv(term2)
    theta = numpy.dot(term3, term1)
    return theta
