import numpy


def poly_features(x_array, poly_deg=8):

    x_poly = numpy.zeros((x_array.shape[0], poly_deg))

    for idx in range(poly_deg):
        x_poly[:, idx] = x_array[:] ** (idx + 1)

    return x_poly
