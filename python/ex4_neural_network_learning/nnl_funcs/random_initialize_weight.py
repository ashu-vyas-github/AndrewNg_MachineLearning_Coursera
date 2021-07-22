import numpy


def random_initialize_weight(in_layer, out_layer, epsilon_init=0.12):

    rand_weight = numpy.zeros((out_layer, 1 + in_layer))
    rand_weight = numpy.random.rand(out_layer, 1 + in_layer) * 2 * epsilon_init - epsilon_init
    rand_weight[:, 0] = 1.0

    return rand_weight
