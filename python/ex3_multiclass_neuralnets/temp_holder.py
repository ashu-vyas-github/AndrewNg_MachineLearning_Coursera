import copy
import numpy

xyz = numpy.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 6, 6, 6, 6])
abc = numpy.zeros(xyz.shape, dtype=int)
print(xyz)

onelabel_idx = numpy.where(xyz == 4)

abc[onelabel_idx] = 1

print(abc)
