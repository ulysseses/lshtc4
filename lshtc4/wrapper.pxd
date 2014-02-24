#distutils: language = c++
from libcpp.vector cimport vector
from lshtc4.container cimport unordered_map, unordered_set
# cimport numpy as np

# ctypedef np.uint32_t uint
from lshtc4.utils cimport uint, flt

cdef void pickleIndex(unordered_map[uint, unordered_set[uint]]& input, char* outname)
cdef void unpickleIndex(unordered_map[uint, unordered_set[uint]]& output, char* inname)
cdef void pickleVector(vector[uint]& input, char* outname)
cdef void unpickleVector(vector[uint]& output, char* inname)