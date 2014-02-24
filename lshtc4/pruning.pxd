#distutils: language = c++
from lshtc4.container cimport unordered_map
# cimport numpy as np
# ctypedef np.uint32_t uint
from lshtc4.utils cimport uint, flt

cdef class LabelCounter(object):
	cdef unordered_map[uint, uint] cmap
	cdef unordered_map[uint, uint].iterator it
	cdef size_t size, d, total_count

# cdef class WordCounter(LabelCounter):
# 	pass