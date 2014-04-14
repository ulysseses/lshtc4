#distutils: language = c++
from container cimport unordered_map
cimport numpy as np
from types cimport *

cdef class LabelCounter(object):
	cdef unordered_map[uint, uint] cmap
	cdef unordered_map[uint, uint].iterator it
	cdef uint total_count

cdef class WordCounter(LabelCounter):
	pass