#distutils: language = c++
from lshtc4.container cimport unordered_map
ctypedef unsigned int uint

cdef class LabelCounter(object):
	cdef unordered_map[uint, uint] cmap
	cdef unordered_map[uint, uint].iterator it
	cdef size_t size, d, total_count

cdef class WordCounter(LabelCounter):
	pass