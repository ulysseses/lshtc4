#distutils: language = c++
from libcpp.vector cimport vector
from types cimport * # <-- hope this works

cdef void extract_parents(char* infilename, object& Y, Family& parents_index)

cdef void parents2children(Family& parents_index, Family& children_index)

cdef class Baggage(object):
	cdef vector[size_t] starts
	cdef vector[size_t] lens
	cdef Word[:] __get_X(self, size_t x)
	cdef Label[:] __get_Y(self, size_t x)
	cdef Doc[:] __get_iidx(self, size_t x)