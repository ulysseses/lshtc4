#distutils: language = c++
from libcpp.vector cimport vector
from types cimport * # <-- hope this works

cdef void extract_parents(char* infilename, Y, Family& parents_index)

cdef void parents2children(Family& parents_index, Family& children_index)

cdef void extract_XY(infilename, Xname, Yname, expectedlen=2000000,
	Baggage& X, Baggage& Y)

cdef void create_iidx(infilename, iidx_name, expectedlen=-1,
	Baggage& iidx)

cdef class Baggage(object):
	cdef vector[uint] starts
	cdef vector[uint] lens
	cdef Word[:] __get_X(self, size_t x)
	cdef Label[:] __get_Y(self, size_t x)
	cdef Doc[:] __get_iidx(self, size_t x)